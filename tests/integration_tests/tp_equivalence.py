"""TP>1 check: TP forward AND weight gradients must equal the single-rank run.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 tests/integration_tests/tp_equivalence.py

This exercises the CPU/gloo fallback path of the TP collectives (the fused
symmetric-memory ops are CUDA-only), so the math being pinned is the math both
paths implement:

* forward: each rank enters with only its half of the sequence (the sequence-
  parallelism premise), the fused-GEMM all-gather reassembles it inside the
  projections, and the local logits shard must equal the single-card logits at
  those rows.
* backward: the gradient of every TP-sharded weight must equal the matching
  shard of the single-card gradient. This is the regression pin for the
  missing-SP-premise bug: when every rank held the full sequence, the gathered
  GEMM saw tp copies of it and each weight gradient came out tp times too
  large -- exact only once the input is genuinely sequence-sharded.

Both sides are built from the same seed, so the TP weight shards are literal
slices of the reference weights -- which also gives the non-vacuity check: the
two ranks must hold *different* slices.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hpmesh.mesh import build_mesh
from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.parallel.tensor_parallel.tp import (
    ColwiseLinear,
    ColwiseLinearNoGather,
    RowwiseLinear,
    apply_tp,
)
from hpmesh.trainer.config import ParallelConfig

SEQ = 16
VOCAB = 64

# logits travel through different op orders on the two paths (gathered GEMMs,
# per-rank attention); fp32 CPU keeps the gap at 1e-6 scale.
FWD_TOL = 1e-5
GRAD_TOL = 1e-4


def _model(seed: int = 0) -> HFTransformerModel:
    torch.manual_seed(seed)
    config = build_model_config(
        "llama",
        seq_len=SEQ,
        arch_overrides={
            "vocab_size": VOCAB,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        },
    )
    return HFTransformerModel(config)


def _batch(seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (SEQ,), generator=g)
    labels = torch.randint(0, VOCAB, (SEQ,), generator=g)
    return ids, labels


def _loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The trainer's loss: a sum over the local (sharded) tokens."""
    return F.cross_entropy(logits.float(), labels, reduction="sum")


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"
    assert SEQ % world == 0
    half = SEQ // world

    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=world, pp=1, ep=1, world_size=world
    )
    mesh = build_mesh(parallel_dims)
    cfg = ParallelConfig(tensor_parallel_size=world)

    ref = _model()
    model = apply_tp(_model(), mesh, cfg)

    ids, labels = _batch()

    # -- single-card reference ------------------------------------------------
    logits_ref = ref(ids)
    loss_ref = _loss_sum(logits_ref, labels)
    loss_ref.backward()

    # -- TP run ---------------------------------------------------------------
    inputs, labels_shard, extra = model.preprocess_inputs(
        {"input": ids, "labels": labels}, parallel_dims=parallel_dims
    )

    failures = []

    # The sequence-parallelism premise: each rank must hold only its own
    # contiguous slice of the sequence. Holding the full sequence on every
    # rank is exactly the bug that inflated weight gradients by tp.
    want_inputs = ids[rank * half : (rank + 1) * half]
    if not torch.equal(inputs, want_inputs):
        failures.append("preprocess_inputs did not TP-shard the sequence")
    if not torch.equal(labels_shard, labels[rank * half : (rank + 1) * half]):
        failures.append("preprocess_inputs did not TP-shard the labels")
    # RoPE/attention see the assembled full sequence after the in-projection
    # all-gather, so positions stay full-length.
    positions = extra.get("positions")
    if positions is None or not torch.equal(positions, torch.arange(SEQ)):
        failures.append(f"positions not full-length under TP: {positions}")

    logits_tp = model(inputs, **extra)
    _loss_sum(logits_tp, labels_shard).backward()

    want_logits = logits_ref[rank * half : (rank + 1) * half]
    fwd_diff = (logits_tp - want_logits).abs().max().item()
    if fwd_diff > FWD_TOL:
        failures.append(f"logits: {fwd_diff:.3e}")

    # -- weight layout and gradients -------------------------------------------
    grad_diffs = []
    tp_modules = [
        (path, mod)
        for path, mod in model.named_modules()
        if isinstance(mod, (ColwiseLinear, ColwiseLinearNoGather, RowwiseLinear))
    ]
    if not tp_modules:
        failures.append("apply_tp swapped no projections -- test is vacuous")

    for path, mod in tp_modules:
        ref_lin = ref.get_submodule(path)
        w_ref = ref_lin.weight
        n, k = w_ref.shape
        if isinstance(mod, (ColwiseLinear, ColwiseLinearNoGather)):
            own = w_ref[rank * n // world : (rank + 1) * n // world]
            other = w_ref[(1 - rank) * n // world : (2 - rank) * n // world]
            want_grad = ref_lin.weight.grad[rank * n // world : (rank + 1) * n // world]
        else:
            own = w_ref[:, rank * k // world : (rank + 1) * k // world]
            other = w_ref[:, (1 - rank) * k // world : (2 - rank) * k // world]
            want_grad = ref_lin.weight.grad[
                :, rank * k // world : (rank + 1) * k // world
            ]

        # The shard is a literal slice of the same initialization...
        if not torch.equal(mod.weight.detach(), own):
            failures.append(f"{path}: weight is not the rank's slice")
        # ...and not the other rank's -- otherwise the sharding did nothing.
        if torch.equal(mod.weight.detach(), other):
            failures.append(f"{path}: ranks hold identical shards (vacuous)")

        diff = (mod.weight.grad - want_grad).abs().max().item()
        grad_diffs.append(diff)
        if diff > GRAD_TOL:
            failures.append(f"{path}: weight grad diff {diff:.3e}")

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"seq={SEQ} vocab={VOCAB} tp_size={world} backend=gloo (fallback path)")
        print(f"swapped projections = {len(tp_modules)}")
        print(f"logits         max abs diff = {fwd_diff:.3e}")
        print(f"weight grads   max abs diff = {max(grad_diffs):.3e}")
        print(f"failed ranks   = {int(local_ok.item())}")
        if failures:
            for f in failures:
                print(f"  FAIL {f}")
        else:
            print("all checks passed")

    assert local_ok.item() == 0, "TP equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
