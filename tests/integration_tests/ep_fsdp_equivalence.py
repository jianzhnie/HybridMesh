"""EP+FSDP check: expert weights must shard on the sparse mesh, not the dense one.

Run under torchrun with 4 ranks (from the repo root):

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/ep_fsdp_equivalence.py

Regression pin for the missing ``moe_enabled``/``moe`` flags on the EP swap:
without them, FSDP's MoE branch never fires and the expert weights -- each
rank holding a DIFFERENT slice of the experts -- are sharded as dense
parameters over the dense DP mesh, mixing ranks of different EP coordinates
into one FSDP group. The forward then all-gathers mismatched shards and the
output is garbage, silently.

Setup: dp_shard=4, ep=2 (so efsdp=2) over a tiny offline qwen3_moe, through
the real ``parallelize_hf_transformers`` entry point. Each rank runs its own
quarter of a global batch; FSDP's gradient reduction (sum, division disabled)
turns the per-rank partials into full-batch gradients. The whole thing is
compared against a single-process reference over the full batch: loss, router
gradients, per-expert weight gradients.

Non-vacuity: the expert parameters must be DTensors on the edp (efsdp) mesh --
size 2 here -- while dense parameters sit on the size-4 dense mesh. That mesh
assignment is exactly what the missing flags broke.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers import AutoConfig

from hpmesh.models.common.moe import MoE
from hpmesh.models.hf_wrapper import HFTransformerModel
from hpmesh.parallel.collectives import dist_sum
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.parallel.parallelize_hf import parallelize_hf_transformers
from hpmesh.trainer import ParallelConfig
from hpmesh.utils.spmd_context import spmd_context

NUM_EXPERTS = 8
TOP_K = 2
DIM = 64
EXPERT_HIDDEN = 48
TOKENS_PER_RANK = 16
TOL = 1e-4


def _config() -> AutoConfig:
    # experts_implementation="eager": transformers 5.x defaults to grouped_mm,
    # which is bf16-only and undispatchable on CPU (see test_ep_swap.py).
    return AutoConfig.for_model(
        "qwen3_moe",
        vocab_size=128,
        hidden_size=DIM,
        intermediate_size=128,
        moe_intermediate_size=EXPERT_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        norm_topk_prob=True,
        max_position_embeddings=256,
        experts_implementation="eager",
    )


def _model(seed: int = 0) -> HFTransformerModel:
    """Deterministically initialized tiny qwen3_moe; identical on every rank."""
    torch.manual_seed(seed)
    return HFTransformerModel(_config()).float().eval()


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    """The global batch ``(dp, T)``; identical on every rank, row-sliced later.

    One row per rank: dp shards a batch, and each row is an independent
    sequence (RoPE makes the position values semantic, so a row's positions
    restart at 0 -- slicing one long sequence across ranks would not be dp).
    """
    ids = torch.randint(
        128, (4, TOKENS_PER_RANK), generator=torch.Generator().manual_seed(1000)
    )
    labels = torch.randint(
        128, (4, TOKENS_PER_RANK), generator=torch.Generator().manual_seed(2000)
    )
    return ids, labels


def _loss_sum(model, ids_T, labels_T) -> torch.Tensor:
    """Summed CE over one packed 1-D sequence (the wrapper's input shape)."""
    positions = torch.arange(ids_T.shape[0])
    return F.cross_entropy(model(ids_T, positions=positions), labels_T, reduction="sum")


def _full(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 4, f"this check assumes 4 ranks, got {world}"

    cfg = ParallelConfig(data_parallel_shard_size=4, expert_parallel_size=2)
    parallel_dims = ParallelDims.from_config(cfg, world)
    parallel_dims.build_mesh()
    loss_mesh = parallel_dims.get_mesh("loss")

    failures: list[str] = []

    # -- single-process full-batch reference (computed identically per rank) ---
    # The wrapper takes one packed 1-D sequence per call, so the reference runs
    # the batch's rows one at a time and sums -- the same total as one batched
    # call, and exactly what dp sharding splits.
    ids_all, labels_all = _data()
    ref = _model()
    with spmd_context(None):
        ref_losses = [
            _loss_sum(ref, ids_all[r], labels_all[r]) for r in range(ids_all.shape[0])
        ]
    sum(ref_losses).backward()
    ref_loss = sum(loss.item() for loss in ref_losses)

    # -- ep=2 + dp_shard=4 through the real orchestration entry point ----------
    model = parallelize_hf_transformers(
        _model(),
        cfg=cfg,
        mesh=parallel_dims.spmd_dense_mesh(),
        parallel_dims=parallel_dims,
        device=torch.device("cpu"),
    )

    with spmd_context(parallel_dims):
        loss_sum = _loss_sum(model, ids_all[rank], labels_all[rank])
    loss_sum.backward()

    got_loss = dist_sum(loss_sum, loss_mesh)
    if abs(got_loss - ref_loss) > TOL:
        failures.append(f"rank {rank}: loss {got_loss:.6f} vs reference {ref_loss:.6f}")

    num_local = NUM_EXPERTS // cfg.ep
    # Sparse mesh layout (dp_replicate, efsdp, ep): ep coordinate is fastest.
    ep_rank = rank % cfg.ep
    lo_e = ep_rank * num_local
    for layer_idx, (layer, ref_layer) in enumerate(
        zip(model.layers, ref.layers, strict=True)
    ):
        moe = layer.mlp
        if not isinstance(moe, MoE):
            failures.append(f"rank {rank} layer {layer_idx}: mlp was not swapped")
            continue

        # Non-vacuity, the BUG-1 pin: the flags FSDP's MoE branch reads, and
        # the mesh each parameter actually landed on.
        if getattr(layer, "moe_enabled", False) is not True or layer.moe is not moe:
            failures.append(
                f"rank {rank} layer {layer_idx}: moe_enabled/moe flags missing"
            )
        grouped = moe.routed_experts.inner_experts
        w1 = grouped.w1_EFD
        gate_w = moe.router.gate.weight
        if not isinstance(w1, DTensor) or w1.device_mesh.size() != 2:
            failures.append(
                f"rank {rank} layer {layer_idx}: expert weights not on the "
                f"efsdp mesh (got {type(w1).__name__}, mesh "
                f"{None if not isinstance(w1, DTensor) else w1.device_mesh.size()})"
            )
        if not isinstance(gate_w, DTensor) or gate_w.device_mesh.size() != 4:
            gate_mesh = (
                gate_w.device_mesh.size() if isinstance(gate_w, DTensor) else None
            )
            failures.append(
                f"rank {rank} layer {layer_idx}: router weight not on the "
                f"dense dp mesh (mesh {gate_mesh})"
            )

        ref_mlp = ref_layer.mlp
        gate_diff = (_full(gate_w.grad) - ref_mlp.gate.weight.grad).abs().max().item()
        if gate_diff > TOL:
            failures.append(
                f"rank {rank} layer {layer_idx}: router grad diff {gate_diff:.3e}"
            )

        gate_up_grad = ref_mlp.experts.gate_up_proj.grad  # (E, 2F, D)
        down_grad = ref_mlp.experts.down_proj.grad  # (E, D, F)
        checks = [
            (
                "w1",
                _full(grouped.w1_EFD.grad),
                gate_up_grad[lo_e : lo_e + num_local, :EXPERT_HIDDEN],
            ),
            (
                "w3",
                _full(grouped.w3_EFD.grad),
                gate_up_grad[lo_e : lo_e + num_local, EXPERT_HIDDEN:],
            ),
            ("w2", _full(grouped.w2_EDF.grad), down_grad[lo_e : lo_e + num_local]),
        ]
        for name, got, want in checks:
            diff = (got - want).abs().max().item()
            if diff > TOL:
                failures.append(
                    f"rank {rank} layer {layer_idx}: {name} grad diff {diff:.3e}"
                )

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"tokens/rank={TOKENS_PER_RANK} experts={NUM_EXPERTS} "
            f"ep=2 dp_shard=4 (efsdp=2) tol={TOL:.0e}"
        )
        print(f"loss: ep+fsdp={got_loss:.6f} ref={ref_loss:.6f}")
        print(f"failed ranks = {int(local_ok.item())}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "EP+FSDP equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
