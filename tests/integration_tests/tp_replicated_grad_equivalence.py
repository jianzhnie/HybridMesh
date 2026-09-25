"""TP>1 replicated-parameter check: non-sharded grads must match single-rank.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/tp_replicated_grad_equivalence.py

``tp_equivalence.py`` pins the SHARDED projections' forward and weight grads.
This pins the other half of the seam the sequence-parallelism premise opens:
parameters TP does not shard (the token embedding, the RMSNorms, the LM head)
see only this rank's ``T / tp`` token shard per forward, so their gradients
are partial and nothing inside the TP modules reduces them. The trainer's
``_allreduce_replicated_tp_grads`` (called after backward, before clipping)
sums them across the TP group.

The check trains both sides with SGD for a few steps and compares, per step:

* the TP rank's local loss shard against the reference's matching slice (the
  loss trajectory), and
* every replicated parameter's gradient against the single-rank gradient.

Non-vacuity: before the all-reduce, a rank's replicated gradient covers half
the tokens and must differ MATERIALLY from the reference's full-batch gradient
-- a run that skipped the reduction would keep exactly that partial value.

Everything runs in fp32 on CPU/gloo (the fallback path of the TP collectives).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hpmesh.accelerator.mesh import build_mesh
from hpmesh.config import ParallelConfig
from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.parallel.tensor_parallel.tp import (
    ColwiseLinear,
    ColwiseLinearNoGather,
    RowwiseLinear,
    apply_tp,
)
from hpmesh.trainer.trainer import Trainer

SEQ = 16
VOCAB = 64
STEPS = 3
LR = 0.05

TOL = 1e-5


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


def _batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (SEQ,), generator=g)
    labels = torch.randint(0, VOCAB, (SEQ,), generator=g)
    return ids, labels


def _loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), labels, reduction="sum")


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"
    half = SEQ // world

    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=world, pp=1, ep=1, world_size=world
    )
    mesh = build_mesh(parallel_dims)
    cfg = ParallelConfig(tensor_parallel_size=world)

    ref = _model()
    model = apply_tp(_model(), mesh, cfg)

    # A bare Trainer double carrying exactly what the method under test reads.
    trainer = Trainer.__new__(Trainer)
    trainer.parallel_dims = parallel_dims
    trainer.model_parts = [model]

    sharded_ids = {
        id(mod.weight)
        for mod in model.modules()
        if isinstance(mod, ColwiseLinear | ColwiseLinearNoGather | RowwiseLinear)
    }
    assert sharded_ids, "apply_tp swapped no projections -- test is vacuous"
    replicated = {
        name: p for name, p in model.named_parameters() if id(p) not in sharded_ids
    }
    assert any("embed" in name for name in replicated)
    ref_params = dict(ref.named_parameters())

    ref_opt = torch.optim.SGD(ref.parameters(), lr=LR)
    tp_opt = torch.optim.SGD(model.parameters(), lr=LR)

    failures: list[str] = []
    max_loss_diff = 0.0
    max_grad_diff = 0.0
    max_prereduce_diff = 0.0

    for step in range(STEPS):
        ids, labels = _batch(seed=step + 1)

        ref_opt.zero_grad(set_to_none=True)
        logits_ref = ref(ids)
        _loss_sum(logits_ref, labels).backward()

        inputs, labels_sh, extra = model.preprocess_inputs(
            {"input": ids, "labels": labels}, parallel_dims=parallel_dims
        )
        tp_opt.zero_grad(set_to_none=True)
        logits_tp = model(inputs, **extra)
        loss_tp = _loss_sum(logits_tp, labels_sh)
        loss_tp.backward()

        # The partial value the reduction must repair: each rank's replicated
        # gradient covers only its own token shard.
        pre = replicated["model.model.embed_tokens.weight"].grad.clone()

        trainer._allreduce_replicated_tp_grads()

        want_loss = _loss_sum(
            logits_ref[rank * half : (rank + 1) * half],
            labels[rank * half : (rank + 1) * half],
        )
        loss_diff = (loss_tp - want_loss).abs().item()
        max_loss_diff = max(max_loss_diff, loss_diff)
        if loss_diff > TOL:
            failures.append(f"step {step}: shard loss diff {loss_diff:.3e}")

        ref_embed_grad = ref_params["model.model.embed_tokens.weight"].grad
        pre_diff = (pre - ref_embed_grad).abs().max().item()
        max_prereduce_diff = max(max_prereduce_diff, pre_diff)

        for name, p in replicated.items():
            grad_diff = (p.grad - ref_params[name].grad).abs().max().item()
            max_grad_diff = max(max_grad_diff, grad_diff)
            if grad_diff > TOL:
                failures.append(
                    f"step {step}: replicated grad {name} diff {grad_diff:.3e}"
                )

        ref_opt.step()
        tp_opt.step()

    # After identical updates, the replicated weights must still be one model.
    for name, p in replicated.items():
        w_diff = (p.detach() - ref_params[name].detach()).abs().max().item()
        if w_diff > TOL:
            failures.append(f"replicated weight {name} drifted: {w_diff:.3e}")

    # Non-vacuity: the un-reduced partial gradient is wrong by an O(1) margin.
    if max_prereduce_diff < 1e-3:
        failures.append(
            f"pre-reduce partial grad matches the reference "
            f"({max_prereduce_diff:.3e}) -- the reduction repaired nothing"
        )

    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"seq={SEQ} vocab={VOCAB} tp_size={world} steps={STEPS} backend=gloo")
        print(f"replicated params  = {len(replicated)}")
        print(f"shard loss     max abs diff = {max_loss_diff:.3e}")
        print(f"replicated grad  max abs diff = {max_grad_diff:.3e}")
        print(f"pre-reduce grad  max abs diff = {max_prereduce_diff:.3e} (non-vacuity)")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "TP replicated-gradient equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
