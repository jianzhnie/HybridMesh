"""AC equivalence: training with activation checkpointing matches without.

Run under torchrun with 2 ranks (from the repo root):

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/ac_equivalence.py

A tiny decoder is trained through the real ``apply_ac`` + ``apply_fsdp`` entry
points in ddp mode (``dp_replicate=2``), with AC ON, on a *different* batch per
rank. Both AC modes -- ``full`` and ``selective`` -- are run, each against the
same reference: the model trained single-rank on the union of both batches with
AC OFF. Step-0 gradients, the loss trajectory, and final parameters must match
within fp32 sum-ordering noise: AC is a memory trade, not a numerics change
(``preserve_rng_state=True`` makes the backward-time recompute see the
forward's RNG state).

Comparing both modes to the AC-OFF reference rather than to each other is
deliberate: two modes that recomputed the same wrong tensor would agree with
each other and still be wrong.

Gradient scaling note mirrors fsdp_mesh_equivalence: FSDP's automatic gradient
division is disabled (the trainer scales by global token count itself), so
both runs divide each rank-local loss by the global token count before
backward.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from hpmesh.parallel.activation_checkpoint import apply_ac
from hpmesh.parallel.collectives import dist_sum
from hpmesh.parallel.fully_shard.fsdp_wrap import apply_fsdp
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.trainer import ParallelConfig
from hpmesh.trainer.config import SelectiveACConfig

AC_MODES = ("full", "selective")

VOCAB = 32
HIDDEN = 16
NUM_LAYERS = 2
SEQ = 64
STEPS = 4
LR = 0.5

# fp32 sum-ordering noise over a handful of steps is O(1e-6); AC must not move
# the numbers at all beyond that.
TOL = 1e-4


class TinyTokenDecoder(nn.Module):
    """Minimal per-token decoder exposing the surface apply_ac/apply_fsdp read."""

    def __init__(self) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList(
            [nn.Linear(HIDDEN, HIDDEN) for _ in range(NUM_LAYERS)]
        )
        self.norm = nn.LayerNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.enable_weight_tying = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(input_ids)
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return self.lm_head(self.norm(h))


def _build_model(seed: int = 0) -> TinyTokenDecoder:
    """Deterministically initialized decoder; identical on every rank."""
    torch.manual_seed(seed)
    return TinyTokenDecoder()


def _data(step: int, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank ``rank``'s batch for ``step``; deterministic and rank-distinct."""
    g = torch.Generator().manual_seed(1000 * rank + step)
    ids = torch.randint(VOCAB, (SEQ,), generator=g)
    labels = torch.randint(VOCAB, (SEQ,), generator=g)
    return ids, labels


def _loss_sum(
    model: nn.Module, ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(model(ids), labels, reduction="sum")


def _plain_name(name: str) -> str:
    """Strip the wrapper level apply_ac inserts into parameter FQNs."""
    return name.replace("._checkpoint_wrapped_module", "")


def _full(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    cfg = ParallelConfig(data_parallel_replicate_size=2, backend="gloo")
    parallel_dims = ParallelDims.from_config(cfg, world)
    parallel_dims.build_mesh()
    loss_mesh = parallel_dims.get_mesh("loss")

    global_tokens = world * SEQ
    failures: list[str] = []

    # -- single-rank reference over the union of both ranks' batches, AC OFF --
    ref = _build_model()
    ref_opt = torch.optim.SGD(ref.parameters(), lr=LR)
    ref_losses: list[float] = []
    ref_step0_grads: dict[str, torch.Tensor] = {}
    for step in range(STEPS):
        ref_opt.zero_grad()
        loss = (
            sum(_loss_sum(ref, *_data(step, r)) for r in range(world)) / global_tokens
        )
        loss.backward()
        if step == 0:
            ref_step0_grads = {
                name: p.grad.clone() for name, p in ref.named_parameters()
            }
        ref_opt.step()
        ref_losses.append(loss.item())

    # -- distributed runs, AC ON, through the real entry points --------------
    # Each mode runs independently against the same AC-OFF reference above, so
    # one mode's result cannot mask the other's (see the module docstring).
    results: dict[str, dict] = {}
    for mode in AC_MODES:
        model = apply_ac(
            _build_model(),
            mode,
            selective=SelectiveACConfig() if mode == "selective" else None,
        )
        model = apply_fsdp(model, None, cfg, parallel_dims)

        opt = torch.optim.SGD(model.parameters(), lr=LR)
        losses: list[float] = []
        for step in range(STEPS):
            ids, labels = _data(step, rank)
            opt.zero_grad()
            loss = _loss_sum(model, ids, labels) / global_tokens
            loss.backward()
            if step == 0:
                for name, p in model.named_parameters():
                    grad_diff = (
                        (_full(p.grad) - ref_step0_grads[_plain_name(name)])
                        .abs()
                        .max()
                        .item()
                    )
                    if grad_diff > TOL:
                        failures.append(
                            f"[{mode}] step0 grad {name}: diff {grad_diff:.3e}"
                        )
            opt.step()
            losses.append(dist_sum(loss.detach(), loss_mesh))

        for step, (got, want) in enumerate(zip(losses, ref_losses, strict=True)):
            if abs(got - want) > TOL:
                failures.append(
                    f"[{mode}] step{step} loss: got {got:.6f}, want {want:.6f}"
                )

        param_diff = max(
            (_full(p).detach() - rp.detach()).abs().max().item()
            for (name, p), (_, rp) in zip(
                sorted(model.named_parameters(), key=lambda kv: _plain_name(kv[0])),
                sorted(ref.named_parameters(), key=lambda kv: kv[0]),
                strict=True,
            )
        )
        if param_diff > TOL:
            failures.append(f"[{mode}] final params: max abs diff {param_diff:.3e}")
        results[mode] = {"losses": losses, "param_diff": param_diff}

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"steps={STEPS} lr={LR} tol={TOL:.0e} (ac=on vs ac=off reference)")
        for mode, res in results.items():
            print(f"ac={mode}:")
            for step, (got, want) in enumerate(
                zip(res["losses"], ref_losses, strict=True)
            ):
                print(f"  step{step}: loss ac={got:.6f} ref={want:.6f}")
            print(f"  final param max abs diff = {res['param_diff']:.3e}")
        print(f"failed ranks = {int(local_ok.item())}")
        if failures:
            for f in failures:
                print(f"  FAIL {f}")
        else:
            print("all checks passed")

    assert local_ok.item() == 0, "AC equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
