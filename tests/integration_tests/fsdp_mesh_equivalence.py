"""FSDP mesh equivalence: replicate-only and shard-only meshes must both train.

Run under torchrun with 2 ranks, one mode per invocation (from the repo root):

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/fsdp_mesh_equivalence.py ddp
    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/fsdp_mesh_equivalence.py shard

Two regressions are pinned here, both rooted in how the FSDP mesh is chosen:

* ``ddp`` (``dp_replicate=2, dp_shard=1, cp=1``): FSDP used to early-return
  because ``fsdp_enabled`` only looked at ``dp_shard``/``cp``, leaving pure
  replicas with no gradient reduction at all. torchtitan applies FSDP
  unconditionally in this configuration -- the all-gather is a no-op and only
  the gradient all-reduce across ``dp_replicate`` remains. The test trains a
  tiny decoder on a *different* batch per rank through the real
  ``apply_fsdp`` entry point and requires step-0 gradients, the loss
  trajectory, and final parameters to match a single-rank reference trained
  on the union of both batches.
* ``shard`` (``dp_shard=2``): the pre-existing pure-FSDP path must not
  regress under the rebuilt submesh. Same equivalence checks, plus a proof
  that parameters are actually sharded (DTensor with a smaller local shard),
  not merely replicated.

Non-vacuity: an identical twin trained WITHOUT any cross-rank communication
must produce rank-local gradients that differ materially between ranks -- so
a wiring that silently skipped the gradient reduction could not pass the
equivalence checks above.

Gradient scaling note: ``apply_fsdp_to_decoder`` disables FSDP's automatic
gradient division (the trainer scales by global token count itself), so
gradients are SUMMED across ranks. Both the distributed runs and the
reference therefore divide each rank-local loss by the global token count
(2 * SEQ) before backward, mirroring the trainer.

Everything runs in float32 with a tolerance far above sum-ordering noise and
far below the O(1) error a missing reduction introduces (each replica would
keep only its own batch's gradient).
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from hpmesh.parallel.collectives import dist_sum
from hpmesh.parallel.fully_shard.fsdp_wrap import apply_fsdp
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.trainer import ParallelConfig

VOCAB = 32
HIDDEN = 16
NUM_LAYERS = 2
SEQ = 64
STEPS = 4
LR = 0.5

# fp32 sum-ordering noise over a handful of steps is O(1e-6); a missing
# gradient reduction is an O(1) error (each rank keeps only its own batch).
TOL = 1e-4
# The no-sync twin's rank-local gradients must differ across ranks by much
# more than noise, otherwise the equivalence checks could pass vacuously.
NONVACUITY = 1e-3

MODES = ("ddp", "shard")


class TinyTokenDecoder(nn.Module):
    """Minimal per-token decoder exposing the surface FSDP reads off a model.

    The five attributes are the contract ``apply_fsdp_to_decoder`` requires
    (the same ones ``HFTransformerModel`` exposes).
    """

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


def _full(t: torch.Tensor) -> torch.Tensor:
    """Global tensor: redistribute where FSDP sharded, identity elsewhere."""
    return t.full_tensor() if isinstance(t, DTensor) else t


def _full_param(p: torch.Tensor) -> torch.Tensor:
    return _full(p).detach().clone()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "ddp"
    assert mode in MODES, f"unknown mode {mode!r}; expected one of {MODES}"

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    if mode == "ddp":
        cfg = ParallelConfig(data_parallel_replicate_size=2)
    else:
        cfg = ParallelConfig(data_parallel_shard_size=2)
    parallel_dims = ParallelDims.from_config(cfg, world)
    parallel_dims.build_mesh()
    loss_mesh = parallel_dims.get_mesh("loss")

    global_tokens = world * SEQ
    failures: list[str] = []

    # -- single-rank reference over the union of both ranks' batches --------
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

    # -- distributed run through the real apply_fsdp entry point ------------
    model = apply_fsdp(_build_model(), cfg, parallel_dims)
    if not isinstance(model, FSDPModule):
        failures.append(f"apply_fsdp left the model unwrapped in {mode} mode")

    if mode == "shard":
        # Proof of actual sharding, not replication: the embedding weight is
        # a DTensor whose local shard is half the global rows.
        w = model.tok_embeddings.weight
        if not isinstance(w, DTensor) or w.to_local().shape[0] != VOCAB // 2:
            failures.append(
                "dp_shard=2 did not shard tok_embeddings.weight "
                f"(type={type(w).__name__}, "
                f"local={w.to_local().shape if isinstance(w, DTensor) else w.shape})"
            )

    opt = torch.optim.SGD(model.parameters(), lr=LR)
    losses: list[float] = []
    for step in range(STEPS):
        ids, labels = _data(step, rank)
        opt.zero_grad()
        loss = _loss_sum(model, ids, labels) / global_tokens
        loss.backward()
        if step == 0:
            # The reduced (global) gradient must equal the reference's.
            for name, p in model.named_parameters():
                grad_diff = (_full(p.grad) - ref_step0_grads[name]).abs().max().item()
                if grad_diff > TOL:
                    failures.append(f"step0 grad {name}: diff {grad_diff:.3e}")
        opt.step()
        # The trainer's loss reporting: sum over the dp loss mesh.
        losses.append(dist_sum(loss.detach(), loss_mesh))

    for step, (got, want) in enumerate(zip(losses, ref_losses, strict=True)):
        if abs(got - want) > TOL:
            failures.append(f"step{step} loss: got {got:.6f}, want {want:.6f}")

    param_diff = max(
        (_full_param(p) - _full_param(rp)).abs().max().item()
        for (name, p), (_, rp) in zip(
            model.named_parameters(), ref.named_parameters(), strict=True
        )
    )
    if param_diff > TOL:
        failures.append(f"final params: max abs diff {param_diff:.3e}")

    # -- non-vacuity: without cross-rank reduction the replicas diverge -----
    twin = _build_model()
    ids, labels = _data(0, rank)
    (_loss_sum(twin, ids, labels) / global_tokens).backward()
    probe = twin.layers[0].weight.grad
    local_gap = (probe - ref_step0_grads["layers.0.weight"]).abs().max()
    gaps = [torch.zeros_like(local_gap) for _ in range(world)]
    dist.all_gather(gaps, local_gap)
    if min(g.item() for g in gaps) < NONVACUITY:
        failures.append(
            f"non-vacuity: local grad matches global ({[g.item() for g in gaps]})"
        )
    across_ranks = abs(gaps[0].item() - gaps[1].item())
    if across_ranks < NONVACUITY:
        failures.append(f"non-vacuity: rank-local grads agree ({across_ranks:.3e})")

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"mode={mode} steps={STEPS} lr={LR} tol={TOL:.0e}")
        for step, (got, want) in enumerate(zip(losses, ref_losses, strict=True)):
            print(f"step{step}: loss dist={got:.6f} ref={want:.6f}")
        print(f"final param max abs diff = {param_diff:.3e}")
        print(
            f"non-vacuity: local-vs-global grad gap per rank = "
            f"{[f'{g.item():.3e}' for g in gaps]}"
        )
        print(f"failed ranks = {int(local_ok.item())}")
        if failures:
            for f in failures:
                print(f"  FAIL {f}")
        else:
            print("all checks passed")

    assert local_ok.item() == 0, f"FSDP mesh equivalence check failed ({mode})"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
