"""CP gradient equivalence: pure CP must reduce gradients across the CP group.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 tests/cp_grad_equivalence.py

With ``dp_shard=1, cp=2`` every rank computes the loss over its own sequence
shard, so its parameter gradients are partial: they only become the global
(full-sequence) gradients once reduced across the CP group. hpmesh wires that
reduction through FSDP -- ``resolve_fsdp_mesh`` puts ``cp`` on the shard axis,
and ``apply_fsdp`` must therefore run whenever CP is enabled, not only when
``dp_shard > 1``. This test pins that contract end to end:

* a deterministic tiny per-token decoder is trained for a few SGD steps under
  pure CP (``dp_shard=1, cp=2``) with real CP input sharding
  (``shard_batch_for_cp``), and its loss trajectory, first-step gradients, and
  final parameters must match a single-rank full-sequence reference trained on
  the same data;
* non-vacuity: an identical twin trained WITHOUT any cross-CP communication
  must produce rank-local gradients that differ materially between ranks and
  from the global gradient -- so a wiring that silently skipped the CP gradient
  reduction could not pass the equivalence checks above.

The model is a per-token decoder (embedding -> linear blocks -> norm -> head,
no attention) so that sharding the sequence is arithmetically exact and the
comparison is pure wiring, not kernel noise. Everything runs in float32 with a
tolerance far above sum-ordering noise and far below the O(1) error a missing
reduction introduces (each rank would see only half the gradient).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from hpmesh.parallel.collectives import dist_sum
from hpmesh.parallel.context_parallel import shard_batch_for_cp
from hpmesh.parallel.fully_shard.fsdp_wrap import apply_fsdp
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.trainer import HybridMeshConfig, ParallelConfig, TrainingConfig

VOCAB = 32
HIDDEN = 16
NUM_LAYERS = 2
SEQ = 64  # divisible by cp=2 under the contiguous (no load balancer) split
STEPS = 4
LR = 0.5

# fp32 sum-ordering noise over a handful of steps is O(1e-6); a missing CP
# gradient reduction is an O(1) error (each rank keeps only half the gradient).
TOL = 1e-4
# The no-sync twin's rank-local gradients must differ from the global ones by
# much more than noise, otherwise the equivalence checks could pass vacuously.
NONVACUITY = 1e-3


class TinyTokenDecoder(nn.Module):
    """Minimal per-token decoder exposing the surface FSDP reads off a model.

    Every position is processed independently, so sharding the sequence across
    CP ranks is exact: the token-loss sums of the shards add up to the
    full-sequence sum. The five attributes are the contract
    ``apply_fsdp_to_decoder`` requires (the same ones ``HFTransformerModel``
    exposes).
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


def _data(step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The step's full-sequence batch; identical on every rank."""
    g = torch.Generator().manual_seed(100 + step)
    ids = torch.randint(VOCAB, (SEQ,), generator=g)
    labels = torch.randint(VOCAB, (SEQ,), generator=g)
    positions = torch.arange(SEQ)
    return ids, labels, positions


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
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    cfg = HybridMeshConfig(
        parallel=ParallelConfig(context_parallel_size=2, backend="gloo"),
        training=TrainingConfig(max_seq_len=SEQ, steps=STEPS),
    )
    parallel_dims = ParallelDims.from_config(cfg.parallel, world)
    parallel_dims.build_mesh()
    cp_mesh = parallel_dims.get_mesh("cp")
    loss_mesh = parallel_dims.get_mesh("loss")

    failures: list[str] = []

    # -- single-rank full-sequence reference (computed identically per rank) --
    ref = _build_model()
    ref_opt = torch.optim.SGD(ref.parameters(), lr=LR)
    ref_losses: list[float] = []
    ref_step0_grads: dict[str, torch.Tensor] = {}
    for step in range(STEPS):
        ids, labels, _ = _data(step)
        ref_opt.zero_grad()
        loss_sum = _loss_sum(ref, ids, labels)
        (loss_sum / SEQ).backward()
        if step == 0:
            ref_step0_grads = {
                name: p.grad.clone() for name, p in ref.named_parameters()
            }
        ref_opt.step()
        ref_losses.append(loss_sum.item() / SEQ)

    # -- pure-CP run through the real apply_fsdp entry point -------------------
    model = apply_fsdp(_build_model(), None, cfg.parallel, parallel_dims)
    if not isinstance(model, FSDPModule):
        failures.append("apply_fsdp left the model unsharded under pure CP")

    cp_opt = torch.optim.SGD(model.parameters(), lr=LR)
    cp_losses: list[float] = []
    for step in range(STEPS):
        ids, labels, positions = _data(step)
        ids_sh, labels_sh, _ = shard_batch_for_cp(ids, labels, positions, cp_mesh)
        cp_opt.zero_grad()
        loss_sum = _loss_sum(model, ids_sh, labels_sh)
        (loss_sum / SEQ).backward()
        if step == 0:
            # The reduced (global) gradient must equal the reference's.
            for name, p in model.named_parameters():
                grad_diff = (_full(p.grad) - ref_step0_grads[name]).abs().max().item()
                if grad_diff > TOL:
                    failures.append(f"step0 grad {name}: diff {grad_diff:.3e}")
        cp_opt.step()
        # The trainer's loss reporting: sum over the dp*cp loss mesh.
        cp_losses.append(dist_sum(loss_sum / SEQ, loss_mesh))

    for step, (got, want) in enumerate(zip(cp_losses, ref_losses, strict=True)):
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

    # -- non-vacuity: without cross-CP reduction the ranks diverge -------------
    twin = _build_model()
    ids, labels, positions = _data(0)
    ids_sh, labels_sh, _ = shard_batch_for_cp(ids, labels, positions, cp_mesh)
    (_loss_sum(twin, ids_sh, labels_sh) / SEQ).backward()
    # The twin's rank-local partial gradient must differ materially both from
    # the global gradient and across ranks -- the two failures a missing CP
    # gradient reduction would hide behind.
    probe = twin.layers[0].weight.grad
    local_vs_global = (probe - ref_step0_grads["layers.0.weight"]).abs().max()
    gaps = [torch.zeros_like(local_vs_global) for _ in range(world)]
    dist.all_gather(gaps, local_vs_global, group=cp_mesh.get_group())
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
        print(f"seq={SEQ} cp_size={world} steps={STEPS} lr={LR} tol={TOL:.0e}")
        for step, (got, want) in enumerate(zip(cp_losses, ref_losses, strict=True)):
            print(f"step{step}: loss cp={got:.6f} ref={want:.6f}")
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

    assert local_ok.item() == 0, "CP gradient equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
