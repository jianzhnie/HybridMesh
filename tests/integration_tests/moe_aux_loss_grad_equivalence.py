"""Aux-loss backward check: token-partial reduction must not scale the gradient.

Run under torchrun with 2 ranks (from the repo root):

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/moe_aux_loss_grad_equivalence.py

``MicrobatchWiseLoadBalanceLoss`` sums its per-expert counts and score sums
over the axes that shard the token stream (a Partial -> Invariant reduction).
Upstream's semantics are all-reduce in FORWARD with an IDENTITY backward: the
reduced value is identical on every rank and each rank's partial is one summand
of it, so the gradient w.r.t. the partial is the gradient w.r.t. the sum. The
previous implementation used ``torch.distributed.nn.all_reduce``, whose
backward is a second all-reduce -- every rank's identical injected gradient was
summed again, multiplying what reaches the router by the group size.

Setup: cp=2 with the trainer's SPMD context registered, so the loss's ``cp``
reduction is a live 2-rank collective. Each rank computes scores from its own
token shard through the same router weight; the whole forward/backward is
compared against a single-process reference over the concatenated stream. With
the identity backward the gradients match the reference exactly; with an
all-reduce backward they come out doubled.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hpmesh.accelerator.spmd_context import spmd_context, spmd_mesh_group
from hpmesh.models.common.aux_loss import AuxLoss
from hpmesh.models.common.moe import MicrobatchWiseLoadBalanceLoss
from hpmesh.parallel.parallel_dims import ParallelDims

E, D, K, T = 4, 8, 2, 16
COEFF = 0.7
# fp32 reduction-order noise is O(1e-7); the doubled backward is an O(1)
# relative error on the injected part.
TOL = 1e-6


def _weights() -> torch.Tensor:
    return torch.randn(E, D, generator=torch.Generator().manual_seed(0))


def _tokens(rank: int) -> torch.Tensor:
    return torch.randn(T, D, generator=torch.Generator().manual_seed(100 + rank))


def _forward_backward(weight: torch.Tensor, x_TD: torch.Tensor) -> torch.Tensor:
    """One training-style router forward with the aux loss, then backward.

    Returns the accumulated value (the loss, since coeff/denominator are
    folded out). Mirrors the router: scores from the gate, top-k selection,
    the aux loss on the pre-topk scores with the gathered top-k scores as the
    injection carrier.
    """
    scores_TE = F.linear(x_TD, weight)
    probs_TE = F.softmax(scores_TE, dim=-1)
    topk_ids_TK = torch.topk(scores_TE.detach(), k=K, dim=-1, sorted=False).indices
    routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter(
        -1, topk_ids_TK, True
    )
    carrier_TK = probs_TE.gather(-1, topk_ids_TK)

    loss = MicrobatchWiseLoadBalanceLoss(coeff=COEFF)
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    loss(scores_TE, routing_map_TE, carrier=carrier_TK).sum().backward()
    return loss.instance_acc.clone()


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=-1, cp=2, tp=1, pp=1, ep=1, world_size=world
    )
    parallel_dims.build_mesh()

    failures: list[str] = []

    # -- single-process reference over the concatenated token stream ----------
    ref_weight = _weights().requires_grad_()
    ref_value = _forward_backward(ref_weight, torch.cat([_tokens(0), _tokens(1)]))
    ref_grad = ref_weight.grad.clone()

    # -- cp=2: the token-partial reduction is a live collective ----------------
    with spmd_context(parallel_dims):
        cp_group = spmd_mesh_group("cp")
        if cp_group is None:
            failures.append(f"rank {rank}: cp group not live inside spmd_context")

        weight = _weights().requires_grad_()
        value = _forward_backward(weight, _tokens(rank))

        # Each rank's gradient is its own shard's partial; training sums them
        # across the group (FSDP's reduce), so do the same before comparing.
        dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM, group=cp_group)

        # Direct pin of the reduction's backward: identity, not all-reduce.
        t = torch.ones(E, requires_grad=True)
        reduced = MicrobatchWiseLoadBalanceLoss(coeff=1.0)._reduce_token_partials(
            t, ("cp",)
        )
        if not torch.equal(reduced.detach(), torch.full((E,), float(world))):
            failures.append(
                f"rank {rank}: forward did not all-reduce (got {reduced.detach()})"
            )
        reduced.backward(torch.ones(E))
        if not torch.equal(t.grad, torch.ones(E)):
            failures.append(
                f"rank {rank}: backward is not identity (grad {t.grad}) -- "
                "an all-reduce backward would read as group_size * ones here"
            )

    value_diff = abs(value.item() - ref_value.item())
    if value_diff > TOL:
        failures.append(
            f"rank {rank}: loss value {value.item():.6f} vs reference "
            f"{ref_value.item():.6f}"
        )
    grad_diff = (weight.grad - ref_grad).abs().max().item()
    grad_scale = ref_grad.abs().max().item()
    if grad_diff > TOL:
        failures.append(
            f"rank {rank}: router grad diff {grad_diff:.3e} "
            f"(reference scale {grad_scale:.3e})"
        )

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"T={T} E={E} K={K} cp={world} coeff={COEFF} tol={TOL:.0e}")
        print(f"loss value: cp={value.item():.6f} ref={ref_value.item():.6f}")
        print(f"router grad max abs diff = {grad_diff:.3e} (scale {grad_scale:.3e})")
        print(f"failed ranks = {int(local_ok.item())}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "MoE aux-loss gradient equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
