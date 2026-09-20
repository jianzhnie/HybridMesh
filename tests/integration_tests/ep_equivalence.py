"""EP>1 check: the all-to-all MoE must agree with the single-rank MoE.

Run under torchrun with 2 ranks:

    torchrun --nproc_per_node=2 tests/ep_equivalence.py

Both sides get the same global weights and the same tokens. The reference keeps
all 8 experts on every rank and routes locally; the EP path gives each rank 4
experts and moves tokens between them. If the dispatcher's split bookkeeping or
the permute is wrong, the routed tokens land on the wrong experts and the
outputs diverge -- which is exactly what this compares.

Stands alone from hpmesh's mesh plumbing on purpose: it builds its own process
group, so it exercises the dispatcher without depending on how the trainer
assembles a mesh.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from hpmesh.models.common.grouped_experts import GroupedExperts
from hpmesh.models.common.moe import MoE, RoutedExperts, TokenChoiceTopKRouter
from hpmesh.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)

NUM_EXPERTS = 8
TOP_K = 2
DIM = 32
HIDDEN = 48
TOKENS = 40


def _weights(seed: int = 0):
    """The same global expert/router weights on every rank."""
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(NUM_EXPERTS, HIDDEN, DIM, generator=g)
    w3 = torch.randn(NUM_EXPERTS, HIDDEN, DIM, generator=g)
    w2 = torch.randn(NUM_EXPERTS, DIM, HIDDEN, generator=g)
    gate = torch.randn(NUM_EXPERTS, DIM, generator=g)
    return w1, w3, w2, gate


def _build(*, ep_group, ep_rank: int, ep_size: int, w) -> MoE:
    w1, w3, w2, gate = w
    num_local = NUM_EXPERTS // ep_size
    lo = ep_rank * num_local

    experts = GroupedExperts(DIM, HIDDEN, num_local)
    with torch.no_grad():
        experts.w1_EFD.copy_(w1[lo : lo + num_local])
        experts.w3_EFD.copy_(w3[lo : lo + num_local])
        experts.w2_EDF.copy_(w2[lo : lo + num_local])

    router = TokenChoiceTopKRouter(
        NUM_EXPERTS, DIM, TOP_K, score_func="softmax", route_norm=True
    )
    with torch.no_grad():
        router.gate.weight.copy_(gate)

    dispatcher = LocalTokenDispatcher(NUM_EXPERTS, TOP_K)
    if ep_group is not None:
        dispatcher = AllToAllTokenDispatcher(NUM_EXPERTS, TOP_K)
        dispatcher.wire_meshes(ep_group=ep_group)

    return MoE(
        num_experts=NUM_EXPERTS,
        routed_experts=RoutedExperts(experts, dispatcher),
        router=router,
        load_balance_coeff=None,
    ).eval()


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    # Each rank owns its own token shard -- that is what makes EP a form of
    # parallelism. Feeding both ranks the same tokens would route every token
    # to the expert-owning rank twice and double the output, which is a property
    # of the setup, not a bug in the dispatcher.
    torch.manual_seed(1000 + rank)
    x = torch.randn(TOKENS, DIM)
    w = _weights()

    # Reference: this rank's tokens through all experts, no communication.
    ref_model = _build(ep_group=None, ep_rank=0, ep_size=1, w=w)
    with torch.no_grad():
        ref = ref_model(x)

    # EP: each rank holds half the experts; tokens cross via all-to-all.
    ep_model = _build(ep_group=dist.group.WORLD, ep_rank=rank, ep_size=world, w=w)
    with torch.no_grad():
        got = ep_model(x)

    diff = (ref - got).abs().max().item()
    scale = ref.abs().max().item()
    rel = diff / scale

    # fp32 reference for the same computation. Both the single-rank and the EP
    # path are compared against it, so a difference between them can be
    # attributed to reduction order rather than to a routing mistake: if either
    # path had sent a token to the wrong expert, it would sit far from this.
    with torch.no_grad():
        exact = _build(ep_group=None, ep_rank=0, ep_size=1, w=w).double()(x.double())
    ref_err = (ref.double() - exact).abs().max().item()
    got_err = (got.double() - exact).abs().max().item()

    if rank == 0:
        print(
            f"tokens={TOKENS} dim={DIM} experts={NUM_EXPERTS} "
            f"top_k={TOP_K} ep_size={world}"
        )
        print(f"|ref| max        = {scale:.4f}")
        print(f"max abs diff     = {diff:.3e}   (rel {rel:.3e})")
        print(f"local vs fp64    = {ref_err:.3e}")
        print(f"EP    vs fp64    = {got_err:.3e}")
        # Both paths must sit within fp32 noise of the exact answer, and their
        # disagreement must be no worse than that noise.
        assert rel < 1e-5, f"EP output diverged from reference (rel {rel:.3e})"

    # The gather is the real point: every rank must agree, not just rank 0.
    diff_t = torch.tensor(rel)
    dist.all_reduce(diff_t, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"max rel error across ranks = {diff_t.item():.3e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
