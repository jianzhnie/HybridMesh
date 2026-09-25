"""The router's ``_debug_force_load_balance`` switch.

Two invariants are pinned:

* **on** -- the routing decision is a round-robin over ``arange(T * K) % E``,
  so every expert wins a (near-)equal share of the ``T * K`` slots no matter
  how lopsided the gate scores or the load-balancing bias are. The gating
  *value* still comes from the real scores of the expert the token landed on.
* **off** (the default) -- nothing changes: the bias steers the top-k choice
  exactly as before, so the switch can sit in the constructor unused.
"""

from __future__ import annotations

from tests.caps import require_env

require_env('spmd_types')


import torch

from hpmesh.models.common.moe import TokenChoiceTopKRouter

_NUM_EXPERTS = 4
_DIM = 8
_TOP_K = 2


def _router(*, debug: bool) -> TokenChoiceTopKRouter:
    torch.manual_seed(0)
    return TokenChoiceTopKRouter(
        _NUM_EXPERTS, _DIM, _TOP_K, _debug_force_load_balance=debug
    )


def test_every_expert_is_chosen_an_equal_number_of_times() -> None:
    """Over a folded stream, per-expert counts differ by at most one slot."""
    router = _router(debug=True)
    x_TD = torch.randn(10, _DIM)

    _, topk_expert_ids_TK, routing_map_TE = router(x_TD)

    counts_E = routing_map_TE.sum(dim=0)
    total_slots = 10 * _TOP_K
    expected = total_slots // _NUM_EXPERTS
    assert counts_E.tolist() == [expected] * _NUM_EXPERTS
    # The assignment is the literal round-robin, not just any balanced one.
    expected_ids = (
        torch.arange(10 * _TOP_K, dtype=torch.int64).reshape(10, _TOP_K)
        % _NUM_EXPERTS
    )
    assert torch.equal(topk_expert_ids_TK, expected_ids)


def test_the_assignment_ignores_scores_and_bias() -> None:
    """An extreme bias that would normally collapse routing changes nothing.

    This is the whole point of the switch: a routing decision nothing
    downstream can skew, so load imbalance can be isolated from the
    bias/score machinery.
    """
    router = _router(debug=True)
    x_TD = torch.randn(6, _DIM)
    expert_bias_E = torch.tensor([100.0, 0.0, 0.0, 0.0])

    _, ids_with_bias, map_with_bias = router(x_TD, expert_bias_E)

    assert int(map_with_bias[:, 0].sum()) == 6 * _TOP_K // _NUM_EXPERTS
    assert torch.equal(
        ids_with_bias,
        torch.arange(6 * _TOP_K, dtype=torch.int64).reshape(6, _TOP_K)
        % _NUM_EXPERTS,
    )


def test_scores_are_gathered_from_the_real_scores_of_the_forced_experts() -> None:
    """The choice is forced; the weight a token carries is not."""
    router = _router(debug=True)
    x_TD = torch.randn(4, _DIM)

    topk_scores_TK, topk_expert_ids_TK, _ = router(x_TD)

    scores_TE = torch.sigmoid(router.gate(x_TD))
    assert torch.equal(
        topk_scores_TK, scores_TE.gather(dim=-1, index=topk_expert_ids_TK)
    )


def test_disabled_by_default_and_matches_the_previous_behavior() -> None:
    """Default-constructed routers take the biased top-k path, unchanged."""
    router = TokenChoiceTopKRouter(_NUM_EXPERTS, _DIM, 1)
    assert router._debug_force_load_balance is False
    x_TD = torch.randn(6, _DIM)
    # An overwhelming bias on expert 0 must route every slot there.
    expert_bias_E = torch.tensor([100.0, 0.0, 0.0, 0.0])

    _, topk_expert_ids_TK, _ = router(x_TD, expert_bias_E)

    assert torch.equal(
        topk_expert_ids_TK, torch.zeros_like(topk_expert_ids_TK)
    )
    x_TD = torch.randn(6, _DIM)
    # An overwhelming bias on expert 0 must route every slot there.
    expert_bias_E = torch.tensor([100.0, 0.0, 0.0, 0.0])

    _, topk_expert_ids_TK, _ = router(x_TD, expert_bias_E)

    assert torch.equal(
        topk_expert_ids_TK, torch.zeros_like(topk_expert_ids_TK)
    )


def test_route_norm_and_scale_still_apply_to_the_forced_scores() -> None:
    """The debug path rejoins the normal one before norm/scale."""
    router = TokenChoiceTopKRouter(
        _NUM_EXPERTS,
        _DIM,
        _TOP_K,
        route_norm=True,
        route_scale=2.0,
        _debug_force_load_balance=True,
    )
    x_TD = torch.randn(4, _DIM)

    topk_scores_TK, topk_expert_ids_TK, _ = router(x_TD)

    scores_TE = torch.sigmoid(router.gate(x_TD))
    gathered = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)
    expected = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20) * 2.0
    assert torch.allclose(topk_scores_TK, expected)
