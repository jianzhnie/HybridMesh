"""Padding-mask filtering of the MoE load-balancing statistics.

Pinned here, mirroring the upstream semantics:

* the mask is **statistics-only** -- routing, dispatch and expert compute run
  on the full token stream, so the block's output is bitwise identical with
  and without a mask;
* **statistics skip padding** -- ``tokens_per_expert_E``, the aux loss's f/p
  terms and the quantile histogram count valid tokens only, and the no-mask
  path is unchanged (an all-valid mask is bitwise the same as no mask);
* the **staged channel** -- ``set_padding_mask`` feeds the next forward
  exactly once, so a stale mask cannot leak into a maskless microbatch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hpmesh.models.common.aux_loss import AuxLoss
from hpmesh.models.common.grouped_experts import GroupedExperts
from hpmesh.models.common.moe import (
    MicrobatchWiseLoadBalanceLoss,
    MoE,
    QuantileBalancedTopKRouter,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from hpmesh.models.common.token_dispatcher import LocalTokenDispatcher

_NUM_EXPERTS = 4
_DIM = 8
_TOP_K = 2
_NUM_TOKENS = 16


def _moe(
    seed: int = 0, *, aux_loss: bool = True, router=None, **kwargs
) -> MoE:
    torch.manual_seed(seed)
    if router is None:
        router = TokenChoiceTopKRouter(
            _NUM_EXPERTS,
            _DIM,
            _TOP_K,
            score_func="sigmoid",
            aux_loss=(
                MicrobatchWiseLoadBalanceLoss(coeff=1.0) if aux_loss else None
            ),
        )
    moe = MoE(
        _NUM_EXPERTS,
        RoutedExperts(
            GroupedExperts(_DIM, 8, _NUM_EXPERTS),
            LocalTokenDispatcher(_NUM_EXPERTS, _TOP_K),
        ),
        router,
        **{"load_balance_coeff": 1e-3, **kwargs},
    )
    # GroupedExperts allocates its weights uninitialized; give them a
    # deterministic value so bitwise output comparisons mean something.
    experts = moe.routed_experts.inner_experts
    with torch.no_grad():
        for weight in (experts.w1_EFD, experts.w3_EFD, experts.w2_EDF):
            weight.normal_(0.0, 0.02)
    return moe


def _tokens(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(_NUM_TOKENS, _DIM)


def _padding_mask() -> torch.Tensor:
    mask = torch.zeros(_NUM_TOKENS, dtype=torch.bool)
    mask[3] = True
    mask[7] = True
    mask[10:] = True
    return mask


# -- no-mask / all-valid invariance -------------------------------------------


def test_all_valid_mask_is_bit_identical_to_no_mask() -> None:
    """An all-False mask must reproduce the no-mask path bit for bit, in the
    output and in every statistic it could touch."""
    x_TD = _tokens()
    all_valid = torch.zeros(_NUM_TOKENS, dtype=torch.bool)

    AuxLoss.set_step_denominator(torch.tensor(1.0))
    moe_plain = _moe()
    out_plain = moe_plain(x_TD)

    moe_masked = _moe()
    out_masked = moe_masked(x_TD, padding_mask=all_valid)

    assert torch.equal(out_plain, out_masked)
    assert torch.equal(moe_plain.tokens_per_expert_E, moe_masked.tokens_per_expert_E)
    assert torch.equal(
        moe_plain.router.aux_loss.instance_acc,
        moe_masked.router.aux_loss.instance_acc,
    )


def test_output_is_bit_identical_with_a_real_mask() -> None:
    """The mask filters statistics only: the routing decision, dispatch and
    expert compute see the full stream, so the block output cannot move."""
    x_TD = _tokens()
    AuxLoss.set_step_denominator(torch.tensor(1.0))

    moe_plain = _moe()
    out_plain = moe_plain(x_TD)

    moe_masked = _moe()
    out_masked = moe_masked(x_TD, padding_mask=_padding_mask())

    assert torch.equal(out_plain, out_masked)
    # But the usage counter did drop the padding tokens.
    assert not torch.equal(
        moe_plain.tokens_per_expert_E, moe_masked.tokens_per_expert_E
    )


# -- masked statistics ---------------------------------------------------------


def test_tokens_per_expert_counts_valid_tokens_only() -> None:
    moe = _moe()
    x_TD = _tokens()
    mask = _padding_mask()
    AuxLoss.set_step_denominator(torch.tensor(1.0))

    moe(x_TD, padding_mask=mask)

    moe.eval()  # a reference routing map without re-running the aux loss
    with torch.no_grad():
        _, _, routing_map_TE = moe.router(x_TD, moe.expert_bias_E)
    expected = (routing_map_TE & ~mask.unsqueeze(-1)).sum(dim=0)
    assert torch.equal(moe.tokens_per_expert_E, expected.to(torch.float32))
    # Every valid token picked K experts; padding tokens picked some too but
    # none of those land in the counter.
    assert moe.tokens_per_expert_E.sum().item() == (~mask).sum().item() * _TOP_K


def test_aux_loss_counts_valid_tokens_only() -> None:
    """f follows the masked routing map (T-free form, so the denominator is
    K * T_valid) and p drops the padding rows of the normalized scores."""
    moe = _moe()
    x_TD = _tokens()
    mask = _padding_mask()
    AuxLoss.set_step_denominator(torch.tensor(1.0))

    moe(x_TD, padding_mask=mask)

    router = moe.router
    moe.eval()  # reference reads, without re-running the aux loss
    with torch.no_grad():
        scores_TE = torch.sigmoid(router.gate(x_TD))
        _, _, routing_map_TE = router(x_TD, moe.expert_bias_E)
    masked_map_TE = routing_map_TE & ~mask.unsqueeze(-1)
    probs_TE = F.normalize(scores_TE, p=1, dim=-1) * ~mask.unsqueeze(-1)

    counts_E = masked_map_TE.to(scores_TE.dtype).sum(dim=0)
    f_E = counts_E / counts_E.sum() * _NUM_EXPERTS
    p_E = probs_TE.sum(dim=0)
    expected = (f_E * p_E).sum()

    assert router.aux_loss.instance_acc.item() == pytest.approx(expected.item())


def test_aux_loss_injects_no_gradient_on_padding_rows() -> None:
    """Padding rows are zeroed in p before the sum, so no gradient can flow
    back into their scores."""
    loss = MicrobatchWiseLoadBalanceLoss(coeff=1.0)
    AuxLoss.set_step_denominator(torch.tensor(1.0))

    torch.manual_seed(2)
    scores_TE = torch.rand(_NUM_TOKENS, _NUM_EXPERTS, requires_grad=True)
    # The router hands over the padding-filtered map; rebuild that here.
    mask = _padding_mask()
    routing_map_TE = torch.zeros(_NUM_TOKENS, _NUM_EXPERTS, dtype=torch.bool)
    routing_map_TE.scatter_(-1, torch.arange(_NUM_TOKENS).unsqueeze(-1) % 4, True)
    routing_map_TE &= ~mask.unsqueeze(-1)
    carrier_TK = torch.ones(_NUM_TOKENS, _TOP_K, requires_grad=True)

    out_TK = loss(
        scores_TE,
        routing_map_TE,
        carrier=carrier_TK,
        padding_mask_T=mask,
    )
    out_TK.sum().backward()

    assert scores_TE.grad is not None
    assert torch.all(scores_TE.grad[mask] == 0)
    assert scores_TE.grad[~mask].abs().sum() > 0


# -- validation ----------------------------------------------------------------


def test_non_bool_mask_is_rejected() -> None:
    moe = _moe()
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    with pytest.raises(ValueError, match="dtype bool"):
        moe(_tokens(), padding_mask=torch.zeros(_NUM_TOKENS, dtype=torch.int64))


def test_mismatched_mask_shape_is_rejected() -> None:
    moe = _moe()
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    with pytest.raises(ValueError, match="shape matching the routing-map"):
        moe(_tokens(), padding_mask=torch.zeros(_NUM_TOKENS + 1, dtype=torch.bool))


# -- combination with the two balancing schemes --------------------------------


def test_debug_force_load_balance_still_counts_valid_only() -> None:
    """Explicit combination semantics: the forced round-robin choice is
    unaffected by the mask (routing never is), but the statistics stay
    padding-filtered."""
    torch.manual_seed(0)
    router = TokenChoiceTopKRouter(
        _NUM_EXPERTS,
        _DIM,
        _TOP_K,
        score_func="sigmoid",
        _debug_force_load_balance=True,
    )
    moe = _moe(router=router)
    x_TD = _tokens()
    mask = _padding_mask()

    moe(x_TD, padding_mask=mask)

    num_valid = int((~mask).sum())
    slots = num_valid * _TOP_K
    expected_total = slots
    assert moe.tokens_per_expert_E.sum().item() == expected_total
    # Round-robin over the FULL stream, filtered afterwards: expert e is hit
    # by slots (t * K + k) % E == e whose token t is valid.
    counts = torch.zeros(_NUM_EXPERTS)
    for t in range(_NUM_TOKENS):
        if mask[t]:
            continue
        for k in range(_TOP_K):
            counts[(t * _TOP_K + k) % _NUM_EXPERTS] += 1
    assert torch.equal(moe.tokens_per_expert_E, counts)


def test_quantile_router_observes_valid_tokens_only() -> None:
    torch.manual_seed(0)
    router = QuantileBalancedTopKRouter(_NUM_EXPERTS, _DIM, _TOP_K)
    moe = _moe(router=router, load_balance_coeff=None)
    x_TD = _tokens()
    mask = _padding_mask()

    moe(x_TD, padding_mask=mask)

    observed = router.quantile_balancer.required_bias_histogram_EB.to(torch.int64)

    # Reference: the same observation over the valid rows only.
    torch.manual_seed(0)
    reference_router = QuantileBalancedTopKRouter(_NUM_EXPERTS, _DIM, _TOP_K)
    scores_TE = torch.sigmoid(reference_router.gate(x_TD))
    biased_TE = scores_TE + moe.expert_bias_E
    topk_plus_one = torch.topk(biased_TE, k=_TOP_K + 1, dim=-1, sorted=True).values
    reference_router.quantile_balancer.observe(
        scores_TE[~mask],
        topk_plus_one[:, _TOP_K:][~mask],
        moe.expert_bias_E,
    )
    expected = reference_router.quantile_balancer.required_bias_histogram_EB.to(
        torch.int64
    )

    assert torch.equal(observed, expected)
    # One histogram entry per (valid token, expert).
    assert observed.sum().item() == (~mask).sum().item() * _NUM_EXPERTS
    # Routing itself is unaffected: same ids as the maskless forward.
    _, ids_masked, _ = router(x_TD, moe.expert_bias_E, padding_mask_T=mask)
    _, ids_plain, _ = router(x_TD, moe.expert_bias_E)
    assert torch.equal(ids_masked, ids_plain)


# -- the staged channel ----------------------------------------------------------


def test_set_padding_mask_is_consumed_once() -> None:
    """The wrapper stages the mask because the HF layer's fixed
    ``self.mlp(hidden_states)`` call cannot thread it. Consumption is
    one-shot: the next forward runs the maskless path again."""
    moe = _moe()
    x_TD = _tokens()
    mask = _padding_mask()
    AuxLoss.set_step_denominator(torch.tensor(1.0))

    moe.set_padding_mask(mask)
    moe(x_TD)
    masked_counts = moe.tokens_per_expert_E.clone()
    assert masked_counts.sum().item() == (~mask).sum().item() * _TOP_K

    # Consumed: a second forward with no staging counts every token.
    moe(x_TD)
    total = moe.tokens_per_expert_E - masked_counts
    assert total.sum().item() == _NUM_TOKENS * _TOP_K


def test_explicit_mask_takes_precedence_over_a_staged_one() -> None:
    moe = _moe()
    x_TD = _tokens()
    AuxLoss.set_step_denominator(torch.tensor(1.0))

    moe.set_padding_mask(_padding_mask())
    moe(x_TD, padding_mask=torch.zeros(_NUM_TOKENS, dtype=torch.bool))
    assert moe.tokens_per_expert_E.sum().item() == _NUM_TOKENS * _TOP_K
    # The staged mask was consumed, not queued behind the explicit one.
    moe(x_TD)
    assert (
        moe.tokens_per_expert_E.sum().item() == 2 * _NUM_TOKENS * _TOP_K
    )


def test_leading_dims_mask_flattens_with_the_input() -> None:
    """The HF layer calls with ``(batch, seq, D)``; the mask follows the same
    flattening."""
    moe = _moe()
    x_BTD = _tokens().reshape(2, _NUM_TOKENS // 2, _DIM)
    mask_BT = _padding_mask().reshape(2, _NUM_TOKENS // 2)
    AuxLoss.set_step_denominator(torch.tensor(1.0))

    out = moe(x_BTD, padding_mask=mask_BT)
    assert out.shape == x_BTD.shape
    assert moe.tokens_per_expert_E.sum().item() == (~mask_BT).sum().item() * _TOP_K
