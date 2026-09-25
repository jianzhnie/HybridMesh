"""Unit tests for ``models/common``: the FFN and the router-gate projection.

The FFN is exercised through its actual contract -- the fused ``w13`` layout, the
logical ``w1``/``w3`` checkpoint keys, and the sigmoid gate -- and the router gate
through the property it exists for: a score that is fp32 in both directions
whatever dtype the model runs in.
"""

from __future__ import annotations

from tests.caps import require_env

require_env('spmd_types')


import pytest
import torch
import torch.nn as nn

from hpmesh.models.common.activation import SwiGLU
from hpmesh.models.common.dist_gemm import DistGEMMFeedForward
from hpmesh.models.common.feed_forward import (
    FeedForward,
    SigmoidGatedFeedForward,
    compute_ffn_hidden_dim,
)
from hpmesh.models.common.grouped_experts import GroupedExperts
from hpmesh.models.common.linear import PartialBiasRowwiseLinear, RouterGateLinear
from hpmesh.models.common.moe import TokenChoiceTopKRouter


def _ffn(dim: int = 8, hidden: int = 6) -> FeedForward:
    return FeedForward(
        w13=nn.Linear(dim, 2 * hidden, bias=False),
        w2=nn.Linear(hidden, dim, bias=False),
    )


# -- FeedForward -------------------------------------------------------------


def test_hidden_dim_applies_the_two_thirds_scaling() -> None:
    # int(2 * 4 * 12 / 3) == 32, no rounding needed.
    assert compute_ffn_hidden_dim(12) == 32


def test_hidden_dim_rounds_up_to_the_multiple() -> None:
    # 32 -> rounded up to a multiple of 7 is 35.
    assert compute_ffn_hidden_dim(12, multiple_of=7) == 35
    assert compute_ffn_hidden_dim(12, multiple_of=7) % 7 == 0


def test_hidden_dim_applies_the_multiplier_before_rounding() -> None:
    assert compute_ffn_hidden_dim(12, ffn_dim_multiplier=1.5) == 48
    assert compute_ffn_hidden_dim(12, ffn_dim_multiplier=1.5, multiple_of=10) == 50


def test_forward_is_w2_of_the_gated_activation() -> None:
    """The fused path and an explicit split must agree exactly."""
    torch.manual_seed(0)
    dim, hidden = 8, 6
    ffn = _ffn(dim, hidden)
    x = torch.randn(4, dim)

    gate_up = ffn.w13(x)
    gate, up = gate_up.unflatten(-1, (-1, 2)).unbind(-1)
    expected = ffn.w2(torch.nn.functional.silu(gate) * up)

    assert torch.equal(ffn(x), expected)


def test_interleaved_layout_pairs_each_gate_with_its_own_up() -> None:
    """``w13`` is ``[g0, u0, g1, u1, ...]``, not ``[g0, g1, ..., u0, u1, ...]``.

    This is the layout the checkpoint hooks assume, so a change to it would
    corrupt every save/load rather than fail loudly.
    """
    torch.manual_seed(0)
    dim, hidden = 4, 3
    w13 = nn.Linear(dim, 2 * hidden, bias=False)
    # Make gate and up trivially distinguishable: row 2i is all ones, 2i+1 all
    # zeros, so a misinterleaved split would pick up the wrong half.
    with torch.no_grad():
        rows = torch.arange(2 * hidden).float().reshape(2 * hidden, 1)
        w13.weight.copy_(rows.repeat(1, dim))
    ffn = FeedForward(w13=w13, w2=nn.Linear(hidden, dim, bias=False))

    gate_up = ffn.w13(torch.ones(1, dim))
    gate, up = ffn._split_gate_up(gate_up)
    assert torch.equal(gate, gate_up[..., 0::2])
    assert torch.equal(up, gate_up[..., 1::2])


def test_checkpoint_keys_expose_the_logical_w1_and_w3() -> None:
    """A save presents ``w1``/``w3``; the fused ``w13`` does not leak out."""
    torch.manual_seed(0)
    ffn = _ffn()
    saved = ffn.state_dict()

    assert set(saved) == {"w1.weight", "w3.weight", "w2.weight"}


def test_checkpoint_round_trip_restores_the_same_weights() -> None:
    torch.manual_seed(0)
    source = _ffn()
    target = _ffn()
    # Different weights to start, so a no-op load would fail the check.
    with torch.no_grad():
        for param in target.parameters():
            param.zero_()

    target.load_state_dict(source.state_dict())

    for a, b in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(a, b)


def test_saved_gate_and_up_are_the_two_halves_of_w13() -> None:
    torch.manual_seed(0)
    ffn = _ffn()
    saved = ffn.state_dict()
    fused = ffn.w13.weight.unflatten(0, (-1, 2))

    assert torch.equal(saved["w1.weight"], fused[:, 0].contiguous())
    assert torch.equal(saved["w3.weight"], fused[:, 1].contiguous())


def test_bias_checkpoint_keys_are_handled_too() -> None:
    """The hooks walk ``("weight", "bias")``; a biased FFN must not break them."""
    ffn = FeedForward(
        w13=nn.Linear(8, 12, bias=True),
        w2=nn.Linear(6, 8, bias=True),
    )
    assert set(ffn.state_dict()) == {
        "w1.weight",
        "w1.bias",
        "w3.weight",
        "w3.bias",
        "w2.weight",
        "w2.bias",
    }


def test_sigmoid_gate_multiplies_the_ffn_output() -> None:
    torch.manual_seed(0)
    dim, hidden = 8, 6
    inner = _ffn(dim, hidden)
    gate = nn.Linear(dim, dim, bias=False)
    gated = SigmoidGatedFeedForward(w13=inner.w13, w2=inner.w2, gate=gate)
    x = torch.randn(4, dim)

    expected = torch.sigmoid(gate(x)) * FeedForward(w13=inner.w13, w2=inner.w2)(x)
    assert torch.equal(gated(x), expected)


def test_sigmoid_gated_ffn_keeps_the_base_checkpoint_keys() -> None:
    dim, hidden = 8, 6
    inner = _ffn(dim, hidden)
    gated = SigmoidGatedFeedForward(
        w13=inner.w13, w2=inner.w2, gate=nn.Linear(dim, dim, bias=False)
    )
    # The extra projection is its own key; the fused ones are still split.
    assert set(gated.state_dict()) == {
        "w1.weight",
        "w3.weight",
        "w2.weight",
        "gate.weight",
    }


def test_default_activation_is_swiglu() -> None:
    assert isinstance(_ffn().activation_fn, SwiGLU)


# -- DistGEMMFeedForward -----------------------------------------------------


def test_dist_gemm_ffn_is_a_feed_forward() -> None:
    """The fused class subclasses the base, so the checkpoint hooks carry over."""
    assert issubclass(DistGEMMFeedForward, FeedForward)


def test_dist_gemm_ffn_splits_its_weights_like_the_base() -> None:
    inner = _ffn()
    fused = DistGEMMFeedForward(w13=inner.w13, w2=inner.w2)
    assert set(fused.state_dict()) == set(inner.state_dict())


def test_dist_gemm_ffn_falls_back_to_the_plain_path_without_tp() -> None:
    """No TP group -> the inherited forward, bit for bit.

    The fallback is announced with a warning rather than silence (a
    misconfiguration would otherwise look like success), so the test also pins
    that the two paths are numerically the same.
    """
    torch.manual_seed(0)
    inner = _ffn()
    fused = DistGEMMFeedForward(w13=inner.w13, w2=inner.w2)
    x = torch.randn(4, 8)

    assert torch.equal(fused(x), inner(x))


# -- RouterGateLinear --------------------------------------------------------


def test_router_gate_returns_fp32_from_a_bf16_input() -> None:
    torch.manual_seed(0)
    gate = RouterGateLinear(8, 4).to(torch.bfloat16)
    scores = gate(torch.randn(3, 8, dtype=torch.bfloat16))

    assert scores.dtype is torch.float32
    assert scores.shape == (3, 4)


def test_router_gate_returns_fp32_from_an_fp32_input() -> None:
    gate = RouterGateLinear(8, 4)
    scores = gate(torch.randn(3, 8))

    assert scores.dtype is torch.float32


def test_router_gate_gradients_are_fp32_in_both_directions() -> None:
    """The custom Function exists to pin the BACKWARD too, not just the forward.

    Without it autograd would derive a bf16 backward from a bf16 input, which is
    the path the router's top-k ordering is most sensitive to.
    """
    torch.manual_seed(0)
    gate = RouterGateLinear(8, 4).to(torch.bfloat16)
    x = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)

    gate(x).sum().backward()

    assert x.grad.dtype is torch.bfloat16
    assert gate.weight.grad is not None


def test_router_gate_matches_a_plain_fp32_linear() -> None:
    """The result is the same projection, computed in fp32."""
    torch.manual_seed(0)
    gate = RouterGateLinear(8, 4)
    plain = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        plain.weight.copy_(gate.weight)

    x = torch.randn(3, 8)
    torch.testing.assert_close(gate(x), plain(x.float()), rtol=0, atol=0)


def test_router_gate_is_the_moe_router_projection() -> None:
    """One class, imported by the MoE router -- not a second copy of it."""
    from hpmesh.models.common.moe import RouterGateLinear as FromMoe

    assert FromMoe is RouterGateLinear


# -- PartialBiasRowwiseLinear ------------------------------------------------


def test_partial_bias_rowwise_requires_a_bias() -> None:
    with pytest.raises(ValueError, match="requires bias=True"):
        PartialBiasRowwiseLinear(8, 4, bias=False)


def test_partial_bias_rowwise_matches_a_plain_linear_without_a_tp_group() -> None:
    """No TP group -> no redistribution, so it is an ordinary F.linear."""
    torch.manual_seed(0)
    layer = PartialBiasRowwiseLinear(8, 4)
    x = torch.randn(3, 8)

    assert torch.equal(
        layer(x), torch.nn.functional.linear(x, layer.weight, layer.bias)
    )


# -- node-limited routing (DeepSeek-V3) ----------------------------------------

_E, _D, _K = 8, 16, 2


def _router(**kw) -> TokenChoiceTopKRouter:
    torch.manual_seed(0)
    return TokenChoiceTopKRouter(_E, _D, _K, **kw)


def test_group_limited_routing_confines_every_token_to_the_chosen_groups() -> None:
    """The whole point of the restriction: no token may reach an unchosen group.

    ``n_group=2`` splits the 8 experts into ``{0..3}`` and ``{4..7}``; with
    ``topk_group=1`` every token must draw its K experts from one of them. This
    is what bounds inter-node traffic when a group is a node, so a silent
    regression here would be a performance bug that still trains correctly --
    exactly the kind that never gets noticed.
    """
    router = _router(num_expert_groups=2, num_limited_groups=1)
    x = torch.randn(64, _D)

    ids = router._select_experts(torch.sigmoid(router.gate(x)))

    group_of = ids // (_E // 2)
    assert bool((group_of[:, :1] == group_of).all()), "a token crossed groups"


def test_without_grouping_tokens_are_free_to_cross_groups() -> None:
    """The non-vacuity check for the test above.

    Same weights, same tokens, grouping off: if the restriction were being
    applied unconditionally -- or if the grouping flags were ignored -- this
    would still show every token in one group, and the test above would prove
    nothing.
    """
    router = _router()
    x = torch.randn(64, _D)

    ids = router._select_experts(torch.sigmoid(router.gate(x)))

    group_of = ids // (_E // 2)
    assert not bool((group_of[:, :1] == group_of).all())


def test_the_group_score_is_the_sum_of_the_two_best_experts() -> None:
    """A group wins on its strongest *pair*, not its single best expert.

    DeepSeek-V3 Sec 2.1.1 scores a group by its top-2 sum, so a group with two
    solid experts beats one holding a single outlier. A ``max`` or ``mean`` rule
    would pick the other group here -- which is why this is pinned on the ids
    rather than on a property the two rules share.

    Groups are ``{0,1,2,3}`` and ``{4,5,6,7}``. Group 1 holds the highest single
    expert (0.90) but group 0 wins on the pair: its top two are 0.80 and 0.70,
    summing to 1.50 against group 1's 0.90 + 0.20. Under a ``max`` rule group 1
    would win and the top-2 ids would come from it.
    """
    router = _router(num_expert_groups=2, num_limited_groups=1)
    scores = torch.tensor([[0.80, 0.70, 0.00, 0.00, 0.90, 0.20, 0.00, 0.00]])

    ids = router._select_experts(scores)

    assert ids.tolist() == [[0, 1]], (
        "group 0 (top-2 sum 1.50) must beat group 1 (1.10), despite 0.90 being "
        "the single largest expert"
    )


def test_the_bias_shifts_which_experts_win_but_not_their_scores() -> None:
    """The load-balancing bias is a routing device, never a value.

    ``forward`` gathers the weight from the *unbiased* scores, so a bias strong
    enough to change the selection must leave the score carried by a given
    expert untouched. Otherwise the bias would quietly rescale the MoE output --
    it is added to choose experts, not to weight them.
    """
    router = _router()
    x = torch.randn(32, _D)
    unbiased = torch.sigmoid(router.gate(x))

    base_scores, base_ids, _ = router(x)
    bias = torch.zeros(_E)
    bias[5] = 10.0
    steered_scores, steered_ids, _ = router(x, bias)

    assert int((steered_ids == 5).sum()) > int((base_ids == 5).sum()), (
        "bias did nothing"
    )

    # Every returned score is that expert's unbiased score, wherever it landed.
    expected = unbiased.gather(1, steered_ids)
    torch.testing.assert_close(steered_scores, expected, rtol=0, atol=0)
    # ...and the same for the unsteered run, so the check above is not vacuous.
    torch.testing.assert_close(
        base_scores, unbiased.gather(1, base_ids), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_expert_groups": 2}, "num_limited_groups must be set"),
        (
            {"num_expert_groups": 3, "num_limited_groups": 1},
            "must be divisible by num_expert_groups",
        ),
        (
            {"num_expert_groups": 2, "num_limited_groups": 5},
            "cannot exceed num_expert_groups",
        ),
        (
            {"num_expert_groups": 8, "num_limited_groups": 1},
            "must be >= 2",
        ),
    ],
)
def test_a_malformed_group_config_is_rejected(kwargs: dict, message: str) -> None:
    """Every one of these would otherwise fail deep inside a topk with a shape error."""
    with pytest.raises(ValueError, match=message):
        TokenChoiceTopKRouter(_E, _D, _K, **kwargs)


# -- GroupedExperts: the two expert-GEMM paths -------------------------------


def _grouped(*, use_grouped_mm: bool | None, dtype=torch.float32, seed: int = 0):
    """A real (not ``torch.empty``) GroupedExperts, so comparisons mean something."""
    torch.manual_seed(seed)
    module = GroupedExperts(
        dim=16, hidden_dim=32, num_experts=4, use_grouped_mm=use_grouped_mm
    ).to(dtype)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, 0.03)
    return module


_COUNTS = torch.tensor([5, 6, 4, 7])


def test_grouped_experts_runs_every_expert_over_its_own_tokens() -> None:
    """The loop path segments the tokens, one expert each, no bleeding.

    Nothing else here would catch a segmentation bug: every expert gets tokens,
    the output shape is right either way, and a wrong segment boundary just
    makes the numbers quietly wrong. So the reference is built by hand, one
    expert at a time.
    """
    module = _grouped(use_grouped_mm=False)
    x = torch.randn(22, 16)

    # Each expert's slice of the tokens, as (start, end) pairs.
    bounds = [0, 5, 11, 15, 22]
    segments = list(zip(bounds[:-1], bounds[1:], strict=True))

    gate = torch.cat([x[s:o] @ module.w1_EFD[e].T for e, (s, o) in enumerate(segments)])
    up = torch.cat([x[s:o] @ module.w3_EFD[e].T for e, (s, o) in enumerate(segments)])
    expected = torch.cat(
        [
            module.activation_fn(gate[s:o], up[s:o]) @ module.w2_EDF[e].T
            for e, (s, o) in enumerate(segments)
        ]
    )
    assert torch.allclose(module(x, _COUNTS), expected, rtol=1e-5, atol=1e-6)


def test_the_grouped_mm_probe_follows_the_op(monkeypatch) -> None:
    """The probe decides the default, so it has to track the op's real behavior.

    Driven by monkeypatching rather than by asking this machine: a test that
    only compares the probe against the local op passes on any host where the
    op works even if the probe is hard-wired to ``True`` -- which is precisely
    the bug that would send every model down a branch that raises elsewhere.
    """
    from hpmesh.models.common import grouped_experts as ge

    def _works(*args, **kwargs):
        return torch.zeros(8, 8, dtype=torch.bfloat16)

    def _raises(*args, **kwargs):
        raise RuntimeError("strides should be multiple of 16 bytes")

    monkeypatch.setattr(torch, "_grouped_mm", _works, raising=False)
    assert ge._grouped_mm_available() is True

    monkeypatch.setattr(torch, "_grouped_mm", _raises, raising=False)
    assert ge._grouped_mm_available() is False

    # An op that is simply absent counts as unavailable, not as an error.
    monkeypatch.delattr(torch, "_grouped_mm", raising=False)
    assert ge._grouped_mm_available() is False


def test_the_grouped_mm_probe_agrees_with_the_real_op_here() -> None:
    """On whatever host this runs, the probe must match the actual call."""
    from hpmesh.models.common.grouped_experts import _grouped_mm_available

    reported = _grouped_mm_available()
    try:
        torch._grouped_mm(
            torch.zeros(8, 8, dtype=torch.bfloat16),
            torch.zeros(2, 8, 8, dtype=torch.bfloat16),
            offs=torch.tensor([4, 8], dtype=torch.int32),
        )
        actually_works = True
    except Exception:
        actually_works = False

    assert reported == actually_works
    # And the decision holds up end to end: a model built with no explicit flag
    # must survive the forward whichever branch the probe chose.
    module = _grouped(use_grouped_mm=None, dtype=torch.bfloat16)
    module(torch.randn(22, 16, dtype=torch.bfloat16), _COUNTS)


def test_the_fused_path_matches_the_loop_bit_for_bit_in_bf16() -> None:
    """Where the op is available, the two paths must agree exactly.

    They are the same arithmetic in the same dtype (bf16 in, bf16 out, no
    intermediate widening on either side), so "close" would be the wrong
    assertion and would hide a mis-segmented expert. Skipped where the op is
    unavailable.
    """
    from hpmesh.models.common.grouped_experts import _grouped_mm_available

    if not _grouped_mm_available():
        pytest.skip("torch._grouped_mm is unavailable on this build")

    loop = _grouped(use_grouped_mm=False, dtype=torch.bfloat16)
    fused = _grouped(use_grouped_mm=True, dtype=torch.bfloat16)
    fused.load_state_dict(loop.state_dict())
    x = torch.randn(22, 16, dtype=torch.bfloat16)

    assert torch.equal(loop(x, _COUNTS), fused(x, _COUNTS))


def test_the_fused_path_is_refused_for_a_wider_dtype() -> None:
    """The bf16-only kernel must not be handed fp32 or fp64 activations.

    ``torch._grouped_mm`` casts both operands to bf16. On an fp32 model that is
    a silent 8-mantissa-bit round trip -- measured at 1.4e-1 relative error --
    which is exactly what a loss-comparison gate cannot see.

    The comparison is explicitly loop-against-fused, both with the flag forced:
    on a build where the op is available, ``use_grouped_mm=None`` resolves to
    *fused*, so comparing a default against a forced fused model would compare
    the same path with itself and pass no matter what the gate did.
    """
    for dtype in (torch.float32, torch.float64):
        loop = _grouped(use_grouped_mm=False, dtype=dtype)
        forced = _grouped(use_grouped_mm=True, dtype=dtype)
        forced.load_state_dict(loop.state_dict())
        x = torch.randn(22, 16, dtype=dtype)

        # Same weights, same dtype: the loop is the reference and the fused
        # request must not reach the kernel, so these are bitwise equal.
        assert torch.equal(loop(x, _COUNTS), forced(x, _COUNTS))


def test_the_default_follows_the_probe() -> None:
    """``None`` means "decide here", and the decision is the probe's."""
    from hpmesh.models.common.grouped_experts import _grouped_mm_available

    assert _grouped(use_grouped_mm=None).use_grouped_mm == _grouped_mm_available()
    # An explicit value still wins, which is what lets a test pin a path.
    assert _grouped(use_grouped_mm=False).use_grouped_mm is False
    assert _grouped(use_grouped_mm=True).use_grouped_mm is True
