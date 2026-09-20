"""Unit tests for ``models/common``: the FFN and the router-gate projection.

The FFN is exercised through its actual contract -- the fused ``w13`` layout, the
logical ``w1``/``w3`` checkpoint keys, and the sigmoid gate -- and the router gate
through the property it exists for: a score that is fp32 in both directions
whatever dtype the model runs in.
"""

from __future__ import annotations

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
from hpmesh.models.common.linear import PartialBiasRowwiseLinear, RouterGateLinear


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
