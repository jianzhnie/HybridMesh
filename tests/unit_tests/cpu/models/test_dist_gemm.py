"""Dist-GEMM modules: the TP-off fallbacks and the fused weights' layout.

These modules fold a TP collective into a GEMM. With TP off there is no
collective, so what is left to test is the arithmetic -- and for
``DistGEMMFeedForward`` that arithmetic has a trap: the fused ``w13`` stores gate
and up *interleaved* (``[g0, u0, g1, u1, ...]``), not as two contiguous halves.
Both pairings produce a correctly-shaped output, so a wrong one trains silently.
The tests below pin the interleaving, and assert the halves-split is genuinely a
different answer so that pin is not vacuous.

The collective path itself needs a live multi-rank TP mesh and is not covered
here.
"""

from __future__ import annotations

from tests.caps import require_env

require_env('spmd_types')


import logging

import pytest
import torch

from hpmesh.models.common.activation import SwiGLU
from hpmesh.models.common.async_linear import (
    AllGatherFusedQKVLinear,
    DistGEMMFeedForward,
    RowParallelLinear,
    validate_dist_gemm_preconditions,
)
from hpmesh.models.common.qkv import QKVLinear

DIM = 16
HIDDEN = 32
N_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 4
FUSED_QKV_OUT = N_KV_HEADS * (N_HEADS // N_KV_HEADS + 2) * HEAD_DIM


def _x(rows: int = 5, seed: int = 1) -> torch.Tensor:
    return torch.randn(rows, DIM, generator=torch.Generator().manual_seed(seed))


def _ffn(seed: int = 0) -> DistGEMMFeedForward:
    torch.manual_seed(seed)
    return DistGEMMFeedForward(
        w13=torch.nn.Linear(DIM, 2 * HIDDEN, bias=False),
        w2=torch.nn.Linear(HIDDEN, DIM, bias=False),
        activation_fn=SwiGLU(),
    )


# -- preconditions -----------------------------------------------------------


def test_preconditions_require_sequence_parallelism() -> None:
    """Without SP there are no boundary collectives for the GEMMs to replace."""
    with pytest.raises(ValueError, match="enable_sequence_parallel"):
        validate_dist_gemm_preconditions(enable_sp=False)


def test_preconditions_accept_sequence_parallelism() -> None:
    validate_dist_gemm_preconditions(enable_sp=True)


# -- DistGEMMFeedForward -----------------------------------------------------


def test_feed_forward_falls_back_without_tp() -> None:
    ffn = _ffn()
    x = _x()
    with torch.no_grad():
        out = ffn(x)
    assert out.shape == x.shape


def test_feed_forward_fallback_matches_the_unfused_computation() -> None:
    """TP off must reduce to two plain projections and the activation."""
    ffn = _ffn()
    x = _x()
    with torch.no_grad():
        out = ffn(x)

    gate_up = ffn.w13(x)
    gate, up = gate_up.unflatten(-1, (-1, 2)).unbind(-1)
    expected = ffn.w2(torch.nn.functional.silu(gate) * up)

    assert torch.equal(out, expected)


def test_feed_forward_splits_gate_and_up_interleaved() -> None:
    """The pairing is (g_i, u_i), not (first half, second half).

    Verified by construction: build a w13 whose two halves differ, then check
    the output matches the interleaved read and *not* the halves read.
    """
    torch.manual_seed(7)
    w13 = torch.nn.Linear(DIM, 2 * HIDDEN, bias=False)
    w2 = torch.nn.Linear(HIDDEN, DIM, bias=False)
    with torch.no_grad():
        # Make the two halves clearly distinguishable.
        w13.weight[:HIDDEN].normal_(0, 0.1)
        w13.weight[HIDDEN:].normal_(5.0, 0.1)
    ffn = DistGEMMFeedForward(w13=w13, w2=w2, activation_fn=SwiGLU())
    x = _x()

    with torch.no_grad():
        out = ffn(x)
        gate_up = w13(x)
        interleaved = w2(
            torch.nn.functional.silu(gate_up.unflatten(-1, (-1, 2))[..., 0])
            * gate_up.unflatten(-1, (-1, 2))[..., 1]
        )
        halves = w2(torch.nn.functional.silu(gate_up[:, :HIDDEN]) * gate_up[:, HIDDEN:])

    assert torch.equal(out, interleaved)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out, halves, rtol=1e-5, atol=1e-4)


def test_feed_forward_uses_the_given_activation() -> None:
    """A swapped activation must change the result, or the wiring is ignored."""
    torch.manual_seed(5)
    w13 = torch.nn.Linear(DIM, 2 * HIDDEN, bias=False)
    w2 = torch.nn.Linear(HIDDEN, DIM, bias=False)
    x = _x()

    class DoubleGate(torch.nn.Module):
        def forward(self, gate, up):
            return 2 * gate * up

    with torch.no_grad():
        swiglu_out = DistGEMMFeedForward(w13=w13, w2=w2, activation_fn=SwiGLU())(x)
        other_out = DistGEMMFeedForward(w13=w13, w2=w2, activation_fn=DoubleGate())(x)

    with pytest.raises(AssertionError):
        torch.testing.assert_close(swiglu_out, other_out, rtol=1e-5, atol=1e-8)


def test_feed_forward_warns_when_tp_is_off(caplog) -> None:
    """Silently running unfused is the failure mode the warning exists for."""
    import hpmesh.models.common.async_linear as dg

    dg._WARNED_NO_TP = False
    ffn = _ffn()
    with caplog.at_level(logging.WARNING, logger=dg.__name__):
        with torch.no_grad():
            ffn(_x())

    assert any("tensor parallelism is not active" in r.message for r in caplog.records)


def test_feed_forward_warns_only_once(caplog) -> None:
    """A per-step warning would flood the log; one per process is enough."""
    import hpmesh.models.common.async_linear as dg

    dg._WARNED_NO_TP = False
    ffn = _ffn()
    with caplog.at_level(logging.WARNING, logger=dg.__name__):
        with torch.no_grad():
            ffn(_x())
            ffn(_x())
            ffn(_x())

    assert sum("tensor parallelism" in r.message for r in caplog.records) == 1


# -- RowParallelLinear -------------------------------------------------------


def test_row_parallel_linear_falls_back_to_a_plain_linear() -> None:
    torch.manual_seed(2)
    row = RowParallelLinear(DIM, DIM, bias=True)
    with torch.no_grad():
        row.weight.normal_()
        row.bias.normal_()
    x = _x(rows=4)

    with torch.no_grad():
        out = row(x)
        expected = torch.nn.functional.linear(x, row.weight, row.bias)

    assert torch.equal(out, expected)


def test_row_parallel_linear_without_bias() -> None:
    torch.manual_seed(2)
    row = RowParallelLinear(DIM, DIM, bias=False)
    assert row.bias is None
    with torch.no_grad():
        out = row(_x(rows=4))
    assert out.shape == (4, DIM)


def test_row_parallel_linear_holds_its_weight_as_a_parameter() -> None:
    """The fused backward computes wgrad itself, so these must be Parameters."""
    row = RowParallelLinear(DIM, 2 * DIM, bias=True)
    assert row.weight.shape == (2 * DIM, DIM)
    assert row.weight.requires_grad
    assert row.bias.requires_grad


def test_row_parallel_linear_initializes_its_parameters() -> None:
    """``torch.empty`` leaves memory uninitialized; the constructor must init.

    Pinned by the same statistics ``nn.Linear``'s kaiming-uniform init has:
    finite, nonzero variance, and bounded by ``1/sqrt(fan_in)``.
    """
    row = RowParallelLinear(DIM, 2 * DIM, bias=True)
    bound = 1 / DIM**0.5
    assert torch.isfinite(row.weight).all()
    assert torch.isfinite(row.bias).all()
    assert row.weight.std() > 0
    assert row.weight.abs().max() <= bound
    assert row.bias.abs().max() <= bound


def test_row_parallel_linear_init_is_seeded() -> None:
    """Same seed, same weights -- an uninitialized buffer would not be."""
    torch.manual_seed(11)
    a = RowParallelLinear(DIM, DIM, bias=True)
    torch.manual_seed(11)
    b = RowParallelLinear(DIM, DIM, bias=True)
    assert torch.equal(a.weight, b.weight)
    assert torch.equal(a.bias, b.bias)


def test_row_parallel_linear_reset_parameters_reinitializes() -> None:
    row = RowParallelLinear(DIM, DIM, bias=True)
    with torch.no_grad():
        row.weight.fill_(0.0)
    row.reset_parameters()
    assert row.weight.std() > 0


# -- AllGatherFusedQKVLinear -------------------------------------------------


def _fused_and_plain() -> tuple[AllGatherFusedQKVLinear, QKVLinear]:
    torch.manual_seed(4)
    linear = torch.nn.Linear(DIM, FUSED_QKV_OUT)
    return (
        AllGatherFusedQKVLinear(
            head_dim=HEAD_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, linear=linear
        ),
        QKVLinear(
            head_dim=HEAD_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, linear=linear
        ),
    )


def test_fused_qkv_fallback_matches_the_plain_projection() -> None:
    """TP off must be exactly the stock QKV, collective and all."""
    fused, plain = _fused_and_plain()
    x = _x(rows=6)

    with torch.no_grad():
        got = fused(x)
        want = plain(x)

    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)


def test_fused_qkv_is_a_qkv_linear() -> None:
    """It overrides only the projection, so it must keep the base class's API."""
    fused, _ = _fused_and_plain()
    assert isinstance(fused, QKVLinear)


def test_fused_qkv_exposes_the_same_checkpoint_keys() -> None:
    """The collective changes the forward, not the parameter layout."""
    fused, plain = _fused_and_plain()
    assert set(dict(fused.state_dict())) == set(dict(plain.state_dict()))


# -- the TP-off group probe --------------------------------------------------


def test_tp_group_is_none_without_a_mesh_context() -> None:
    """No registered SPMD mesh means no TP axis, so the fallback path is taken."""
    from hpmesh.accelerator.spmd_context import spmd_mesh_group

    assert spmd_mesh_group("tp") is None
