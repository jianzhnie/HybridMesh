"""Rotary position embeddings: shapes, conventions, and round-trips.

The vendored ``rope.py`` was verified bit-identical to torchtitan's original at
migration time (all five class x scaling combinations, both yarn truncate modes,
cache and forward output compared with ``torch.equal``). That check needs
torchtitan on the path and is not reproducible here, so what these tests cover
is the *behaviour* the embeddings must keep on their own -- the conventions that
would silently corrupt attention if a refactor inverted them.

The two formats differ in a way worth pinning down:

* ``ComplexRoPE`` rotates the pair ``(x[2i], x[2i+1])`` -- adjacent dims.
* ``CosSinRoPE`` rotates ``(x[:d/2], x[d/2:])`` -- split halves.

They are not interchangeable, and a model built for one will produce garbage
with the other. The tests below assert each format's own invariant rather than
comparing the two.
"""

from __future__ import annotations

import pytest
import torch

from hpmesh.models.common.rope import (
    ComplexRoPE,
    CosSinRoPE,
    RoPE,
    RoPEConfig,
    _yarn_inv_freq,
)

DIM = 64
MAX_LEN = 128
TOKENS = 16
HEADS = 8


def _qk(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(TOKENS, HEADS, DIM, generator=g),
        torch.randn(TOKENS, HEADS, DIM, generator=g),
    )


# -- cache construction ------------------------------------------------------


def test_complex_cache_holds_half_width_complex_values() -> None:
    rope = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    assert rope.cache.shape == (MAX_LEN, DIM // 2)
    assert rope.cache.is_complex()


def test_cossin_cache_holds_full_width_real_values() -> None:
    rope = CosSinRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    assert rope.cache.shape == (MAX_LEN, DIM * 2)
    assert not rope.cache.is_complex()


def test_cache_is_not_persisted_into_checkpoints() -> None:
    """The cache is derived from the config -- saving it would only go stale."""
    rope = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    assert "cache" not in rope.state_dict()


# -- forward contracts -------------------------------------------------------


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_forward_returns_query_and_key_with_input_shapes(cls: type[RoPE]) -> None:
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, k = _qk()
    with torch.no_grad():
        q_out, k_out = rope(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_forward_returns_only_query_when_key_is_none(cls: type[RoPE]) -> None:
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    with torch.no_grad():
        out = rope(q, None)
    assert isinstance(out, torch.Tensor)
    assert out.shape == q.shape


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_forward_preserves_dtype(cls: type[RoPE]) -> None:
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, k = _qk()
    with torch.no_grad():
        q_out, k_out = rope(q.bfloat16(), k.bfloat16())
    assert q_out.dtype is torch.bfloat16
    assert k_out.dtype is torch.bfloat16


# -- position handling -------------------------------------------------------


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_positions_select_the_matching_cache_row(cls: type[RoPE]) -> None:
    """Token t must be rotated by cache row ``positions[t]``, not row ``t``.

    Asserted on the gathered cache directly rather than on forward output: an
    output comparison cannot separate "used the wrong row" from "reordered the
    output the same way", which is exactly the bug worth catching.
    """
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    positions = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3])

    with torch.no_grad():
        gathered = rope._reshape_cache(q, positions)

    assert gathered.shape == (TOKENS, 1, rope.cache.shape[-1])
    for token in range(TOKENS):
        assert torch.equal(gathered[token, 0], rope.cache[positions[token]])


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_omitting_positions_uses_rows_in_order(cls: type[RoPE]) -> None:
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()

    with torch.no_grad():
        gathered = rope._reshape_cache(q, None)

    assert torch.equal(gathered[:, 0], rope.cache[:TOKENS])


@pytest.mark.parametrize("cls", [ComplexRoPE, CosSinRoPE])
def test_position_past_the_cache_is_rejected(cls: type[RoPE]) -> None:
    """An out-of-range position indexes no row; catching it beats silent garbage."""
    rope = cls(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    positions = torch.full((TOKENS,), MAX_LEN, dtype=torch.long)
    with pytest.raises(RuntimeError):
        # _assert_async defers the failure to the next kernel launch.
        with torch.no_grad():
            rope(q, None, positions)
        torch.cuda.synchronize() if torch.cuda.is_available() else None


# -- rotation alternatives ---------------------------------------------------


def test_token_zero_is_unrotated() -> None:
    """Position 0 is angle 0, so the rotation is the identity there."""
    rope = CosSinRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    with torch.no_grad():
        out = rope(q, None)
    torch.testing.assert_close(out[0], q[0], rtol=1e-5, atol=1e-6)


def test_complex_rope_inverse_undoes_the_forward_rotation() -> None:
    """Conjugating the cache is the inverse rotation -- the defining property."""
    rope = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, k = _qk()
    with torch.no_grad():
        q_rot, k_rot = rope(q, k)
        q_back, k_back = rope(q_rot, k_rot, inverse=True)
    torch.testing.assert_close(q_back, q, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k_back, k, rtol=1e-5, atol=1e-6)


def test_cossin_rope_rejects_inverse_rotation() -> None:
    """The split-half convention has no cached conjugate to reuse."""
    rope = CosSinRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q, _ = _qk()
    with pytest.raises(NotImplementedError):
        with torch.no_grad():
            rope(q, None, inverse=True)


def test_complex_rope_rotates_adjacent_dimension_pairs() -> None:
    """Pin the convention: (x[0], x[1]) rotates as one pair, not (x[0], x[d/2])."""
    rope = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q = torch.zeros(TOKENS, HEADS, DIM)
    q[1, 0, 0] = 1.0  # a single adjacent-pair component at position 1
    with torch.no_grad():
        out = rope(q, None)
    # Only dims 0 and 1 of that position may move.
    assert out[1, 0, 0] != 0.0
    assert out[1, 0, 1] != 0.0
    assert torch.equal(out[1, 0, 2:], torch.zeros(DIM - 2))


def test_cossin_rope_rotates_split_halves() -> None:
    """The other convention: x[0] pairs with x[d/2], not x[1]."""
    rope = CosSinRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    q = torch.zeros(TOKENS, HEADS, DIM)
    q[1, 0, 0] = 1.0
    with torch.no_grad():
        out = rope(q, None)
    assert out[1, 0, 0] != 0.0
    assert out[1, 0, DIM // 2] != 0.0
    # Adjacent dim 1 stays untouched under this convention.
    assert out[1, 0, 1] == 0.0


# -- scaling modes -----------------------------------------------------------


def test_yarn_scaling_changes_the_cache() -> None:
    """A yarn config with rope_factor > 1 must rescale frequencies."""
    plain = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    yarn = ComplexRoPE(
        RoPEConfig(
            dim=DIM,
            max_context_length=MAX_LEN,
            scaling="yarn",
            rope_factor=4.0,
            original_seq_len=64,
        )
    )
    assert not torch.equal(plain.cache, yarn.cache)


def test_llama_scaling_changes_the_cache() -> None:
    plain = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    llama = ComplexRoPE(
        RoPEConfig(dim=DIM, max_context_length=MAX_LEN, scaling="llama")
    )
    assert not torch.equal(plain.cache, llama.cache)


def test_cossin_rope_rejects_llama_scaling() -> None:
    with pytest.raises(NotImplementedError):
        CosSinRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN, scaling="llama"))


def test_yarn_factor_of_one_is_a_no_op() -> None:
    """``scaling='yarn'`` only engages above 1.0, so 1.0 must match plain rope."""
    plain = ComplexRoPE(RoPEConfig(dim=DIM, max_context_length=MAX_LEN))
    yarn = ComplexRoPE(
        RoPEConfig(dim=DIM, max_context_length=MAX_LEN, scaling="yarn", rope_factor=1.0)
    )
    assert torch.equal(plain.cache, yarn.cache)


def test_yarn_truncate_controls_fractional_cutoffs() -> None:
    """truncate=True floors/ceils the beta cutoffs; False keeps them fractional."""
    truncated = _yarn_inv_freq(DIM, 10000.0, 4.0, 32.0, 1.0, 64, True)
    fractional = _yarn_inv_freq(DIM, 10000.0, 4.0, 32.0, 1.0, 64, False)
    assert not torch.equal(truncated, fractional)
