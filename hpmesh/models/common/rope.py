"""Rotary position embeddings.

Vendored from torchtitan ``models/common/rope.py``. What changed:

* No ``Configurable``. The hyperparameters live in a plain ``RoPEConfig``
  dataclass and reach ``__init__`` as one argument, which is how hpmesh carries
  every other config (see ``models/spec.py``).
* ``spmd.no_typecheck()`` and ``spmd.local_map(...)`` are gone. The first was a
  type-checker suppression with no runtime effect; the second declared that
  ``_reshape_for_broadcast`` is a pure per-rank reshape, which is true as
  written -- the function only indexes ``rope_cache`` with token positions.
* ``_init_self_buffers`` is gone. It existed so a meta-device build could
  recompute the cache after ``to_empty()``; hpmesh builds real HF models, and a
  registered buffer already follows the module across ``.to()``.

The cache is recomputed for every position up to ``max_context_length`` at
construction time and stored as a non-persistent buffer, so it moves with the
model and stays out of checkpoints.

Shape suffix legend (per module):
  T = token positions, N = attention heads, H = head dimension
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

__all__ = [
    "ComplexRoPE",
    "CosSinRoPE",
    "RoPE",
    "RoPEConfig",
]


# TODO: This is an async bounds check rather than a host-side comparison so it
# does not force a device sync. It is a no-op under torch.compile, where the
# position range is already guaranteed by the BlockMask.
def _maybe_check_max_pos(positions: torch.Tensor, *, max_valid_pos: int) -> None:
    """Assert every position fits the cache, without syncing device to host.

    Uses ``torch._assert_async`` so the failure surfaces at a later kernel
    launch instead of blocking on a ``.item()`` call.
    """
    if torch.compiler.is_compiling():
        return
    torch._assert_async(
        torch.all(positions <= max_valid_pos),
        f"position_ids exceed {max_valid_pos=}",
    )


def _yarn_inv_freq(
    dim: int,
    base: float,
    rope_factor: float,
    beta_fast: float,
    beta_slow: float,
    original_seq_len: int,
    truncate: bool,
) -> torch.Tensor:
    """Shared YaRN ("NTK-by-parts") inverse-frequency computation.

    Single source of truth for both ``ComplexRoPE`` and ``CosSinRoPE`` so the
    two cache formats are guaranteed to agree. Follows the YaRN paper / HF
    convention: ``low <- beta_fast`` (extrapolation boundary), ``high <-
    beta_slow`` (interpolation boundary). ``truncate`` floors/ceils the cutoffs
    (DeepSeek style); ``truncate=False`` keeps fractional cutoffs (gpt-oss
    style). The range is always clamped to ``[0, dim - 1]``. The YaRN
    attention "mscale" is intentionally NOT applied here -- the rope stays a
    pure rotation and the model folds mscale into its softmax scale.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_seq_len / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001

    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(
        0, 1
    )
    return inv_freq / rope_factor * ramp + inv_freq * (1 - ramp)


@dataclass
class RoPEConfig:
    """RoPE hyperparameters.

    The scaling fields are grouped by the mode that reads them: ``llama`` reads
    ``scaling_factor``/``low_freq_factor``/``high_freq_factor``/
    ``original_max_position_embeddings``, ``yarn`` reads the ``rope_factor``/
    ``beta_*``/``original_seq_len``/``truncate`` set. Fields for an unused mode
    are kept at their defaults, and ``CosSinRoPE`` rejects ``llama`` outright.
    """

    dim: int
    max_context_length: int
    theta: float = 10000.0
    scaling: Literal["none", "llama", "yarn"] = "none"
    # llama scaling params
    scaling_factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192
    # yarn scaling params
    rope_factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_seq_len: int = 4096
    truncate: bool = True


class RoPE(nn.Module):
    """Rotary position embedding, with the cache format left to subclasses.

    Concrete subclasses pick a cache layout -- complex exponentials
    (``ComplexRoPE``) or concatenated cos/sin (``CosSinRoPE``) -- and therefore
    how the rotation is applied. Everything else (cache sizing, position
    lookup, broadcast shape) is shared here.
    """

    def __init__(self, config: RoPEConfig):
        super().__init__()
        self.config = config
        # Non-persistent: the cache is derived from the config, so saving it
        # would only bloat checkpoints and risk it going stale on a reload.
        self.register_buffer("cache", self._precompute_cache(), persistent=False)

    def _precompute_cache(self) -> torch.Tensor:
        """Build the reusable cache for all positions up to ``max_context_length``."""
        raise NotImplementedError

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a cache aligned to ``query`` and ``positions``.

        Args:
            query: Query tensor with shape ``[T, N, H]``.
            positions: Optional position IDs with shape ``[T]``.
        """
        raise NotImplementedError

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply a prepared RoPE cache to query and optional key.

        Args:
            query: Query tensor with shape ``[T, N, H]``.
            key: Optional key tensor with the same leading dimensions as
                ``query``. If ``None``, only ``query`` is rotated and returned.
            rope_cache: Prepared cache broadcastable to ``query`` and ``key``
                according to the concrete RoPE format.
            inverse: Whether to apply the inverse rotation.

        Returns:
            Rotated query tensor when ``key`` is ``None``; otherwise rotated
            query and key tensors with the same shapes and dtypes as inputs.
        """
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and optional key tensors."""
        reshaped_cache = self._reshape_cache(query, positions)
        return self.apply_rotary_emb(query, key, reshaped_cache, inverse=inverse)


class ComplexRoPE(RoPE):
    """RoPE over adjacent dimension pairs, stored as complex exponentials.

    The cache holds ``cis(freq * t)`` values, so applying the rotation is one
    complex multiply. This is the format Llama and DeepSeek use.
    """

    def _precompute_cache(self) -> torch.Tensor:
        """Precompute complex cis values, shape ``(max_context_length, dim / 2)``."""
        cfg = self.config
        dim = cfg.dim
        end = cfg.max_context_length
        theta = cfg.theta

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        if cfg.scaling == "llama":
            scaling_factor = cfg.scaling_factor
            low_freq_factor = cfg.low_freq_factor
            high_freq_factor = cfg.high_freq_factor
            original_max_position_embeddings = cfg.original_max_position_embeddings
            wavelen = 2 * math.pi / freqs
            high_freq_wavelen = original_max_position_embeddings / high_freq_factor
            low_freq_wavelen = original_max_position_embeddings / low_freq_factor
            freqs = torch.where(
                wavelen > low_freq_wavelen, freqs / scaling_factor, freqs
            )
            smooth_factor = (
                original_max_position_embeddings / wavelen - low_freq_factor
            ) / (high_freq_factor - low_freq_factor)
            smoothed_freqs = (
                1 - smooth_factor
            ) * freqs / scaling_factor + smooth_factor * freqs
            is_medium_freqs = ~(wavelen < high_freq_wavelen) * ~(
                wavelen > low_freq_wavelen
            )
            freqs = torch.where(is_medium_freqs, smoothed_freqs, freqs)
        elif cfg.scaling == "yarn" and cfg.rope_factor > 1.0:
            # YaRN (DeepSeek V3 style)
            freqs = _yarn_inv_freq(
                dim,
                theta,
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
                cfg.original_seq_len,
                cfg.truncate,
            )

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        # complex64; the imaginary part carries the rotation angle.
        return torch.polar(torch.ones_like(freqs), freqs)

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the complex cache shaped ``(T, 1, dim / 2)`` for broadcast."""
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        # Half the width: each complex value covers a pair of real dimensions.
        complex_query_shape = (*query.shape[:-1], query.shape[-1] // 2)
        return _reshape_for_broadcast(self.cache, complex_query_shape, positions)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply complex RoPE using adjacent-dim pairs."""
        if inverse:
            # The inverse rotation is the conjugate -- cheap for a cached cis.
            rope_cache = rope_cache.conj()

        xq_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
        query_out = torch.view_as_real(xq_ * rope_cache).flatten(-2).type_as(query)
        if key is None:
            return query_out

        xk_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
        key_out = torch.view_as_real(xk_ * rope_cache).flatten(-2).type_as(key)
        return query_out, key_out


class CosSinRoPE(RoPE):
    """RoPE over split halves, stored as concatenated cos/sin tables.

    The cache holds ``[cos, sin]`` side by side, so applying the rotation is an
    elementwise multiply and add. This is the format gpt-oss and the vision
    towers use.
    """

    def _precompute_cache(self) -> torch.Tensor:
        """Precompute cos/sin values, shape ``(max_context_length, dim * 2)``."""
        cfg = self.config
        dim = cfg.dim
        max_context_length = cfg.max_context_length
        base = cfg.theta

        if cfg.scaling == "llama":
            raise NotImplementedError("Cos/sin RoPE does not support Llama scaling.")

        if cfg.scaling == "yarn" and cfg.rope_factor > 1.0:
            inv_freq = _yarn_inv_freq(
                dim,
                base,
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
                cfg.original_seq_len,
                cfg.truncate,
            )
        else:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )

        t = torch.arange(
            max_context_length, dtype=inv_freq.dtype, device=inv_freq.device
        )
        freqs = torch.outer(t, inv_freq).float()
        # Doubled so cos and sin each cover a full head_dim.
        theta = torch.cat([freqs, freqs], dim=-1)

        cos = theta.cos()
        sin = theta.sin()
        return torch.cat([cos, sin], dim=-1)

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the cos/sin cache shaped ``(T, 1, dim * 2)`` for broadcast."""
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        return _reshape_for_broadcast(self.cache, query.shape, positions)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply cos/sin RoPE using the rotate-half convention."""
        if inverse:
            raise NotImplementedError("CosSinRoPE does not support inverse rotation.")

        head_dim = query.shape[-1]
        cos = rope_cache[..., :head_dim]
        sin = rope_cache[..., head_dim:]
        query_f = query.float()
        xq_out = (query_f * cos) + (CosSinRoPE._rotate_half(query_f) * sin)
        if key is None:
            return xq_out.type_as(query)

        key_f = key.float()
        xk_out = (key_f * cos) + (CosSinRoPE._rotate_half(key_f) * sin)
        return xq_out.type_as(query), xk_out.type_as(key)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


def _reshape_for_broadcast(
    rope_cache: torch.Tensor,
    query_shape: torch.Size | tuple[int, ...],
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reshape a RoPE cache for broadcasting with query/key tensors.

    ``cache_width`` is ``head_dim * 2`` for ``CosSinRoPE`` and ``head_dim // 2``
    for ``ComplexRoPE``; either way the head axis is inserted so the cache
    multiplies every head identically.
    """
    cache_width = rope_cache.shape[-1]
    num_tokens = query_shape[0]
    if positions is None:
        rope_cache = rope_cache[:num_tokens]
    else:
        rope_cache = rope_cache[positions]
    return rope_cache.view(num_tokens, 1, cache_width)
