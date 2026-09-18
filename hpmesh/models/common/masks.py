"""Flex-attention mask builders and variable-length metadata.

Lifted from torchtitan ``models/common/attention.py`` -- the mask helpers and the
packed-document metadata builder, none of which need that file's attention
modules. A mask *modifier* is just a predicate over
``(batch, head, query_idx, key_idx)`` that flex compiles into a BlockMask, so it
stands on its own.

Why a BlockMask at all, rather than ``is_causal=True``: the package feeds
documents packed end-to-end into one sequence, so a causal mask alone would let
token 0 of document N attend to the tail of document N-1. Expressing "causal AND
same document" needs a mask modifier. See ``HFTransformerModel.forward`` for
where these get attached.

Every helper here reads document boundaries from the same convention --
``positions`` resets to 0 at each packed document's first token -- so a mask and
the varlen metadata built from one ``positions`` tensor always agree.

Shape legend, scoped to this file: ``T`` = tokens, ``D`` = model dimension.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    and_masks,
    create_block_mask,
)

__all__ = [
    "and_masks",
    "create_attention_mask",
    "create_varlen_metadata_for_document",
    "get_causal_mask_mod",
    "get_document_mask_mod",
    "get_efficient_causal_mask_mod_for_packed_document",
    "get_fixed_block_mask_mod",
    "get_sliding_window_mask_mod",
    "round_up",
    "VarlenMetadata",
]


class VarlenMetadata(NamedTuple):
    """Cumulative sequence positions for queries and keys/values.

    ``cu_seq_*`` are the offsets that define document boundaries in a packed
    sequence; ``max_*`` are the longest single document, which the attention
    kernel uses to size its tiles.
    """

    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    max_q: int
    max_k: int


def round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the next multiple of ``multiple``."""
    return ((value + multiple - 1) // multiple) * multiple


def get_causal_mask_mod() -> _mask_mod_signature:
    """A mask modifier that prevents attention to future tokens."""

    def _causal_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return q_idx >= kv_idx

    return _causal_mask


def get_document_mask_mod(positions: torch.Tensor) -> _mask_mod_signature:
    """A mask modifier that prevents attention across document boundaries.

    Boundaries are where ``positions`` resets to 0, which marks the start of a
    new packed document.

    Args:
        positions: Per-token positions, shape ``[T]``, resetting to 0 at each
            document start.
    """
    doc_ids = torch.cumsum((positions == 0).int(), dim=0) - 1

    def document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return doc_ids[q_idx] == doc_ids[kv_idx]

    return document_mask


def get_efficient_causal_mask_mod_for_packed_document(
    positions: torch.Tensor,
) -> _mask_mod_signature:
    """A fast path for causal packed-document masking, to compose with causal.

    Same convention as ``get_document_mask_mod``, but instead of comparing
    document ids it resolves each query's document start into a lookup table.
    The causal mask supplies the upper bound ``kv_idx <= q_idx`` and this one
    supplies the lower bound ``doc_start[q_idx] <= kv_idx``; composed, they give
    same-document causal masking. It is manually tuned, which is why it coexists
    with the generic document-id mask rather than replacing it.

    Not intended for non-causal use -- the upper bound is the causal mask's job.
    """
    seq_len = positions.shape[0]
    document_starts = positions == 0
    document_id = torch.cumsum(document_starts.int(), dim=0).to(torch.int32) - 1
    token_idx = torch.arange(seq_len, device=positions.device, dtype=torch.int32)
    # One padded slot past the end holds the out-of-range sentinel, so a query
    # in a document whose start was not recorded cannot leak into another.
    offsets = torch.full(
        (round_up(seq_len + 1, 128),),
        seq_len,
        device=positions.device,
        dtype=torch.int32,
    )
    offsets.scatter_(
        0,
        torch.where(
            document_starts, document_id, torch.full_like(document_id, seq_len)
        ).to(torch.int64),
        torch.where(document_starts, token_idx, torch.full_like(token_idx, seq_len)),
    )

    def packed_document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return kv_idx >= offsets[document_id[q_idx]]

    return packed_document_mask


def get_fixed_block_mask_mod(fixed_block_size: int) -> _mask_mod_signature:
    """A mask modifier that only allows attention within the same fixed block.

    Args:
        fixed_block_size: number of tokens per block.
    """

    # Credit to @drisspg.
    def blocked_mask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        q_block = q_idx // fixed_block_size
        kv_block = kv_idx // fixed_block_size
        return q_block == kv_block

    blocked_mask_mod.__name__ = f"blocked_mask_mod_fixed_block_size_{fixed_block_size}"

    return blocked_mask_mod


def get_sliding_window_mask_mod(window_size: int) -> _mask_mod_signature:
    """A causal sliding-window mask: attend to self and the previous window.

    Args:
        window_size: the most tokens to attend to, including the current one.
            Must be >= 1; ``1`` attends to self only.

    Raises:
        ValueError: if ``window_size`` is below 1, which would mask out even the
            diagonal and produce all-zero attention rows.
    """

    if window_size < 1:
        raise ValueError(
            "window_size must be >= 1 for sliding window attention mask, "
            f"got {window_size}"
        )

    def sliding_window_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return (kv_idx <= q_idx) & (q_idx - kv_idx < window_size)

    sliding_window_mod.__name__ = f"sliding_window_mod_window_size_{window_size}"

    return sliding_window_mod


def create_varlen_metadata_for_document(
    positions: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None = None,
    max_num_documents: int | None = None,
    max_context_length: int | None = None,
) -> VarlenMetadata:
    """Build cumulative sequence lengths for variable-length attention.

    Document boundaries are detected where ``positions`` resets to 0, matching
    :func:`get_document_mask_mod`.

    Args:
        positions: Per-token position tensor with shape ``[T]``, resetting to 0
            at each document start.
        padding_mask: Per-token boolean tensor, true for padding. Separating
            padding resets from real document starts lets their metadata
            capacity be reserved independently.
        max_num_documents: Upper bound on non-padding document segments in this
            token batch. Setting it fixes the metadata's shape, which is what
            CUDA graph capture needs.
        max_context_length: Longest single document segment. Required with
            ``max_num_documents`` so the fixed shape can be computed without a
            device-to-host sync.

    Returns:
        VarlenMetadata over the packed sequence.

    Raises:
        ValueError: if ``max_num_documents`` is set without
            ``max_context_length``, which would force the very sync the fixed
            shape exists to avoid.
    """
    num_tokens = positions.shape[0]
    device = positions.device

    real_doc_starts = positions == 0
    padding_doc_starts = None
    if padding_mask is None:
        is_doc_start = real_doc_starts
    else:
        padding_mask = padding_mask.to(torch.bool)
        real_doc_starts = real_doc_starts & ~padding_mask
        padding_doc_starts = (positions == 0) & padding_mask
        is_doc_start = real_doc_starts | padding_doc_starts

    if max_num_documents is not None:
        if max_context_length is None:
            raise ValueError(
                "max_context_length is required when max_num_documents is set"
            )

        # A padding run longer than one context window would start a second
        # segment, so reserve for that many before the real capacity.
        max_num_padding_segments = (
            (num_tokens + max_context_length - 1) // max_context_length
            if padding_mask is not None
            else 0
        )
        max_num_segments = max_num_documents + max_num_padding_segments
        num_slots = max_num_segments + 1
        slot = torch.cumsum(is_doc_start, 0) - 1
        scatter_index = torch.where(
            is_doc_start & (slot < max_num_segments),
            slot,
            torch.full_like(slot, num_slots),
        )
        packed_cu_seqlens = torch.full(
            (num_slots + 1,), num_tokens, dtype=torch.int32, device=device
        )
        packed_cu_seqlens.scatter_(
            0,
            scatter_index,
            torch.arange(num_tokens, dtype=torch.int32, device=device),
        )
        # Deferred asserts: an overflow here would otherwise write past the
        # reserved slots and be caught much later, or not at all.
        torch._assert_async(real_doc_starts.sum() <= max_num_documents)
        if padding_doc_starts is not None:
            torch._assert_async(padding_doc_starts.sum() <= max_num_padding_segments)
        packed_cu_seqlens = packed_cu_seqlens[:num_slots]
        max_seqlen = max_context_length
    else:
        doc_starts = is_doc_start.nonzero(as_tuple=True)[0].to(torch.int32)
        packed_cu_seqlens = torch.cat(
            [
                doc_starts,
                torch.tensor([num_tokens], dtype=torch.int32, device=device),
            ]
        )
        seq_lengths = torch.diff(packed_cu_seqlens)

        if seq_lengths.numel() > 0:
            # Device-to-host sync, but only once per forward.
            max_seqlen = int(seq_lengths.max().item())
        else:
            max_seqlen = 0

    return VarlenMetadata(
        cu_seq_q=packed_cu_seqlens,
        cu_seq_k=packed_cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )


_compiled_create_block_mask = torch.compile(create_block_mask)


def create_attention_mask(*args, **kwargs):
    """Build a BlockMask, with the (re)compilation cached across calls."""
    return _compiled_create_block_mask(*args, **kwargs)
