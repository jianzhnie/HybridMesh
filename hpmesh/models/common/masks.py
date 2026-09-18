"""Flex-attention mask builders.

Lifted from torchtitan ``models/common/attention.py`` (the three helpers the HF
backend uses). They are self-contained -- a mask *modifier* is just a predicate
over ``(batch, head, query_idx, key_idx)`` that flex compiles into a BlockMask --
so there is no reason to pull in that file's 951 lines of attention modules.

Why a BlockMask at all, rather than ``is_causal=True``: the package feeds
documents packed end-to-end into one sequence, so a causal mask alone would let
token 0 of document N attend to the tail of document N-1. Expressing "causal AND
same document" needs a mask modifier. See ``HFTransformerModel.forward`` for
where these get attached.
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    and_masks,
    create_block_mask,
)

__all__ = [
    "and_masks",
    "create_attention_mask",
    "get_causal_mask_mod",
    "get_document_mask_mod",
]


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


_compiled_create_block_mask = torch.compile(create_block_mask)


def create_attention_mask(*args, **kwargs):
    """Build a BlockMask, with the (re)compilation cached across calls."""
    return _compiled_create_block_mask(*args, **kwargs)
