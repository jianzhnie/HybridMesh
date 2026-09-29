
from __future__ import annotations

import numpy as np

from ...components.loss import IGNORE_INDEX
from ..dataset import (
    TextSequence,
)


def packing_output_is_full(packing_output: dict[str, np.ndarray]) -> bool:
    """Return whether concat-then-split filled the entire token batch."""
    return bool(np.all(np.asarray(packing_output["input_ids_segment_ids"]) != 0))


def text_sequence_to_packing_input(
    text_sequence: TextSequence,
    *,
    max_context_length: int,
) -> dict[str, np.ndarray]:
    """Convert a `TextSequence` to the array dictionary expected by text packing.

    Missing positions become `0..num_tokens-1`.
    """
    positions = text_sequence.positions
    if positions is None:
        positions = (
            np.arange(len(text_sequence.input_ids), dtype=np.int64) % max_context_length
        )
    padding_mask = text_sequence.padding_mask
    if padding_mask is None:
        padding_mask = np.zeros(len(text_sequence.input_ids), dtype=np.bool_)
    return {
        "input_ids": np.asarray(text_sequence.input_ids),
        "labels": np.asarray(text_sequence.labels),
        "positions": np.asarray(positions),
        "padding_mask": np.asarray(padding_mask),
    }


def packing_output_to_text_sequence(
    packing_output: dict[str, np.ndarray],
    *,
    max_context_length: int,
) -> TextSequence:
    """Finalize packed text by masking padding and canonicalizing positions."""
    segment_ids = np.asarray(packing_output["input_ids_segment_ids"])
    padding_mask = np.asarray(packing_output["padding_mask"], dtype=np.bool_).copy()
    padding_mask[segment_ids == 0] = True
    labels = np.asarray(packing_output["labels"]).copy()
    labels[padding_mask] = IGNORE_INDEX

    # A zero starts a document. For [0, 1, 2, 0, 1], segment_starts is
    # [0, 0, 0, 3, 3], so subtracting it restores [0, 1, 2, 0, 1].
    boundaries = np.asarray(packing_output["positions"]) == 0
    token_indices = np.arange(len(boundaries), dtype=np.int64)
    segment_starts = np.maximum.accumulate(np.where(boundaries, token_indices, 0))
    positions = token_indices - segment_starts

    packing_padding = segment_ids == 0
    if np.any(packing_padding):
        first_padding_token = int(np.flatnonzero(packing_padding)[0])
        positions[first_padding_token:] = (
            np.arange(len(positions) - first_padding_token) % max_context_length
        )

    return TextSequence(
        input_ids=np.asarray(packing_output["input_ids"]),
        labels=labels,
        positions=positions,
        padding_mask=padding_mask,
    )