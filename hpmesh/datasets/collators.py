"""Conversion from dataset rows to trainer batches.

Vendored from torchtitan ``components/data/collators.py``. A collator is
constructed directly with the build context and reads what it needs from there.
The padding rules are unchanged: padded
positions are `arange % max_context_length`, not zeros, so a padded row looks
like a fresh document rather than a continuation of the previous one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeAlias

import torch

from ..accelerator.device import should_use_pin_memory
from ..components.loss import IGNORE_INDEX
from .dataset import TextSequence
from .types import DatasetBuildContext

__all__ = ["Collator", "HAS_PIN_MEMORY", "TextCollator", "TrainerBatch"]


# The input dict holds the model's forward kwargs, labels, and
# ``num_valid_tokens``, the number of labels that contribute to the loss.
# Collators over token labels count them here so the trainer does not rescan
# every batch on the critical path; the trainer pops the count before the batch
# reaches the model.
TrainerBatch: TypeAlias = dict[str, Any]

# Page-locked batches let the trainer issue an async host-to-device copy; a copy
# out of pageable memory is synchronous whatever ``non_blocking`` says. There has
# to be an accelerator to pin for -- allocating with ``pin_memory=True`` raises
# without one -- so CPU-only runs fall back to ordinary pageable memory.
HAS_PIN_MEMORY = should_use_pin_memory()


class Collator(ABC):
    """Row-to-batch conversion."""

    @abstractmethod
    def __call__(self, rows: Sequence[Any]) -> TrainerBatch: ...

    def num_rows_per_batch(self) -> int:
        """Return the number of dataset rows consumed by one trainer batch."""
        return 1


class TextCollator(Collator):
    """Packs text rows into one page-locked, pre-padded token batch."""

    def __init__(self, *, context: DatasetBuildContext) -> None:
        self._num_tokens_per_batch = context.num_tokens_per_batch
        self._max_context_length = context.max_context_length

    def __call__(self, rows: Sequence[TextSequence]) -> TrainerBatch:
        num_tokens = sum(len(row.input_ids) for row in rows)
        if num_tokens > self._num_tokens_per_batch:
            raise ValueError("text rows exceed the configured token batch")

        size = self._num_tokens_per_batch
        input_ids = torch.zeros(size, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY)
        positions = torch.zeros(size, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY)
        labels = torch.full(
            (size,), IGNORE_INDEX, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY
        )
        padding_mask = torch.ones(size, dtype=torch.bool, pin_memory=HAS_PIN_MEMORY)

        torch.cat(
            [torch.as_tensor(row.input_ids) for row in rows],
            out=input_ids[:num_tokens],
        )
        torch.cat(
            [torch.as_tensor(row.labels) for row in rows],
            out=labels[:num_tokens],
        )
        torch.cat(
            [
                (
                    torch.arange(len(row.input_ids))
                    if row.positions is None
                    else torch.as_tensor(row.positions)
                )
                for row in rows
            ],
            out=positions[:num_tokens],
        )
        torch.cat(
            [
                (
                    torch.zeros(len(row.input_ids), dtype=torch.bool)
                    if row.padding_mask is None
                    else torch.as_tensor(row.padding_mask, dtype=torch.bool)
                )
                for row in rows
            ],
            out=padding_mask[:num_tokens],
        )

        pad_len = self._num_tokens_per_batch - num_tokens
        if pad_len:
            torch.arange(pad_len, out=positions[num_tokens:])
            positions[num_tokens:].remainder_(self._max_context_length)

        return {
            "input": input_ids,
            "labels": labels,
            "positions": positions,
            "padding_mask": padding_mask,
            "num_valid_tokens": int((labels != IGNORE_INDEX).sum()),
        }
