"""Shared data-pipeline types.

Vendored from torchtitan ``components/data/types.py``. Both dataclasses are
frozen values passed down the build, never mutated, so ``slots=True`` is safe
here -- nothing subclasses them.
"""

from __future__ import annotations

from dataclasses import dataclass

import grain.python as grain

from ..components.tokenizer import BaseTokenizer

__all__ = ["DatasetBuildContext", "DatasetIterationPolicy"]


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetBuildContext:
    """Runtime values shared while building the data pipeline."""

    tokenizer: BaseTokenizer
    max_context_length: int
    num_tokens_per_batch: int
    read_options: grain.ReadOptions
    max_num_documents: int | None = None

    def __post_init__(self) -> None:
        if self.max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        if self.num_tokens_per_batch <= 0:
            raise ValueError("num_tokens_per_batch must be positive")
        if self.max_num_documents is not None and self.max_num_documents <= 0:
            raise ValueError("max_num_documents must be positive")


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetIterationPolicy:
    """Controls dataset order, repetition, and data-parallel ownership."""

    seed: int
    shuffle: bool
    repeat: bool
    dp_rank: int
    dp_world_size: int
    streaming_shuffle_buffer_size: int

    def __post_init__(self) -> None:
        if self.dp_world_size <= 0:
            raise ValueError("dp_world_size must be positive")
        if not 0 <= self.dp_rank < self.dp_world_size:
            raise ValueError(
                f"dp_rank must be in [0, {self.dp_world_size}), got {self.dp_rank}"
            )
        if self.streaming_shuffle_buffer_size <= 0:
            raise ValueError("streaming_shuffle_buffer_size must be positive")
