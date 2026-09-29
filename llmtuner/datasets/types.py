"""Shared data-pipeline types.

Vendored from torchtitan ``components/data/types.py``. Both dataclasses are
frozen values passed down the build, never mutated, so ``slots=True`` is safe
here -- nothing subclasses them.

``Batch`` (the synthetic path's micro-batch container) also lives here rather
than in ``random_data.py``: consumers of the *type* -- the model wrapper's
``preprocess_inputs``, the trainer's batch handling -- must not have to import
the synthetic source itself. A models-layer file importing a data source was
the dependency inversion that placement caused.
"""

from __future__ import annotations

from dataclasses import dataclass

import grain.python as grain
import torch

from ..components.tokenizer import BaseTokenizer

__all__ = [
    "Batch",
    "DatasetBuildContext",
    "DatasetIterationPolicy",
    "require_positive",
]


def require_positive(name: str, value: int) -> None:
    """Raise unless ``value`` is positive, naming ``name``.

    The data pipeline's one shape check, shared by the dataclasses below and by
    the builders that take the same numbers as plain parameters (the loader's
    ``max_num_documents``, the packing builders' ``num_packing_bins``, the
    synthetic loader's ``dp_world_size``). It used to be three copied checks --
    with the message text spelled out at each site, so the wording could drift
    between the two entry points for the same value.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetBuildContext:
    """Runtime values shared while building the data pipeline."""

    tokenizer: BaseTokenizer
    max_context_length: int
    num_tokens_per_batch: int
    read_options: grain.ReadOptions
    max_num_documents: int | None = None

    def __post_init__(self) -> None:
        require_positive("max_context_length", self.max_context_length)
        require_positive("num_tokens_per_batch", self.num_tokens_per_batch)
        if self.max_num_documents is not None:
            require_positive("max_num_documents", self.max_num_documents)


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
        require_positive("dp_world_size", self.dp_world_size)
        if not 0 <= self.dp_rank < self.dp_world_size:
            raise ValueError(
                f"dp_rank must be in [0, {self.dp_world_size}), got {self.dp_rank}"
            )
        if self.streaming_shuffle_buffer_size <= 0:
            raise ValueError("streaming_shuffle_buffer_size must be positive")


@dataclass
class Batch:
    """One micro-batch, on CPU: ``input_ids`` and ``labels`` of shape ``(B, T)``."""

    input_ids: torch.Tensor
    labels: torch.Tensor
