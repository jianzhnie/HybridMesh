# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetIterationPolicy:
    """Controls dataset order, repetition, and data-parallel ownership."""

    seed: int
    shuffle: bool
    repeat: bool
    dp_rank: int
    dp_world_size: int
    streaming_shuffle_buffer_size: int
