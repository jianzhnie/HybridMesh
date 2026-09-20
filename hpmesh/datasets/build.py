# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Factory functions that turn configuration into a built object.

These sit one level above the modules whose classes they assemble, and that is
what they are for: ``loader.py`` and ``lr_scheduler.py`` hold one implementation
each and must not depend on their siblings, while a factory has to know all of
them. Building a dataloader means reading the recipe registry, constructing the
tokenizer, deriving the iteration policy and packing the graph -- four modules
whose only thing in common is that this function calls them.

The other direction is what does not belong here. A class that carries a builder
of its own suggests the data it produces is one of its fields, and for a config
it is not: ``DataloaderConfig`` is a parsed description, and building from it is
a separate act with its own arguments (which rank am I, how many tokens per
batch). Keeping them apart also keeps ``datasets/`` and ``components/`` free of
any reference back to ``trainer/``.

Same shape as ``components/optimizer.build_lr_scheduler``, which exists for
the same reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import grain.python as grain

from .hf.text import DATASETS, make_local_jsonl
from .loader import (
    BaseDataLoader,
    GrainDataLoader,
    GrainDataLoaderConfig,
    build_dataset_iteration_policy,
)
from .packing import ConcatThenSplitPackingConfig
from .random_data import RandomTokenDataLoader
from .types import DatasetBuildContext

if TYPE_CHECKING:
    from ..trainer.config import DataloaderConfig

__all__ = ["build_dataloader"]


def build_dataloader(
    config: DataloaderConfig,
    *,
    seed: int,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    dp_rank: int,
    dp_world_size: int,
    max_context_length: int,
    num_tokens_per_batch: int,
) -> BaseDataLoader:
    """Build the loader a :class:`~hpmesh.trainer.config.DataloaderConfig` names.

    ``num_tokens_per_batch`` is the per-rank token count, matching
    torchtitan's ``num_tokens_per_microbatch_per_dp_rank``: the Grain
    loader divides every dataset's rows among ``dp_world_size`` ranks and
    hands each one exactly that many tokens, so the DP slice the trainer
    used to perform no longer exists on this path.
    """
    if config.dataset == "random":
        return RandomTokenDataLoader(
            seed=seed,
            vocab_size=vocab_size,
            batch_size=batch_size,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )

    # Imported here, not at module scope: building the tokenizer pulls in
    # ``tokenizers``/``jinja2``, and a random-token run should not have to
    # have them installed.
    from hpmesh.components.tokenizer import HuggingFaceTokenizer

    tokenizer = HuggingFaceTokenizer(tokenizer_path=config.tokenizer_path)
    recipe = (
        make_local_jsonl(path=config.dataset_path)
        if config.dataset == "local_jsonl"
        else DATASETS[config.dataset]
    )
    context = DatasetBuildContext(
        tokenizer=tokenizer,
        max_context_length=max_context_length,
        num_tokens_per_batch=num_tokens_per_batch,
        read_options=grain.ReadOptions(),
        max_num_documents=config.max_num_documents,
    )
    # The loader's config is built first and the graph filled in after,
    # because ``build_dataset_iteration_policy`` derives the policy the
    # graph is built with *from* that config -- restating the seed and
    # shuffle flags here instead would let the two drift, and a shuffle
    # flag the graph never sees is a silent no-op. ``dataset`` is the one
    # field the policy does not read, so the placeholder cannot leak.
    loader_config = GrainDataLoaderConfig(
        dataset=None,
        seed=seed,
        shuffle=config.shuffle,
        streaming_shuffle_buffer_size=config.streaming_shuffle_buffer_size,
        num_prefetch_batches=config.num_prefetch_batches,
        max_num_documents=config.max_num_documents,
    )
    graph = ConcatThenSplitPackingConfig(dataset=recipe).build(
        context=context,
        dataset_iteration_policy=build_dataset_iteration_policy(
            loader_config, dp_rank=dp_rank, dp_world_size=dp_world_size
        ),
    )
    loader_config.dataset = graph
    return GrainDataLoader(
        loader_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        max_context_length=max_context_length,
        num_tokens_per_batch=num_tokens_per_batch,
    )
