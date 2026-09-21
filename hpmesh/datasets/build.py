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
the same reason. Both take the whole run config and read their own fields off
it rather than taking those fields restated as loose scalars: a scalar restated
at the call site can drift from the field the config carries, and half of the
ones this used to take had already stopped being read at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import grain.python as grain

from .collators import TextCollator
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
    from ..trainer.config import HybridMeshConfig

__all__ = ["build_dataloader"]


def build_dataloader(
    config: HybridMeshConfig,
    *,
    dp_rank: int,
    dp_world_size: int,
    num_tokens_per_batch: int,
) -> BaseDataLoader:
    """Build the loader a :class:`~hpmesh.trainer.config.HybridMeshConfig` names.

    The config answers everything that describes the *run*; the three keyword
    arguments answer the two things it cannot. ``config.dataloader`` (a
    ``DataloaderConfig``) names the corpus and how to shuffle, tokenize and pack
    it, while the flat view on the same object supplies the scalars that go with
    it -- ``seed``, ``vocab_size``, ``global_batch_size``, ``max_seq_len``.

    The keywords are all per-rank: ``dp_rank``/``dp_world_size`` say which slice
    of the corpus this process reads, and ``num_tokens_per_batch`` is the
    per-rank token count, matching torchtitan's
    ``num_tokens_per_microbatch_per_dp_rank``. The Grain loader divides every
    dataset's rows among ``dp_world_size`` ranks and hands each one exactly that
    many tokens, so the DP slice the trainer used to perform no longer exists on
    this path.
    """
    dataset_config = config.dataloader
    max_context_length = config.max_seq_len
    if dataset_config.dataset == "random":
        return RandomTokenDataLoader(
            seed=config.seed,
            vocab_size=config.vocab_size,
            batch_size=config.global_batch_size,
            seq_len=max_context_length,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )

    # A name that is neither a text recipe nor ``local_jsonl`` is tried
    # against the multimodal registry. That import stays inside the branch:
    # the multimodal subtree pulls in torchvision (and a video backend behind
    # it), and a text-only run must not have to install them. Unknown recipe
    # names are rejected here, at build time, rather than in
    # ``DataloaderConfig.__post_init__``: the registries live in this package,
    # and the config layer must not import them. Checked before the tokenizer
    # is built so a bad name fails fast without loading tokenizer assets.
    is_multimodal = (
        dataset_config.dataset != "local_jsonl"
        and dataset_config.dataset not in DATASETS
    )
    if is_multimodal:
        try:
            from .hf.multimodal.mm_collator import MultiModalCollator
            from .hf.multimodal.mm_datasets import MM_DATASETS, MMSamplePackingConfig
        except ImportError as exc:
            raise ImportError(
                f"dataset {dataset_config.dataset!r} is not one of the text "
                f"recipes {sorted(DATASETS)}, so it was looked up in the "
                "multimodal registry -- which failed to import. Multimodal "
                "recipes need the optional dependencies torchvision and Pillow "
                "(and av for video): install them with `pip install torchvision "
                "pillow av`, or name a text recipe instead."
            ) from exc
        if dataset_config.dataset not in MM_DATASETS:
            raise ValueError(
                f"unknown dataset {dataset_config.dataset!r}. Expected 'random', "
                f"'local_jsonl', a text recipe {sorted(DATASETS)}, or a "
                f"multimodal recipe {sorted(MM_DATASETS)}"
            )

    # Imported here, not at module scope: building the tokenizer pulls in
    # ``tokenizers``/``jinja2``, and a random-token run should not have to
    # have them installed.
    from hpmesh.components.tokenizer import HuggingFaceTokenizer

    if is_multimodal:
        from hpmesh.components.tokenizer import MultiModalTokenizer

        tokenizer = MultiModalTokenizer(
            tokenizer_path=dataset_config.tokenizer_path,
            image_token=dataset_config.mm_image_token,
            video_token=dataset_config.mm_video_token,
            vision_start_token=dataset_config.mm_vision_start_token,
            vision_end_token=dataset_config.mm_vision_end_token,
            pad_token=dataset_config.mm_pad_token,
        )
        recipe = MM_DATASETS[dataset_config.dataset]
        # Multimodal samples carry media lists alongside their token fields,
        # so they pack by whole documents (FirstFit) rather than concat-then-
        # split, and the collator reshapes the media into patches.
        packing_config = MMSamplePackingConfig(dataset=recipe)
        collator = MultiModalCollator
    else:
        tokenizer = HuggingFaceTokenizer(tokenizer_path=dataset_config.tokenizer_path)
        recipe = (
            make_local_jsonl(path=dataset_config.dataset_path)
            if dataset_config.dataset == "local_jsonl"
            else DATASETS[dataset_config.dataset]
        )
        packing_config = ConcatThenSplitPackingConfig(dataset=recipe)
        collator = TextCollator
    context = DatasetBuildContext(
        tokenizer=tokenizer,
        max_context_length=max_context_length,
        num_tokens_per_batch=num_tokens_per_batch,
        read_options=grain.ReadOptions(),
        max_num_documents=dataset_config.max_num_documents,
    )
    # The loader's config is built first and the graph filled in after,
    # because ``build_dataset_iteration_policy`` derives the policy the
    # graph is built with *from* that config -- restating the seed and
    # shuffle flags here instead would let the two drift, and a shuffle
    # flag the graph never sees is a silent no-op. ``dataset`` is the one
    # field the policy does not read, so the placeholder cannot leak.
    loader_config = GrainDataLoaderConfig(
        dataset=None,
        collator=collator,
        seed=config.seed,
        shuffle=dataset_config.shuffle,
        streaming_shuffle_buffer_size=dataset_config.streaming_shuffle_buffer_size,
        num_prefetch_batches=dataset_config.num_prefetch_batches,
        max_num_documents=dataset_config.max_num_documents,
    )
    graph = packing_config.build(
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
