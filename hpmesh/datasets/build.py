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

Same shape as ``components/optimizer/lr_scheduler.build_lr_scheduler``, which exists for
the same reason. Both take the whole run config and read their own fields off
it rather than taking those fields restated as loose scalars: a scalar restated
at the call site can drift from the field the config carries, and half of the
ones this used to take had already stopped being read at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import grain.python as grain

from .collators import TextCollator
from .loader import BaseDataLoader, GrainDataLoader
from .packing import build_concat_then_split_packing, build_first_fit_packing
from .random_data import RandomTokenDataLoader
from .text.text import (
    DATASETS,
    make_local_jsonl,
    make_local_jsonl_sft,
    make_local_jsonl_sft_multiturn,
)
from .types import DatasetBuildContext, DatasetIterationPolicy

if TYPE_CHECKING:
    from ..config import HybridMeshConfig

__all__ = ["build_dataloader"]


def build_dataloader(
    config: HybridMeshConfig,
    *,
    dp_rank: int,
    dp_world_size: int,
    num_tokens_per_batch: int,
    repeat: bool = True,
    dataset: str | None = None,
) -> BaseDataLoader:
    """Build the loader a :class:`~hpmesh.config.HybridMeshConfig` names.

    The config answers everything that describes the *run*; the keyword
    arguments answer the things it cannot. ``config.dataloader`` (a
    ``DataloaderConfig``) names the corpus and how to shuffle, tokenize and pack
    it, while the flat view on the same object supplies the scalars that go with
    it -- ``seed``, ``vocab_size``, ``global_batch_size``, ``max_seq_len``.

    ``dp_rank``/``dp_world_size`` say which slice of the corpus this process
    reads, and ``num_tokens_per_batch`` is the per-rank token count, matching
    torchtitan's ``num_tokens_per_microbatch_per_dp_rank``. The Grain loader
    divides every dataset's rows among ``dp_world_size`` ranks and hands each
    one exactly that many tokens, so the DP slice the trainer used to perform
    no longer exists on this path.

    ``repeat=False`` builds a finite, single-pass loader -- the validation
    loop's shape: a bounded pass sets it so exhaustion (not a step count) ends
    the pass. ``dataset`` overrides ``config.dataloader.dataset`` for that same
    caller, so validation can score a held-out corpus without forking the
    config. Both default to the training behaviour.
    """
    dataloader_config = config.dataloader
    max_context_length = config.max_seq_len
    dataset_name = dataloader_config.dataset if dataset is None else dataset
    if dataset_name == "random":
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
        dataset_name not in {"local_jsonl", "local_jsonl_sft"}
        and dataset_name not in DATASETS
    )
    if is_multimodal:
        try:
            from .multimodal.mm_collator import MultiModalCollator
            from .multimodal.mm_datasets import (
                MM_DATASETS,
                build_mm_sample_packing,
            )
        except ImportError as exc:
            raise ImportError(
                f"dataset {dataset_name!r} is not one of the text "
                f"recipes {sorted(DATASETS)}, so it was looked up in the "
                "multimodal registry -- which failed to import. Multimodal "
                "recipes need the optional dependencies torchvision and Pillow "
                "(and av for video): install them with `pip install torchvision "
                "pillow av`, or name a text recipe instead."
            ) from exc
        if dataset_name not in MM_DATASETS:
            raise ValueError(
                f"unknown dataset {dataset_name!r}. Expected 'random', "
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
            tokenizer_path=dataloader_config.tokenizer_path,
            image_token=dataloader_config.mm_image_token,
            video_token=dataloader_config.mm_video_token,
            vision_start_token=dataloader_config.mm_vision_start_token,
            vision_end_token=dataloader_config.mm_vision_end_token,
            pad_token=dataloader_config.mm_pad_token,
        )
        recipe = MM_DATASETS[dataset_name]
        # Multimodal samples carry media lists alongside their token fields,
        # so they pack by whole documents (FirstFit) rather than concat-then-
        # split, and the collator reshapes the media into patches. This is not
        # ``--packing``: that selector names a text recipe, and the media path
        # ignores it.
        build_packing = build_mm_sample_packing
        packing_kwargs: dict[str, int] = {}
        collator = MultiModalCollator
    else:
        tokenizer = HuggingFaceTokenizer(
            tokenizer_path=dataloader_config.tokenizer_path
        )
        if dataset_name == "local_jsonl":
            recipe = make_local_jsonl(path=dataloader_config.dataset_path)
        elif dataset_name == "local_jsonl_sft":
            if dataloader_config.chat_renderer is not None:
                # Imported here, not at module scope: the renderers package is
                # an optional dependency, and a run that leaves chat_renderer
                # unset must not have to install it.
                from hpmesh.components.renderer import build_chat_renderer

                recipe = make_local_jsonl_sft_multiturn(
                    path=dataloader_config.dataset_path,
                    messages_field=dataloader_config.messages_field,
                    renderer=build_chat_renderer(
                        tokenizer=tokenizer,
                        renderer_name=dataloader_config.chat_renderer,
                    ),
                )
            else:
                recipe = make_local_jsonl_sft(
                    path=dataloader_config.dataset_path,
                    prompt_field=dataloader_config.prompt_field,
                    response_field=dataloader_config.response_field,
                )
        else:
            recipe = DATASETS[dataset_name]
        # Both recipes are built the same way and differ only in kind: the
        # discriminating work is in the packing node, never in the collator,
        # which is why the trainer needs no way to tell them apart.
        if dataloader_config.packing == "first_fit":
            build_packing = build_first_fit_packing
            # Always passed, at its default when unset: forwarding it only for
            # first_fit would make it a field whose value is silently dropped.
            packing_kwargs = {"num_packing_bins": dataloader_config.num_packing_bins}
        else:
            build_packing = build_concat_then_split_packing
            packing_kwargs = {}
        collator = TextCollator
    context = DatasetBuildContext(
        tokenizer=tokenizer,
        max_context_length=max_context_length,
        num_tokens_per_batch=num_tokens_per_batch,
        read_options=grain.ReadOptions(),
        max_num_documents=dataloader_config.max_num_documents,
    )
    # The iteration policy is spent here, on the graph it orders, rather than
    # handed down: the loader takes the built graph, so a policy on the loader
    # would be a parameter no line of it reads. What does reach the loader is
    # what batching -- not ordering -- reads: the prefetch depth, the drop
    # policy, and the token batch the collator assembles into.
    graph = build_packing(
        recipe,
        **packing_kwargs,
        context=context,
        dataset_iteration_policy=DatasetIterationPolicy(
            seed=config.seed,
            shuffle=dataloader_config.shuffle,
            repeat=repeat,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            streaming_shuffle_buffer_size=dataloader_config.streaming_shuffle_buffer_size,
        ),
    )
    return GrainDataLoader(
        graph,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        max_context_length=max_context_length,
        num_tokens_per_batch=num_tokens_per_batch,
        collator=collator,
        num_prefetch_batches=dataloader_config.num_prefetch_batches,
        max_num_documents=dataloader_config.max_num_documents,
    )
