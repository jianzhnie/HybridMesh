
from __future__ import annotations

from functools import partial

import grain.python as grain

from ...components.loss import IGNORE_INDEX
from ..dataset import (
    DatasetConcat,
    DatasetMix,
    SingleDataset,
    as_iter_dataset,
    build_dataset,
)
from ..types import DatasetBuildContext, DatasetIterationPolicy
from .conversions import (
    _packing_output_is_full,
    _packing_output_to_text_sequence,
    _text_sequence_to_packing_input,
)
from .iterators import (
    _DocumentAwareConcatThenSplitIterDataset,
    _SplitTextSequenceDocuments,
)


def _row_lengths(context: DatasetBuildContext) -> dict[str, int]:
    """The per-feature row length every packing node fills to.

    All four token features are packed to the same length; the dict is what the
    Grain packing iterators take, and building it in one place is what keeps the
    two packers' idea of a row identical.
    """
    return {
        "input_ids": context.num_tokens_per_batch,
        "labels": context.num_tokens_per_batch,
        "positions": context.num_tokens_per_batch,
        "padding_mask": context.num_tokens_per_batch,
    }


def build_concat_then_split_packing(
    dataset: (SingleDataset | DatasetMix | DatasetConcat),
    *,
    context: DatasetBuildContext,
    dataset_iteration_policy: DatasetIterationPolicy,
) -> grain.IterDataset:
    """Concatenate documents, chunking them into fixed-length rows."""
    dataset_graph = build_dataset(
        dataset,
        context=context,
        dataset_iteration_policy=dataset_iteration_policy,
    )
    if context.max_num_documents is not None:
        dataset_graph = as_iter_dataset(dataset_graph, context=context)
        return _DocumentAwareConcatThenSplitIterDataset(
            dataset_graph,
            max_num_documents_per_row=context.max_num_documents,
            max_context_length=context.max_context_length,
            num_tokens_per_row=context.num_tokens_per_batch,
        )
    dataset_graph = dataset_graph.map(
        partial(
            _text_sequence_to_packing_input,
            max_context_length=context.max_context_length,
        )
    )
    dataset_graph = as_iter_dataset(dataset_graph, context=context)
    dataset_graph = grain.experimental.ConcatThenSplitIterDataset(
        dataset_graph, length_struct=_row_lengths(context)
    )
    dataset_graph = dataset_graph.filter(_packing_output_is_full)
    return dataset_graph.map(
        partial(
            _packing_output_to_text_sequence,
            max_context_length=context.max_context_length,
        )
    )


def build_first_fit_packing(
    dataset: (SingleDataset | DatasetMix | DatasetConcat),
    *,
    context: DatasetBuildContext,
    dataset_iteration_policy: DatasetIterationPolicy,
    num_packing_bins: int = 8,
) -> grain.IterDataset:
    """Pack document chunks no longer than the context window.

    ``num_packing_bins`` is how many candidate rows are kept open; more bins can
    reduce padding, but buffer more samples.
    """
    if num_packing_bins <= 0:
        raise ValueError("num_packing_bins must be positive")

    dataset_graph = build_dataset(
        dataset,
        context=context,
        dataset_iteration_policy=dataset_iteration_policy,
    )
    dataset_graph = as_iter_dataset(dataset_graph, context=context)
    dataset_graph = grain.experimental.FlatMapIterDataset(
        dataset_graph,
        _SplitTextSequenceDocuments(
            max_context_length=context.max_context_length,
        ),
    )
    dataset_graph = dataset_graph.map(
        partial(
            _text_sequence_to_packing_input,
            max_context_length=context.max_context_length,
        )
    )
    # TODO(data-global-pack-plan): Consider packing before DP sharding so
    # ranks receive similarly filled rows.
    dataset_graph = grain.experimental.FirstFitPackIterDataset(
        dataset_graph,
        length_struct=_row_lengths(context),
        padding_struct={
            "input_ids": 0,
            "labels": IGNORE_INDEX,
            "positions": 0,
            "padding_mask": True,
        },
        num_packing_bins=num_packing_bins,
        meta_features=("labels", "positions"),
        seed=dataset_iteration_policy.seed,
        shuffle_bins=dataset_iteration_policy.shuffle,
        max_sequences_per_bin=(
            context.max_num_documents if context.max_num_documents is not None else None
        ),
    )
    return dataset_graph.map(
        partial(
            _packing_output_to_text_sequence,
            max_context_length=context.max_context_length,
        )
    )

