"""Composable dataset recipes backed by Grain.

Vendored from torchtitan ``components/data/dataset.py``. Each node here is a
plain dataclass describing a recipe; the free function :func:`build_dataset`
turns one into a Grain graph. Construction lives outside the dataclass because
it takes arguments the description does not have -- which rank am I, how many
tokens per batch -- and because a class that carries its own builder suggests
the built graph is one of its fields, which it is not. That is the shape
``trainer/config.py`` states for every config in the package (descriptions, not
builders) and the one ``components/optimizer`` and ``components/profiler``
already use. A ``SampleProcessor`` gets its runtime values, the tokenizer above
all, as ``__init__(*, context)``.

The three nodes are named for what they are -- a single dataset, a mix, a
concatenation -- rather than suffixed ``Config``. A ``...Config`` name promises
a description of something built later; these are nodes, and the thing built
from them is a ``GrainDataset`` the caller already holds the type of.

The ordering in :func:`_build_map_dataset` is the part that must not move. It
runs pre-filters, then the processor, then post-filters, and only then shuffles,
shards, and repeats. Shuffling after filtering is what makes the DP slices
disjoint *and* uniform; sharding before filtering would leave ranks with
different numbers of surviving rows, and the loader's even-step assumption
(``repeat=True``) would silently stop holding.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, TypeAlias, cast

import grain.python as grain
import numpy as np

from .sources import (
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
    IndexedJsonlSource,
    RandomAccessDataSource,
    build_source,
)
from .types import DatasetBuildContext, DatasetIterationPolicy

__all__ = [
    "DatasetConcat",
    "DatasetMix",
    "GrainDataset",
    "SampleProcessor",
    "SingleDataset",
    "TextSequence",
    "WeightedDataset",
    "build_dataset",
]


GrainDataset: TypeAlias = grain.MapDataset | grain.IterDataset


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSequence:
    """Next-token-aligned text preserved through composition and packing.

    NOTE: It is the dataset's processor responsibility to shift tokens into
    aligned input and label pairs. The trainer does not do it.
    """

    input_ids: np.ndarray
    """Tokens provided to the model."""
    labels: np.ndarray
    """Target for each input token, with `IGNORE_INDEX` where loss is disabled."""
    positions: np.ndarray | None = None
    """Per-token positions; `None` until packing or collation materializes them."""
    padding_mask: np.ndarray | None = None
    """Per-token mask that is true for padding; `None` before padding."""

    def __post_init__(self) -> None:
        lengths = [len(self.input_ids), len(self.labels)]
        if self.positions is not None:
            lengths.append(len(self.positions))
        if self.padding_mask is not None:
            lengths.append(len(self.padding_mask))
        if len(set(lengths)) != 1:
            raise ValueError(
                f"TextSequence fields must have equal lengths, got {lengths}"
            )


class SampleProcessor(ABC):
    """Row processor using Grain-provided deterministic randomness.

    A subclass takes whatever it needs from ``context`` -- the tokenizer, the
    context length -- and encodes it in ``__init__``; ``__call__`` is the
    per-row transform. Both halves run in the data pipeline, so anything read
    here must already be set up by the time :func:`build_dataset` runs.
    """

    def __init__(self, *, context: DatasetBuildContext) -> None:
        del context

    @abstractmethod
    def __call__(
        self,
        sample: Any,
        rng: np.random.Generator,
    ) -> Any: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class SingleDataset:
    """One dataset built from a source and row-level transforms.

    `pre_filters` run before `processor`; `post_filters` run afterward.
    This node owns the leaf shuffle, repeat, and effective-DP sharding.
    """

    source: (
        IndexedJsonlSource
        | HuggingFaceRandomAccessSource
        | HuggingFaceStreamingSource
        | RandomAccessDataSource
        | grain.IterDataset
    )
    pre_filters: tuple[Callable[[Any], bool], ...] = ()
    processor: type[SampleProcessor] | None = None
    """Processor class, constructed per build as ``processor(context=context)``.

    A class rather than a bound :func:`functools.partial`: a partial's fixed
    keyword arguments are *overridden* by an explicit keyword at the call site,
    so ``partial(Proc, context=other)`` here would be silently replaced by the
    build context rather than raising.
    """
    post_filters: tuple[Callable[[Any], bool], ...] = ()


def build_dataset(
    node: SingleDataset | DatasetMix | DatasetConcat,
    *,
    context: DatasetBuildContext,
    dataset_iteration_policy: DatasetIterationPolicy,
) -> GrainDataset:
    """Build one node of a Grain dataset graph from its description.

    Dispatches on the concrete class rather than a method, so a new recipe has
    to be added here -- and the union below stops type-checking until it is.
    """
    if isinstance(node, SingleDataset):
        return _build_single(
            node,
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
    if isinstance(node, DatasetMix):
        return _build_mix(
            node,
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
    if isinstance(node, DatasetConcat):
        return _build_concat(
            node,
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
    raise TypeError(f"unhandled dataset type {type(node).__qualname__}")


def _build_single(
    node: SingleDataset,
    *,
    context: DatasetBuildContext,
    dataset_iteration_policy: DatasetIterationPolicy,
) -> GrainDataset:
    source = build_source(
        node.source,
        dataset_iteration_policy=dataset_iteration_policy,
    )
    if isinstance(source, RandomAccessDataSource):
        dataset: GrainDataset = grain.MapDataset.source(source)
    elif isinstance(source, grain.IterDataset):
        dataset = source
    else:
        raise TypeError("source must be a RandomAccessDataSource or grain.IterDataset")

    if isinstance(dataset, grain.MapDataset):
        return _build_map_dataset(
            node,
            dataset,
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
    return _build_iter_dataset(
        node,
        dataset,
        context=context,
        dataset_iteration_policy=dataset_iteration_policy,
    )


def _build_map_dataset(
    node: SingleDataset,
    dataset: grain.MapDataset,
    *,
    context: DatasetBuildContext,
    dataset_iteration_policy: DatasetIterationPolicy,
) -> grain.MapDataset:
    """Process globally indexed rows before shuffle, sharding, and repeat."""
    # Filter raw rows.
    for filter_fn in node.pre_filters:
        dataset = dataset.filter(filter_fn)

    # Process rows into training samples.
    if node.processor is not None:
        dataset = dataset.random_map(
            node.processor(context=context),
            seed=dataset_iteration_policy.seed,
        )

    # Filter processed samples.
    for filter_fn in node.post_filters:
        dataset = dataset.filter(filter_fn)

    # Shuffle globally, then give each DP rank a disjoint slice.
    if dataset_iteration_policy.shuffle:
        dataset = dataset.shuffle(seed=dataset_iteration_policy.seed)
    dataset = _shard_for_dp(dataset, dataset_iteration_policy)
    if dataset_iteration_policy.repeat:
        # Grain preserves the epoch through sliced map indices, so the
        # upstream shuffle uses seed + epoch on each repeat.
        dataset = dataset.repeat()
    return dataset


def _build_iter_dataset(
    node: SingleDataset,
    dataset: grain.IterDataset,
    *,
    context: DatasetBuildContext,
    dataset_iteration_policy: DatasetIterationPolicy,
) -> grain.IterDataset:
    """Shuffle streaming rows before row processing."""
    # Shuffle raw stream rows.
    if dataset_iteration_policy.shuffle:
        dataset = grain.experimental.WindowShuffleIterDataset(
            dataset,
            window_size=dataset_iteration_policy.streaming_shuffle_buffer_size,
            seed=dataset_iteration_policy.seed,
        )

    # Filter and process rows in stream order.
    for filter_fn in node.pre_filters:
        dataset = dataset.filter(filter_fn)
    if node.processor is not None:
        dataset = dataset.random_map(
            node.processor(context=context),
            # A stream has no global index to shuffle, so the seed is what
            # separates ranks: without the dp_rank offset every rank would
            # draw the same samples from the same window.
            seed=dataset_iteration_policy.seed + dataset_iteration_policy.dp_rank,
        )
    for filter_fn in node.post_filters:
        dataset = dataset.filter(filter_fn)
    return dataset


def _shard_for_dp(
    dataset: grain.MapDataset,
    dataset_iteration_policy: DatasetIterationPolicy,
) -> grain.MapDataset:
    """Give ``dp_rank`` a contiguous, disjoint slice of ``dataset``.

    Contiguous rather than strided because Grain's shuffle is a permutation
    computed from the index: a contiguous slice of the shuffled index space is
    still an even share of the whole, and it keeps each rank's reads sequential.
    """
    dp_world_size = dataset_iteration_policy.dp_world_size
    dp_rank = dataset_iteration_policy.dp_rank
    if len(dataset) < dp_world_size:
        raise ValueError(
            f"dataset has {len(dataset)} rows, fewer than dp_world_size={dp_world_size}"
        )
    shard_size, remainder = divmod(len(dataset), dp_world_size)
    shard_start = dp_rank * shard_size + min(dp_rank, remainder)
    shard_stop = shard_start + shard_size
    if dp_rank < remainder:
        shard_stop += 1
    return dataset[shard_start:shard_stop]


@dataclass(frozen=True, kw_only=True, slots=True)
class WeightedDataset:
    """A dataset and its relative selection weight."""

    dataset: SingleDataset | DatasetMix | DatasetConcat
    weight: float = 1.0


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetMix:
    """Interleave weighted children.

    `MapDataset.filter` leaves rejected indices as `None`, so the all-map path
    weights attempted indices rather than accepted samples. Otherwise, weights
    select elements emitted by each iterable child. The child defines the
    element: mixing `TextSequence` children weights documents; mixing packed
    fixed-length children weights physical tokens.

    With `repeat=True` the mix is infinite. With `repeat=False` the mix stops
    at the first exhausted child, so larger children are not fully covered.
    """

    datasets: tuple[WeightedDataset, ...]


def _build_mix(
    node: DatasetMix,
    *,
    context: DatasetBuildContext,
    dataset_iteration_policy: DatasetIterationPolicy,
) -> GrainDataset:
    if not node.datasets or any(
        not math.isfinite(item.weight) or item.weight <= 0 for item in node.datasets
    ):
        raise ValueError("DatasetMix requires finite, positive-weight datasets")
    children = [
        build_dataset(
            item.dataset,
            context=context,
            # Inserting or reordering a child reseeds every later child.
            dataset_iteration_policy=replace(
                dataset_iteration_policy,
                seed=dataset_iteration_policy.seed + index,
            ),
        )
        for index, item in enumerate(node.datasets)
    ]
    weights = [item.weight for item in node.datasets]
    # TODO(data-token-weighted-mix): Support source weights by token count,
    # not only emitted examples or packed rows. Checkpoint running estimates.
    if all(isinstance(child, grain.MapDataset) for child in children):
        return grain.MapDataset.mix(
            cast(list[grain.MapDataset], children),
            weights=weights,
        )
    children = [
        child.to_iter_dataset(read_options=context.read_options)
        if isinstance(child, grain.MapDataset)
        else child
        for child in children
    ]
    return grain.IterDataset.mix(children, weights=weights)


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetConcat:
    """Concatenates finite children before global shuffle and DP sharding."""

    datasets: tuple[SingleDataset | DatasetMix | DatasetConcat, ...]


def _build_concat(
    node: DatasetConcat,
    *,
    context: DatasetBuildContext,
    dataset_iteration_policy: DatasetIterationPolicy,
) -> grain.MapDataset:
    child_iteration_policy = replace(
        dataset_iteration_policy,
        shuffle=False,
        repeat=False,
        dp_rank=0,
        dp_world_size=1,
    )
    children = [
        build_dataset(
            dataset,
            context=context,
            dataset_iteration_policy=child_iteration_policy,
        )
        for dataset in node.datasets
    ]

    if not children or not all(
        isinstance(child, grain.MapDataset) for child in children
    ):
        raise TypeError("DatasetConcat requires map-style children")

    dataset = grain.MapDataset.concatenate(cast(list[grain.MapDataset], children))

    if dataset_iteration_policy.shuffle:
        dataset = dataset.shuffle(seed=dataset_iteration_policy.seed)

    dataset = _shard_for_dp(dataset, dataset_iteration_policy)

    if dataset_iteration_policy.repeat:
        dataset = dataset.repeat()
    return dataset
