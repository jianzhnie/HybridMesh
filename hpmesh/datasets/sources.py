# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Storage adapters for Grain datasets.

Vendored from torchtitan ``components/data/sources.py``. Each source is its own
config: the class carries its own fields and implements ``build()``, which is
the shape ``SourceConfig`` describes -- a ``build()`` that returns either a
random-access source or a ``grain.IterDataset``.

Note the asymmetry, which is deliberate and load-bearing: the two Hugging Face
sources take ``dataset_iteration_policy`` and ignore it, while the streaming one
shards by it. ``split_dataset_by_node`` is what gives each DP rank a disjoint
stream; the random-access and JSONL sources leave sharding to
``SingleDatasetConfig._build_map_dataset``, which slices after the shuffle.
Sharding twice would drop rows.
"""

from __future__ import annotations

import glob
import json
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import datasets
import grain.python as grain
from datasets.distributed import split_dataset_by_node

from .types import DatasetIterationPolicy

__all__ = [
    "HuggingFaceRandomAccessSource",
    "HuggingFaceStreamingSource",
    "IndexedJsonlSource",
    "RandomAccessDataSource",
    "SourceConfig",
]


@runtime_checkable
class RandomAccessDataSource(Protocol):
    """Finite data source addressable by integer index."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


class SourceConfig(Protocol):
    """Builds a random-access or streaming source."""

    def build(
        self,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> RandomAccessDataSource | grain.IterDataset: ...


@dataclass(kw_only=True)
class IndexedJsonlSource:
    """Provides random access to JSONL rows through compact byte offsets.

    Accepts either a single pattern or a list of them; several patterns resolve
    to one index over the union of the matched files.
    """

    patterns: tuple[str, ...]
    """File globs to index. Every pattern must match at least one file."""

    def __post_init__(self) -> None:
        if isinstance(self.patterns, str):
            self.patterns = (self.patterns,)
        else:
            self.patterns = tuple(self.patterns)

    def build(
        self,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> RandomAccessDataSource:
        del dataset_iteration_policy
        return _IndexedJsonlDataSource(self.patterns)


class _IndexedJsonlDataSource:
    """The offset index ``IndexedJsonlSource`` builds.

    Separate from the config so the index is built once per ``build()`` and not
    carried around in a dataclass that is compared and hashed as configuration.
    """

    def __init__(self, patterns: tuple[str, ...]) -> None:
        self._paths = _file_patterns_to_paths(patterns)
        self._path_ids = array("I")
        self._byte_offsets = array("Q")
        # TODO(data-jsonl-sidecar): Startup rescans every JSONL file per rank and
        # worker. Build one validated offset index that all processes can
        # memory-map.
        for path_id, path in enumerate(self._paths):
            with open(path, "rb") as file:
                while True:
                    offset = file.tell()
                    line = file.readline()
                    if not line:
                        break
                    if line.strip():
                        self._path_ids.append(path_id)
                        self._byte_offsets.append(offset)

    def __len__(self) -> int:
        return len(self._byte_offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        path = self._paths[self._path_ids[index]]
        with open(path, "rb") as file:
            file.seek(self._byte_offsets[index])
            return json.loads(file.readline())


@dataclass(kw_only=True)
class HuggingFaceRandomAccessSource:
    """Provides random access to a materialized Hugging Face dataset."""

    path: str
    split: str
    name: str | None = None
    revision: str | None = None
    load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        duplicated = {"split", "name", "revision", "streaming"} & (
            self.load_dataset_kwargs.keys()
        )
        if duplicated:
            raise ValueError(
                "first-class Hugging Face fields repeated in kwargs: "
                f"{sorted(duplicated)}"
            )

    def build(
        self,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> RandomAccessDataSource:
        del dataset_iteration_policy
        dataset = datasets.load_dataset(
            self.path,
            name=self.name,
            split=self.split,
            revision=self.revision,
            streaming=False,
            **self.load_dataset_kwargs,
        )
        if not isinstance(dataset, datasets.Dataset):
            raise TypeError(
                "random-access Hugging Face source requires one Dataset; "
                f"got {type(dataset).__qualname__}"
            )
        return _HuggingFaceRandomAccessDataSource(dataset)


class _HuggingFaceRandomAccessDataSource:
    """The materialized dataset ``HuggingFaceRandomAccessSource`` builds."""

    def __init__(self, dataset: datasets.Dataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._dataset[index]


@dataclass(kw_only=True)
class HuggingFaceStreamingSource:
    """Provides a DP-sharded Hugging Face stream with cursor checkpointing.

    The dataset is *not* loaded here. A dataset catalog holds these at module
    scope, so loading in ``__init__`` would make importing a catalog reach the
    Hub; ``build()`` is what resolves and splits the stream.
    """

    path: str
    split: str
    name: str | None = None
    revision: str | None = None
    load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        duplicated = {"split", "name", "revision", "streaming"} & (
            self.load_dataset_kwargs.keys()
        )
        if duplicated:
            raise ValueError(
                "first-class Hugging Face fields repeated in kwargs: "
                f"{sorted(duplicated)}"
            )

    def build(
        self,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.IterDataset:
        dataset = datasets.load_dataset(
            self.path,
            name=self.name,
            split=self.split,
            revision=self.revision,
            streaming=True,
            **self.load_dataset_kwargs,
        )
        if not isinstance(dataset, datasets.IterableDataset):
            raise TypeError(
                "streaming Hugging Face source requires one IterableDataset; "
                f"got {type(dataset).__qualname__}"
            )
        if not hasattr(dataset, "state_dict") or not hasattr(
            dataset, "load_state_dict"
        ):
            raise TypeError(
                "Hugging Face streaming source does not support exact resume"
            )
        return _HuggingFaceStreamingIterDataset(
            # Split before the shuffle window, not after: every rank must draw
            # from its own stream, or two ranks would shuffle the same rows
            # into different orders and train on the same data twice.
            split_dataset_by_node(
                dataset,
                rank=dataset_iteration_policy.dp_rank,
                world_size=dataset_iteration_policy.dp_world_size,
            ),
            repeat=dataset_iteration_policy.repeat,
            shuffle=dataset_iteration_policy.shuffle,
        )


class _HuggingFaceStreamingIterDataset(grain.IterDataset):
    """The built node ``HuggingFaceStreamingSource`` produces.

    Separate from the source so the source can stay an inert dataclass. This
    node has only one parent-adjacent job -- hand out a cursor iterator -- and
    the cursor is what carries the checkpointable read position.
    """

    def __init__(
        self,
        dataset: datasets.IterableDataset,
        *,
        repeat: bool,
        shuffle: bool,
    ) -> None:
        self._dataset = dataset
        self._repeat = repeat
        self._shuffle = shuffle
        super().__init__()

    def __iter__(self) -> grain.DatasetIterator:
        return _HuggingFaceCursorIterator(
            self._dataset,
            repeat=self._repeat,
            shuffle=self._shuffle,
        )


def _file_patterns_to_paths(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """Return sorted, unique absolute paths matched by file patterns.

    Every pattern must match at least one file. Duplicate resolved paths are
    rejected so one file cannot be indexed twice.
    """
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"pattern matched no files: {pattern!r}")
        paths.extend(str(Path(match).resolve()) for match in matches)
    if len(paths) != len(set(paths)):
        raise ValueError("patterns resolve to the same file more than once")
    return tuple(paths)


class _HuggingFaceCursorIterator(grain.DatasetIterator):
    """Exposes a Hugging Face streaming cursor to Grain checkpoint recursion."""

    def __init__(
        self,
        dataset: datasets.IterableDataset,
        *,
        repeat: bool,
        shuffle: bool,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._repeat = repeat
        self._shuffle = shuffle
        self._epoch = 0
        self._initial_state = dataset.state_dict()
        self._iterator = iter(dataset)

    def __next__(self) -> dict[str, Any]:
        try:
            return next(self._iterator)
        except StopIteration:
            if not self._repeat:
                raise
            self._epoch += 1
            if self._shuffle:
                self._dataset.set_epoch(self._epoch)
            self._dataset.load_state_dict(self._initial_state)
            self._iterator = iter(self._dataset)
            return next(self._iterator)

    def get_state(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "hf": self._dataset.state_dict(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._epoch = state["epoch"]
        if self._shuffle:
            self._dataset.set_epoch(self._epoch)
        self._dataset.load_state_dict(state["hf"])
        self._iterator = iter(self._dataset)
