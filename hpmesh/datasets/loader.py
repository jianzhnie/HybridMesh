"""Grain-backed dataloader.

Vendored from torchtitan ``components/data/loader.py``. Everything the loader
needs is a constructor argument: the already-built graph, the collator, the
iteration knobs, and the runtime objects -- the tokenizer, the DP extent -- that
no description can hold. There is no config dataclass in between, because a
config here would be a parameter bag built one line before its only consumer.

``GrainDataLoader``'s ``dataset`` is the already-built Grain graph, not something
this module constructs. The caller has the dataset registry, and building it
there is what keeps ``loader.py`` free of a dependency on every concrete dataset
-- which is also why :func:`~hpmesh.datasets.build.build_dataloader`, which does
own that registry, lives one level up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import grain.python as grain
from grain import experimental as grain_experimental
from torch.distributed.checkpoint.stateful import Stateful

from ..components.tokenizer import BaseTokenizer
from .collators import Collator, TextCollator, TrainerBatch
from .types import DatasetBuildContext

__all__ = [
    "BaseDataLoader",
    "DataloaderExhaustedError",
    "GrainDataLoader",
]


# NOTE: This class deliberately inherits from `Exception` and not `StopIteration`.
# According to PEP 479, raising a `StopIteration` or its subclass from within a
# generator will wrap it in a `RuntimeError`. Since this exception is designed
# to be raised from a generator-based dataloader and caught by the training loop,
# inheriting from `StopIteration` would make it uncatchable and would crash the
# program.
# See: https://peps.python.org/pep-0479/
class DataloaderExhaustedError(Exception):
    """An exception that indicates dataloader exhaustion."""

    pass


class BaseDataLoader(Stateful, ABC):
    """Enforces the `Stateful`, `state_dict()`, and `load_state_dict()` contract."""

    max_num_documents: int | None = None

    @abstractmethod
    def __iter__(self) -> Iterator[TrainerBatch]: ...

    def close(self) -> None:
        pass


class GrainDataLoader(BaseDataLoader):
    """Batches and checkpoints one composed Grain dataset graph."""

    def __init__(
        self,
        dataset: grain.MapDataset | grain.IterDataset,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        max_context_length: int,
        num_tokens_per_batch: int,
        collator: type[Collator] = TextCollator,
        repeat: bool = True,
        read_options: grain.ReadOptions | None = None,
        num_prefetch_batches: int = 2,
        max_num_documents: int | None = None,
    ) -> None:
        """``dataset`` is the built graph: which node it starts from decides
        shuffle order, so it is built by the caller, not here.

        Without a config there is no field list to declare, so this is the flat
        version of one: everything the loader reads is a named argument and
        nothing else is carried. The seed and the shuffle flags are *not* here,
        because the graph has already consumed them by the time it arrives --
        keeping a copy would be a knob that changes no behavior.

        ``read_options`` defaults to a fresh ``grain.ReadOptions`` rather than
        being shared as a mutable default. ``max_num_documents`` is the maximum
        non-padding document segments in one local token batch, so the same
        argument reaches both the build context and ``collator``.
        """
        if max_num_documents is not None and max_num_documents <= 0:
            raise ValueError("max_num_documents must be positive")
        if read_options is None:
            read_options = grain.ReadOptions()
        # The graph is built before this loader exists and may already have been
        # built for a different rank -- the trainer derives the policy from a
        # config that is not handed here. Catch the mismatch rather than train
        # on a slice that silently disagrees with the rest of the mesh.
        expected_rank_id = f"dp_rank_{dp_rank}"
        self._dp_world_size = dp_world_size
        self._rank_id = expected_rank_id
        self.max_num_documents = max_num_documents

        # A finite dataset cannot be shared by several ranks: each one reaches
        # the end at a different step, and the ranks that ran out first stop
        # entering the next collective while the others block in it. Checked
        # here rather than at the first short batch, which is the point -- by
        # the time a rank notices, its peers are already waiting.
        # TODO(data-finite-dp): Support finite distributed datasets with a global
        # remainder policy. Simple map datasets can truncate or pad before DP
        # sharding; filtered, mixed, packed, and streaming datasets need
        # coordinated exhaustion so every rank runs the same number of steps.
        if dp_world_size > 1 and not repeat:
            raise ValueError(
                "repeat=False with data parallelism can exhaust ranks at different "
                "steps and hang collectives; use repeat=True with a trainer-"
                "controlled step count"
            )
        context = DatasetBuildContext(
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            num_tokens_per_batch=num_tokens_per_batch,
            read_options=read_options,
            max_num_documents=max_num_documents,
        )

        collator = collator(context=context)

        # TODO(data-multiprocessing): CPU-heavy processing should use multiple
        # processes rather than only threads. Grain can divide map-style data among
        # workers, but packing and mixing map data with a stream produce an iterable
        # before the loader sees it. Investigate an earlier boundary where one
        # shared worker pool processes samples, instead of creating a pool per
        # dataset or packing per worker.
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=read_options)

        # Batch and collate samples.
        dataset = dataset.batch(
            collator.num_rows_per_batch(),
            drop_remainder=repeat,
            batch_fn=collator,
        )

        # Queue completed batches while the trainer consumes the previous batch.
        dataset = grain_experimental.ThreadPrefetchIterDataset(
            dataset, prefetch_buffer_size=num_prefetch_batches
        )
        self._iterator = iter(dataset)

    def __iter__(self) -> Iterator[TrainerBatch]:
        return self._iterator

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dp_world_size": self._dp_world_size,
            self._rank_id: self._iterator.get_state(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict["version"] != 1:
            raise ValueError(
                f"unsupported GrainDataLoader state version {state_dict['version']}"
            )
        if state_dict["dp_world_size"] != self._dp_world_size:
            raise ValueError(
                "cannot resume after changing the effective data-parallel degree"
            )
        if self._rank_id not in state_dict:
            raise ValueError(
                f"checkpoint is missing dataloader state for {self._rank_id}"
            )
        try:
            self._iterator.set_state(state_dict[self._rank_id])
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._iterator.close()
