"""The data iterator the training loop pulls micro-batches from.

Vendored in shape from torchtitan ``trainer.py:batch_generator``, and from the
``components/data`` contract it satisfies. What is kept is the *contract*, not
the data source:

* :class:`DataLoaderExhausted` is raised, not swallowed. Running out of data
  mid-step cancels the whole step rather than training on a partial batch, so
  the training loop can catch it and stop cleanly.
* The iterator is infinite from the loop's point of view: a finite source is
  restarted, an empty one raises immediately rather than spinning forever.

What is not kept: hpmesh has no collator or ``max_num_documents`` here. The
source shipped in this module is synthetic random tokens; the real corpus lives
behind the Grain dataset graph in ``loader.py``. Both satisfy
:class:`~hpmesh.datasets.loader.BaseDataLoader`, so the training loop has one
path rather than two.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from .loader import BaseDataLoader

__all__ = [
    "Batch",
    "DataLoaderExhausted",
    "RandomTokenDataLoader",
    "RandomTokenSource",
    "batch_iterator",
]



@dataclass
class Batch:
    """One micro-batch, on CPU: ``input_ids`` and ``labels`` of shape ``(B, T)``."""

    input_ids: torch.Tensor
    labels: torch.Tensor


class DataLoaderExhausted(Exception):
    """The source ran out of data part-way through an optimizer step.

    Raised by :func:`batch_iterator` so the trainer can abandon the step. Named
    after torchtitan's ``DataloaderExhaustedError``, minus the redundant suffix.
    """


def batch_iterator(source: Iterable[Batch]) -> Iterator[Batch]:
    """Yield batches forever, restarting a finite source.

    Every batch is yielded on CPU. The caller moves it to the device: keeping the
    move outside the iterator is what lets the trainer overlap it with gradient
    accumulation instead of blocking the fetch on a host-to-device copy.
    """
    while True:
        exhausted = True
        for batch in source:
            exhausted = False
            yield batch
        if exhausted:
            # Without this, an empty source would spin here forever.
            raise DataLoaderExhausted(
                "The data source yielded nothing; there is no batch to train on."
            )


class RandomTokenSource:
    """A synthetic random-token corpus, standing in for a real dataset.

    Deterministic by construction: batch ``step`` depends only on ``(seed, step)``,
    on every rank. That is what makes two runs comparable and, under data
    parallelism, makes each rank's slice a slice of the *same* global batch.
    """

    def __init__(
        self, *, seed: int, vocab_size: int, batch_size: int, seq_len: int
    ) -> None:
        self.seed = seed
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[Batch]:
        step = 0
        while True:
            # A fresh generator per step, so the sequence is reproducible from
            # (seed, step) alone with no dependence on how many steps ran before.
            generator = torch.Generator(device="cpu").manual_seed(
                self.seed * 100_000 + step
            )
            input_ids = torch.randint(
                0,
                self.vocab_size,
                (self.batch_size, self.seq_len),
                generator=generator,
            )
            # Next-token prediction: labels are the input shifted by one at loss
            # time, so the same tensor serves as both.
            yield Batch(input_ids=input_ids, labels=input_ids.clone())
            step += 1


class RandomTokenDataLoader(BaseDataLoader):
    """The synthetic corpus behind the ``BaseDataLoader`` contract.

    Each rank takes a contiguous slice of every global batch, which is the
    static form of what ``GrainDataLoader`` gets from ``_shard_for_dp`` in the
    dataset graph. Slicing here rather than in the trainer keeps the two
    loaders interchangeable: by the time a batch leaves either one it is
    already this rank's shard, and the loop does not need to know which.

    ``vocab_size``, ``batch_size`` and ``seq_len`` are the shapes the loop
    would otherwise have to know to build the source itself. They are not
    derived from ``num_tokens_per_batch`` the way the Grain path's
    ``num_tokens_per_batch`` is, because the two loaders disagree on what a
    batch means: this one yields ``(B, T)`` rows of one document each, the
    Grain one a flat packed token stream.
    """

    def __init__(
        self,
        *,
        seed: int,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        if batch_size % dp_world_size != 0:
            raise ValueError(
                f"batch_size={batch_size} not divisible by "
                f"dp_world_size={dp_world_size}"
            )
        self._dp_world_size = dp_world_size
        self._dp_rank = dp_rank
        self._rows_per_rank = batch_size // dp_world_size
        self._source = RandomTokenSource(
            seed=seed, vocab_size=vocab_size, batch_size=batch_size, seq_len=seq_len
        )
        self._iterator = self._slice_each(batch_iterator(self._source))
        # The number of batches handed out so far. The source is a pure
        # function of ``(seed, step)`` and cannot restart from an arbitrary
        # cursor -- Grain derives such a cursor from the index, but a plain
        # generator does not -- so resuming replays this many batches forward.
        self._num_batches_yielded = 0

    def __iter__(self) -> Iterator[Batch]:
        for batch in self._iterator:
            self._num_batches_yielded += 1
            yield batch

    def state_dict(self) -> dict[str, Any]:
        # See ``_num_batches_yielded``: the position is what a resume needs.
        return {
            "dp_world_size": self._dp_world_size,
            "steps": self._num_batches_yielded,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        # Same reasoning as ``GrainDataLoader``: a checkpoint written under a
        # different DP degree describes a different global batch, so resuming
        # would silently train on a different sample split.
        if state_dict.get("dp_world_size") != self._dp_world_size:
            raise ValueError(
                "cannot resume after changing the effective data-parallel degree"
            )
        resume_at = state_dict.get("steps", 0)
        if resume_at < self._num_batches_yielded:
            raise ValueError(
                f"cannot resume at batch {resume_at}: this loader has already "
                f"yielded {self._num_batches_yielded}. A fresh loader is required."
            )
        # This loader cannot seek, so a replay from the start is the only way
        # to reach ``resume_at``: a checkpoint must be loaded into a FRESH
        # loader, and the seek is followed by re-counting from zero.
        self._num_batches_yielded = 0
        for _ in range(resume_at):
            next(self._iterator)

    def _slice_each(self, batches: Iterator[Batch]) -> Iterator[Batch]:
        start = self._dp_rank * self._rows_per_rank
        for batch in batches:
            yield Batch(
                input_ids=batch.input_ids[start : start + self._rows_per_rank],
                labels=batch.labels[start : start + self._rows_per_rank],
            )

