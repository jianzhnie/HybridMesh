"""The data iterator the training loop pulls micro-batches from.

Vendored in shape from torchtitan ``trainer.py:batch_generator``, and from the
``components/data`` contract it satisfies. What is kept is the *contract*, not
the data source:

* :class:`DataLoaderExhausted` is raised, not swallowed. Running out of data
  mid-step cancels the whole step rather than training on a partial batch, so
  the training loop can catch it and stop cleanly.
* The iterator is infinite from the loop's point of view: a finite source is
  restarted, an empty one raises immediately rather than spinning forever.

What is not kept: hpmesh has no dataloader component, no collator and no
``max_num_documents``. The source shipped here is synthetic random tokens.
Swapping in a real corpus means writing another ``Iterable[Batch]`` and passing
it to :func:`batch_iterator` -- nothing in the loop changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

__all__ = ["Batch", "DataLoaderExhausted", "RandomTokenSource", "batch_iterator"]


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
