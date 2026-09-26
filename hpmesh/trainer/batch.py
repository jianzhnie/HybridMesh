"""Batch handling for the trainer: data loading, accounting, preprocessing.

Split out of ``trainer.py`` (batch 4b): the dataloading-axis helpers, the
loader construction and fetch wrapper, the token counting, and the
device-move / preprocess seam. These are module-level functions whose first
parameter stays named ``self`` -- the bodies moved here verbatim and
``Trainer`` keeps same-named thin delegates, so call sites, tests and the
monkeypatch surface are unchanged. Everything they read (``cfg``,
``parallel_dims``, ``metrics``, ``ntokens_seen``, ...) stays on the trainer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from time import perf_counter
from typing import Any

import torch

from ..components.loss import IGNORE_INDEX, next_token_targets
from ..datasets import build_dataloader as build_dataset_dataloader
from ..datasets.loader import BaseDataLoader, DataloaderExhaustedError, TrainerBatch
from ..datasets.random_data import DataLoaderExhausted, RandomTokenDataLoader
from ..datasets.types import Batch


def dp_rank_world_size(self) -> tuple[int, int]:
    """This rank's position and extent along the dataloading (DP) axis."""
    # ``getattr``, not attribute access: a Trainer built with ``__new__``
    # (the tests' way of exercising the pure helpers) has no mesh, and the
    # only correct answer there is "one rank, no sharding".
    if getattr(self, "parallel_dims", None) is None:
        return 0, 1
    if self.parallel_dims.pp_enabled:
        # Under PP the dataloading view is the "batch" axis: it spans
        # dp_replicate * dp_shard and excludes pp, so every stage of one
        # pipeline reads the same shard of the global batch.
        batch_mesh = self.parallel_dims.get_optional_mesh(
            "batch", include_singleton_axes=True
        )
        return batch_mesh.get_local_rank(), batch_mesh.size()
    # The dense DP group spans replicate * shard; unsplit on torchrun it is
    # a plain 1-D mesh.
    dp_mesh = self.parallel_dims.get_optional_mesh(
        "dp", include_singleton_axes=True
    )
    return dp_mesh.get_local_rank(), dp_mesh.size()

def batch_size_per_rank(self, dp_world_size: int) -> int:
    """This rank's share of the global batch, checked rather than floored.

        Both loader paths divide the global batch by ``dp_world_size`` -- the
        random path slices rows, the Grain path is handed a token count -- and
        both are wrong in the same silent way when it does not divide: the run
        reads a smaller global batch than the config names, and every number
        derived from it (the lr, the token count, the value logged as
        ``batch_size``) describes a batch that is not the one being read.

        ``RandomTokenDataLoader`` rejects an indivisible ``batch_size`` of its
        own, but only once it is constructed and only on the random path. The
        token count is computed here, before either loader exists, so this is
        the one place both paths pass through -- which is what makes the
        failure the same for both, and the same up front.
        """
    global_batch_size = self.cfg.global_batch_size
    if global_batch_size % dp_world_size != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"the number of data-parallel ranks ({dp_world_size}); each rank "
            f"reads global_batch_size // dp_world_size samples, and the "
            f"remainder would be dropped silently."
        )
    return global_batch_size // dp_world_size

def build_dataloader(self) -> BaseDataLoader | None:
    """Build the micro-batch source the config names.

        Whatever the source, it rides along in the checkpoint's ``states``:
        resuming without its read position would resume the weights and restart
        the data, silently training a second pass over the beginning of the
        corpus. The synthetic loader's ``load_state_dict`` reaches the saved
        position by replaying generated batches, which is exact (batch k is a
        pure function of ``(seed, k)``) if not free.
        """
    dp_rank, dp_world_size = self.dp_rank_world_size()
    batch_size_per_rank = self.batch_size_per_rank(dp_world_size)
    loader = build_dataset_dataloader(
        self.cfg,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        # Per rank, not global: the Grain loader splits every dataset's
        # rows across ``dp_world_size`` ranks itself, so this many tokens
        # per rank is this many tokens per rank of the global batch.
        num_tokens_per_batch=batch_size_per_rank * self.cfg.max_seq_len,
    )
    return loader

def data_iterator(self) -> Iterator[Batch | TrainerBatch]:
    """The raw micro-batch source.

        A method rather than an attribute so tests can drive the loop with a
        fixed batch without touching the loop itself. The loader is a
        :class:`~hpmesh.datasets.loader.BaseDataLoader` whenever there is one,
        so both the synthetic and the Grain path arrive here the same way.
        """
    if self.dataloader is not None:
        return iter(self.dataloader)
    dp_rank, dp_world_size = self.dp_rank_world_size()
    return iter(
        RandomTokenDataLoader(
            seed=self.cfg.seed,
            vocab_size=self.cfg.vocab_size,
            batch_size=self.cfg.global_batch_size,
            seq_len=self.cfg.max_seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    )

def batch_generator(
    self, data_iterable: Iterable[Batch | TrainerBatch]
) -> Iterator[Batch | TrainerBatch]:
    """Wrap the source with the per-fetch accounting the step needs.

        Mirrors the reference's ``batch_generator``: the timing and token
        counters are updated as each batch is *taken*, not when it is used, so
        a step that raises partway through still accounts for the reads it
        performed. Yields the batch unchanged -- the loader's shape is the
        step's business, not this wrapper's.

        The token counter takes every label, not just the predictable ones.
        It reports throughput -- tokens the loader produced, which is what
        ``MFU`` wants -- and is not the loss denominator. That one is
        ``local_valid_tokens`` in ``train_step``, which is separately reduced
        across DP.

        Running out of data raises ``DataloaderExhaustedError`` rather than
        letting ``StopIteration`` escape: the loop must abandon the whole step
        rather than train on a partial batch, and ``StopIteration`` inside a
        generator would be read as "this generator is empty" and silently end
        training instead of failing the step.
        """
    data_iterator = iter(data_iterable)
    while True:
        data_load_start = perf_counter()
        try:
            batch = next(data_iterator)
        except (DataLoaderExhausted, StopIteration) as ex:
            # Two spellings of one event -- the synthetic source raises its
            # own type, a real loader just stops -- mapped to the one the
            # loop catches.
            raise DataloaderExhaustedError() from ex
        labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
        self.metrics.add_tokens(labels.numel())
        self.metrics.add_data_loading_time(perf_counter() - data_load_start)
        yield batch

def count_valid_tokens(batch: Batch | TrainerBatch) -> int:
    """The number of labels that contribute to the loss, pre-shard.

        The trainer's half of the token accounting, and it stays in the trainer
        for a reason: the count divides the loss *before* the first backward and
        is reduced across DP before that, so it cannot be produced per
        micro-batch by a model-side counter.

        It is taken from the batch as the loader handed it over -- before the
        model normalizes shapes, shifts for row ends, or shards for CP -- which
        is what makes it rank-independent: every CP rank of a DP group holds
        the same unsharded batch, so the dp-only reduction counts the whole batch
        once rather than once per sequence shard.

        A collator counts its own tokens while it already has the labels in
        hand; the synthetic source has no collator, so its count is derived here
        from the row shift. Both agree on what counts: a document's final
        position predicts nothing and is excluded.
        """
    if isinstance(batch, Batch):
        seq_len = batch.labels.shape[-1]
        targets = next_token_targets(batch.labels.reshape(-1), seq_len=seq_len)
        return int((targets != IGNORE_INDEX).sum())
    num_valid_tokens = batch.get("num_valid_tokens")
    if num_valid_tokens is None:
        # Recounted rather than permissively defaulted, so a dict that
        # silently lacks the key still produces a correct denominator.
        labels = batch["labels"]
        num_valid_tokens = int((labels != IGNORE_INDEX).sum())
    return num_valid_tokens


def microbatch(self, batch: Batch | TrainerBatch) -> dict[str, Any]:
    """Everything one accumulation group's forward/backward needs.

        The split of responsibility here mirrors torchtitan's ``train_step``,
        and each half is load-bearing:

        * **The count is popped by the trainer.** It is the loss denominator,
          which must be reduced across DP before the first backward, so it
          cannot come out of a per-micro-batch model call.
        * **The accounting is taken by the trainer**, from the loader's own
          labels, before any reshaping: throughput is a report about the loader
          ("tokens it produced"), not about the loss. The two numbers differ --
          a document's final position is loaded but never predicted -- and that
          is why they are not one field. The cumulative count is divided by
          ``cp * tp`` because every rank of a CP/TP group reads the same batch
          and the report sums it over the loss mesh (see ``train_step``).
        * **Everything else stays on the host until its group is consumed.**
          ``preprocess`` moves one group's tensors to the device just ahead of
          that group's forward (see ``to_device``), so holding the rest of the
          accumulation window costs host memory, not device memory -- the CPU
          invariant ``torchtitan`` documents for ``batch_generator``.
        """
    labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
    # Count this rank's share of the sequence, not the whole batch: every
    # rank of a CP/TP group reads the *same* batch from the loader, and the
    # sequence is cut across the group later (``shard_batch_for_cp`` /
    # ``shard_batch_for_tp``). Counting the full batch here and then summing
    # over the dp*cp*tp loss mesh in ``train_step`` would report a corpus
    # cp * tp times its true size; counting the share makes that sum
    # reconstruct the tokens actually read.
    parallel_dims = getattr(self, "parallel_dims", None)
    sequence_shards = (
        1 if parallel_dims is None else parallel_dims.cp * parallel_dims.tp
    )
    self.ntokens_seen += labels.numel() // sequence_shards
    num_valid_tokens = self.count_valid_tokens(batch)

    if isinstance(batch, dict):
        # ``num_valid_tokens`` is the model's to ignore, and a plain int
        # among tensors would be splatted into the forward as a kwarg.
        batch.pop("num_valid_tokens", None)
    return {"batch": batch, "num_valid_tokens": num_valid_tokens}

def to_device(self, batch: Batch | TrainerBatch) -> Batch | TrainerBatch:
    """Move one consumption group's tensors to the training device.

        Called by ``preprocess``, once per group just ahead of that group's
        forward -- not at read time. Reading the whole accumulation window onto
        the device up front would keep every micro-batch resident in device
        memory for the whole window, which is exactly what deferring avoids.
        """
    if isinstance(batch, dict):
        return {
            key: value.to(self.device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
    return Batch(
        input_ids=batch.input_ids.to(self.device, non_blocking=True),
        labels=batch.labels.to(self.device, non_blocking=True),
    )

def preprocess(
    self, microbatch: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Ask the model to turn its batch into forward inputs.

        A thin wrapper so the two bodies call the seam the same way and neither
        has to know which model object is canonical on its stage. The group's
        tensors are moved to the device here -- at consumption, not at read --
        so an accumulation window's unread groups stay on the host.
        """
    return self._example_model.preprocess_inputs(
        self.to_device(microbatch["batch"]),
        parallel_dims=self.parallel_dims,
        parallelism=self.cfg.parallel,
        max_context_length=self.cfg.max_seq_len,
    )
