# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DCP checkpoint manager: sharded I/O, async saving, HF export.

Vendored from torchtitan's ``components/checkpointer/dcp.py``. ``torch.distributed
.checkpoint`` is what this buys over ``torch.save``, and it buys two distinct
things:

* **A sharded layout.** Under FSDP each rank holds one shard of every parameter
  and only needs to write that shard; DCP records the global layout and stitches
  the shards back together on load. hpmesh's old ``torch.save``-per-rank format
  could not express that -- it happened to work only because every run it was
  used for had one rank owning the whole tensor.
* **Async saving and lazy loading.** ``async_save`` stages to host memory and
  writes from a background thread, so the next training step does not wait on
  disk. ``load`` reads into the live state dict in place, so a resumed run does
  not materialize a second copy of the model.

Two departures from torchtitan, both subtractions:

* **No ``structured_logger``.** torchtitan wraps saves in ``sl.log_trace_span``
  and emits a latency metric from a ``save_future`` callback. Those go; the
  ``logger.info`` timing lines that say the same thing stay.

* **No ``torchtitan.tools.filesystem``.** The local equivalent is
  ``hpmesh.utils.filesystem``, which has the same API for the paths this module
  uses. The one difference is deliberate: hpmesh's ``rmtree`` mirrors
  ``shutil.rmtree(..., ignore_errors=True)`` for local paths, so a purge of an
  already-deleted directory is a no-op rather than a ``FileNotFoundError`` in
  the purge thread.

``AsyncMode.DISABLED`` is the default here, matching torchtitan.
"""

from __future__ import annotations

import enum
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)

from ...utils import filesystem
from ...utils.gc import GarbageCollection
from .base import (
    DATALOADER,
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    BaseCheckpointManager,
    BaseCheckpointManagerConfig,
    ModelWrapper,
    OptimizerWrapper,
    purge_thread,
)

logger = logging.getLogger(__name__)


# Mirrors torchtitan's config-level dtype map, narrowed to the three values the
# checkpoint config already restricts ``export_dtype`` to.
EXPORT_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class _FilesystemCheckpointStorage:
    """``CheckpointStorage`` backed by ``hpmesh.utils.filesystem``.

    Local paths go through ``os``/``shutil`` and remote fsspec URIs through
    fsspec, so DCP keeps reading and writing remote checkpoint folders exactly as
    it does when the paths are local. The storage does not itself reject a remote
    URI it cannot address -- ``Config.__post_init__`` already refuses the
    combinations that would need to.
    """

    def isdir(self, path: str) -> bool:
        return filesystem.isdir(path)

    def isfile(self, path: str) -> bool:
        return filesystem.isfile(path)

    def listdir(self, path: str) -> list[str]:
        return filesystem.listdir(path)

    def remove(self, path: str) -> None:
        filesystem.rmtree(path)


class CheckpointManager(BaseCheckpointManager):
    """Checkpoint manager backed by ``torch.distributed.checkpoint``.

    Note on Pipeline Parallelism and Virtual Stages:

    1. Even for simple PP schedules there is a separate optimizer per PP rank.
       Rank 0's optimizer has a ``param_group[0]`` referring to ``layers.0`` of
       the original model; rank 1's *also* has a ``param_group[0]``, since the
       index is positional, but referring to ``layers.1``. When saving, these
       collide and one of them is lost. Then on reload only one stage can restore
       its optimizer state and the others error.

       The solution is optimizer flattening, which ``components/optimizer.py``
       does: optimizer state dicts are keyed by FQN and flattened into one dict.

    2. With complex PP schedules there are multiple model chunks per PP rank,
       which compounds (1) by also requiring us to reason about multiple local
       ``optim`` objects. ``ModelWrapper`` flattens the state dicts from each
       chunk into one before saving or loading, relying on the individual
       state dicts not to collide -- guaranteed for the model by correct
       pipeline splitting, and for the optimizer by the flattening in (1).

    3. LR schedulers index model state like optimizers do, so they are flattened
       the same way, under the assumption that all of them share a state dict.

    Args:
        config: how checkpointing is configured for this run.
        model_parts: the model chunks to checkpoint (one entry per PP stage).
        optimizer: the optimizer to checkpoint, wrapped as an
            ``OptimizerWrapper`` so a resumed run's fresh optimizer has its
            state materialized before DCP plans the load.
        states: extra states to save beyond the model and optimizer.
        folder: absolute directory the checkpoints live in. Already joined with
            the run's dump folder by the caller.
        sd_adapter: converts model state dicts between the native layout and
            another format (HF safetensors). Required for the HF export paths.
            hpmesh ships none yet, so those paths reject at construction.
    """

    @dataclass(kw_only=True)
    class Config(BaseCheckpointManagerConfig):
        async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
        """DCP save mode: synchronous, threaded async, or pinned-memory async."""

        def __post_init__(self) -> None:
            super().__post_init__()
            async_lowered = self.async_mode.lower()
            if async_lowered not in (
                "disabled",
                "async",
                "async_with_pinned_mem",
            ):
                raise ValueError(f"Invalid async_mode: {async_lowered}")
            self.async_mode = async_lowered

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        optimizer: Any,
        states: dict[str, Any],
        folder: str,
        sd_adapter: Any | None = None,
    ) -> None:
        self.enable = config.enable
        if not self.enable:
            return

        self.folder = filesystem.join(folder, config.folder)
        self.interval = config.interval
        self._storage = _FilesystemCheckpointStorage()

        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                # Wrapped rather than passed bare: see OptimizerWrapper. DCP
                # writes into the tensors a Stateful reports, so a resumed run's
                # fresh optimizer needs its Adam moments materialized first.
                OPTIMIZER: OptimizerWrapper(optimizer),
            }
        )

        # -- loading and saving policy --
        self.load_only = config.load_only
        self.exclude_from_loading = config.exclude_from_loading
        self.initial_load_path = config.initial_load_path
        self.initial_load_model_only = config.initial_load_model_only
        self.initial_load_in_hf = config.initial_load_in_hf
        self.initial_load_in_hf_quantized = config.initial_load_in_hf_quantized

        self.enable_first_step_checkpoint = config.enable_first_step_checkpoint
        self.last_save_model_only = config.last_save_model_only
        self.last_save_in_hf = config.last_save_in_hf
        self.export_dtype = EXPORT_DTYPE_MAP[config.export_dtype]

        self.sd_adapter = sd_adapter
        # hpmesh declares the HF safetensors options in its config but ships no
        # state-dict adapter to drive them, so every one of them is a capability
        # gap rather than a working option. Reject here -- at construction --
        # rather than where the option is first used, because that is minutes
        # into a run for a save. A loud rejection keeps the gap visible; a
        # silently ignored flag would train happily and write the wrong format.
        if self.sd_adapter is None:
            hf_options = [
                name
                for name, enabled in (
                    ("last_save_in_hf", self.last_save_in_hf),
                    ("initial_load_in_hf", self.initial_load_in_hf),
                    (
                        "initial_load_in_hf_quantized",
                        self.initial_load_in_hf_quantized,
                    ),
                )
                if enabled
            ]
            if hf_options:
                raise ValueError(
                    f"checkpoint.{', '.join(hf_options)} is not wired: the HF "
                    "safetensors paths need a state_dict adapter, and hpmesh "
                    "ships none. Use the native DCP format, or supply one by "
                    "passing sd_adapter to CheckpointManager."
                )

        # -- async and distributed infrastructure --
        try:
            self.async_mode = AsyncMode(config.async_mode)
        except ValueError as e:
            raise ValueError(
                f"Unknown checkpoint async_mode {config.async_mode}"
            ) from e

        # A gloo group, not the training backend: the async save thread
        # communicates over this group while the main thread keeps using NCCL.
        # Reusing the training group would have the two contend for the same
        # collectives.
        self.pg: dist.ProcessGroup | None = None
        if self.async_mode in (AsyncMode.ASYNC, AsyncMode.ASYNC_WITH_PINNED_MEM):
            self.pg = dist.new_group(backend="gloo")

        self.stager: DefaultStager | None = None
        self.staging_future: Future | None = None
        self.save_future: Future | None = None

        # -- retention policy --
        self.keep_latest_k = config.keep_latest_k
        self.purge_exempt = config.purge_exempt
        self.purge_thread: threading.Thread | None = None
        if self.keep_latest_k > 0:
            self.purge_queue: queue.Queue[str | None] = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread,
                args=(self.purge_queue, self._storage.remove),
                daemon=True,
            )
            self.purge_thread.start()

        logger.info(
            "Checkpointing active. Checkpoints will be loaded from and saved "
            f"to {self.folder}"
        )

    def __del__(self) -> None:
        self.close()

    def _close(self) -> None:
        if (
            hasattr(self, "purge_thread")
            and self.purge_thread
            and self.purge_thread.is_alive()
        ):
            self.purge_queue.put(None)
            self.purge_thread.join()

        if self.stager is not None:
            self.stager.close()

    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Future | AsyncSaveResponse | None:
        """Run the DCP save.

        Orchestrates the state dict transformation (to HuggingFace format, when
        asked), picks the storage writer, and dispatches on the requested
        synchronicity mode.

        Args:
            state_dict: the state dict to save.
            checkpoint_id: path identifying the checkpoint.
            async_mode: the saving/staging strategy.
            enable_garbage_collection: run a manual GC collect after the save.
            to_hf: use a ``HuggingFaceStorageWriter`` and adapt the state dict to
                safetensors and HF model definitions.

        Returns:
            ``None`` when saved synchronously (``AsyncMode.DISABLED``); a
            ``Future`` for ``AsyncMode.ASYNC`` (tracks disk I/O); an
            ``AsyncSaveResponse`` for ``AsyncMode.ASYNC_WITH_PINNED_MEM``
            (tracks both staging and disk I/O).
        """
        ret: Future | AsyncSaveResponse | None = None

        storage_writer: HuggingFaceStorageWriter | None = None
        fqn_to_index_mapping: dict[Any, int] | None = None

        if to_hf:
            if self.sd_adapter is None:
                raise ValueError("sd_adapter is required for to_hf=True")
            state_dict = self.sd_adapter.to_hf(state_dict)
            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping

            # If sharded, save into a subdirectory then consolidate up into
            # checkpoint_id.
            save_path = (
                os.path.join(checkpoint_id, "sharded")
                if fqn_to_index_mapping
                else checkpoint_id
            )
            storage_writer = HuggingFaceStorageWriter(
                path=save_path,
                save_distributed=True,
                fqn_to_index_mapping=fqn_to_index_mapping,
                enable_consolidation=not fqn_to_index_mapping,
            )
            # NOTE: with no fqn_to_index_mapping all FQNs go into a single
            # unified file, and the StorageWriter consolidates internally on one
            # rank. With a mapping the weights are spread across multiple files
            # (sharded), so the writer's internal consolidation is off and
            # consolidate_safetensors_files_on_every_rank does the merging.

        # For HF the storage_writer owns the path, so DCP gets no checkpoint_id.
        checkpoint_save_id = None if to_hf else checkpoint_id

        if async_mode == AsyncMode.ASYNC:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
            )
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self.stager,
            )
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
            )

        if to_hf and fqn_to_index_mapping:
            consolidate_safetensors_files_on_every_rank(
                input_dir=os.path.join(checkpoint_id, "sharded"),
                output_dir=checkpoint_id,
                fqn_to_index_mapping=fqn_to_index_mapping,
                num_threads=5,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def _load_checkpoint(
        self,
        states: dict[str, Any],
        checkpoint_id: str,
        *,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        """Restore selected states through DCP or its HuggingFace reader.

        Handles both standard DCP sharded checkpoints and HuggingFace
        safetensors. Loading from HF routes through the adapter, which maps FQNs
        and handles format-specific sharding.

        Raises:
            ValueError: if ``from_hf`` is set without an ``sd_adapter``.
        """
        state_dict = self._flattened_model_states_sd(states)

        if from_hf:
            if self.sd_adapter is None:
                raise ValueError(
                    "trying to load checkpoint in HF safetensors format, "
                    "but sd_adapter is not provided."
                )

            hf_state_dict = self.sd_adapter.to_hf(state_dict)
            hf_storage_reader = self.sd_adapter.get_hf_storage_reader(
                checkpoint_id, from_quantized
            )

            dcp.load(hf_state_dict, storage_reader=hf_storage_reader)

            state_dict = self.sd_adapter.from_hf(hf_state_dict)
            states[MODEL].load_state_dict(state_dict)
        else:
            dcp.load(state_dict, checkpoint_id=checkpoint_id)

            # The model states were flattened into the top-level dict by
            # ``_flattened_model_states_sd``, so DCP writes into those flat
            # entries rather than into the model. Pushing them back is a second
            # step, done explicitly here.
            if MODEL in states:
                states[MODEL].load_state_dict(state_dict)

    def _save(self, curr_step: int, last_step: bool = False) -> bool:
        """Save the checkpoint for the current step.

        A save happens when any of these hold:
        1. It is the initial seed checkpoint (step 0).
        2. The current step matches the configured interval.
        3. ``last_step`` is set, which forces a save regardless of interval.

        Returns:
            Whether a checkpoint was written (or staged, in async modes).
        """
        if not self._should_save(curr_step, last_step):
            return False

        self.maybe_wait_for_saving()
        self._purge_stale_checkpoints(saving_step=curr_step)

        begin = time.monotonic()
        checkpoint_phase = (
            "saving" if self.async_mode == AsyncMode.DISABLED else "staging"
        )
        logger.info(f"{checkpoint_phase.capitalize()} the checkpoint.")

        if last_step:
            self._save_last_step(curr_step)
            logger.info(
                f"Last step checkpoint completed in {time.monotonic() - begin:.2f}s"
            )
            return True

        checkpoint_id = self._create_checkpoint_id(curr_step)
        states = self._flattened_model_states_sd()

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            if self.stager is None:
                self.stager = DefaultStager(
                    StagingOptions(
                        use_pinned_memory=True,
                        use_shared_memory=True,
                        use_async_staging=True,
                        use_non_blocking_copy=True,
                    )
                )

            result = self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=self.async_mode,
            )
            # No GC needed on this path: the staging buffers are reused.
            if not isinstance(result, AsyncSaveResponse):
                raise TypeError(
                    "ASYNC_WITH_PINNED_MEM save must return an AsyncSaveResponse, "
                    f"got {type(result).__name__}"
                )
            self.staging_future = result.staging_completion
            self.save_future = result.upload_completion

        elif self.async_mode == AsyncMode.ASYNC:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            result = self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=self.async_mode,
            )
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            if not isinstance(result, Future):
                raise TypeError(
                    f"ASYNC save must return a Future, got {type(result).__name__}"
                )
            self.save_future = result

        else:
            self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=AsyncMode.DISABLED,
                enable_garbage_collection=True,
            )

        logger.info(
            f"Finished {checkpoint_phase} the checkpoint in "
            f"{time.monotonic() - begin:.2f} seconds."
        )
        return True

    def _maybe_wait_for_staging(self) -> None:
        """Wait for staging to finish, when it is active.

        In ``ASYNC_WITH_PINNED_MEM`` mode the checkpoint data is first staged
        from device memory to pinned host memory. That staging is asynchronous
        and designed to overlap with the following training steps. This method
        ensures it has finished before the next checkpoint cycle begins or before
        training completes, avoiding memory contention on the pinned buffers.

        Raises:
            RuntimeError: if a staging future is set while the async mode is not
                ``ASYNC_WITH_PINNED_MEM``.
        """
        if self.staging_future is None:
            return

        if self.async_mode != AsyncMode.ASYNC_WITH_PINNED_MEM:
            raise RuntimeError(
                "self.staging_future is not None, "
                "but self.async_mode isn't ASYNC_WITH_PINNED_MEM."
            )

        self.staging_future.result()
        self.staging_future = None

    def _wait_for_saving(self) -> None:
        """Wait for any background save to complete.

        Blocking; ensures all checkpoint data has reached storage. The tracking
        future is cleared first so a failed save is not retried by a later
        ``close()`` or ``__del__``.

        Raises:
            RuntimeError: if a save future is set while the mode is ``DISABLED``.
        """
        if self.async_mode == AsyncMode.DISABLED:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is DISABLED."
            )

        save_future = self.save_future
        assert save_future is not None
        self.save_future = None
        save_future.result()

    def _is_resumable_checkpoint(self, checkpoint_dir: str) -> bool:
        return self._storage.isfile(filesystem.join(checkpoint_dir, ".metadata"))

    def _is_valid_checkpoint(self, checkpoint_dir: str) -> bool:
        return self._is_resumable_checkpoint(checkpoint_dir) or (
            self._storage.isfile(
                filesystem.join(checkpoint_dir, "model.safetensors.index.json")
            )
        )

    def _flattened_model_states_sd(
        self, state_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Flatten model parameters into the top-level state dictionary.

        Merges the model's state dict into the top-level dict while leaving
        auxiliary states (optimizer, lr_scheduler) unflattened, giving the DCP
        writer one consistent format.

        Args:
            state_dict: a custom dict to flatten. Defaults to ``None``, which
                uses the instance's own ``states``.
        """
        states = state_dict if state_dict is not None else self.states
        sd = {k: v for k, v in states.items() if k != MODEL}
        if MODEL in states:
            sd.update(states[MODEL].state_dict())
        return sd

    def _save_last_step(self, curr_step: int) -> None:
        """Save the final checkpoint at the end of training.

        Handles the specific requirements of the final artifact: allowing model
        weights only (stripping optimizer state), converting to the export dtype,
        and optionally writing HF-compatible output.
        """
        # When last_save_model_only is False the full training state is written
        # with no dtype conversion, so the run can still be resumed. Otherwise
        # training is assumed complete and only the model is written, converted
        # if the current dtype differs from the export dtype.
        if self.last_save_in_hf:
            assert self.last_save_model_only, (
                "Only model can be saved when saving in HF safetensors format."
            )

        if self.last_save_model_only:
            states = self.states[MODEL].state_dict()

            states = {
                k: v.to(self.export_dtype)
                if isinstance(v, torch.Tensor)
                and v.is_floating_point()
                and v.dtype != self.export_dtype
                else v
                for k, v in states.items()
            }
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flattened_model_states_sd()

        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
        )


# Bound late so the module attribute ``CheckpointManager`` (used by the
# ``from .dcp import CheckpointManager`` export and by tests) stays the class,
# while DATALOADER and LR_SCHEDULER remain importable from here for callers that
# build a fuller ``states`` dict.
__all__ = [
    "AsyncMode",
    "CheckpointManager",
    "DATALOADER",
    "EXPORT_DTYPE_MAP",
    "LR_SCHEDULER",
]
