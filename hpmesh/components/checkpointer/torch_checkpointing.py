"""The ``torch_checkpointing`` checkpoint manager.

Vendored from torchtitan's ``components/checkpointer/torch_checkpointing.py``.
Its class structure, policy wiring, and comments are kept verbatim where they
carry a reason; three things were altered, and each is a subtraction or a
necessary adaptation rather than a redesign.

**The backend is not installed here, so its imports are deferred.** Every
``torch_checkpointing`` and ``torch_checkpointing.*`` import in torchtitan sits
at module scope. Doing that here would make this file unimportable on any machine
without the package -- which is every machine hpmesh runs on today -- and would
take ``components/checkpointer/__init__.py`` down with it. They are hoisted into
``_require_torch_checkpointing()`` instead, called from ``__init__``. So:

* this module imports fine, and so does the package that re-exports it;
* constructing a ``TorchCheckpointingManager`` without the backend installed
  raises ``ImportError`` with the install hint, at the earliest point the
  dependency is actually needed.

**No ``structured_logger``.** torchtitan stamps ``sl.add_step_tag`` and writes a
``checkpoint_logging_context`` step into the async save subprocess. Neither
exists here; the backend still emits its own metrics on the
``torch_checkpointing`` logger, which is re-leveled in the subprocess the same
way torchtitan does it.

**No tyro.** The config is a plain dataclass (see ``base.py``), and
``purge_exempt`` is a plain callable.

Reachability note, stated plainly because it is easy to mistake for working code:
nothing in hpmesh constructs this manager yet. It is here because it is the third
of torchtitan's checkpointer backends and the port is meant to be complete, and
because the moment someone installs ``torch_checkpointing`` it is the faster
backend with no further porting work. Until then it is unexercised: the tests in
``test_checkpointer.py`` cover only the import guard and the per-config branch of
``__init__``. Nothing above that line has ever run, so the backend-dependent
paths -- ``_save`` / ``_load_checkpoint`` / ``_save_last_step`` and the storage
adapter -- are a faithful transcription rather than verified behaviour, and the
first real use should start by round-tripping a checkpoint on a machine that has
the package installed.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ...utils import filesystem
from ...utils.gc import GarbageCollection

if TYPE_CHECKING:
    from ...trainer.config import CheckpointConfig

from .base import (
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    BaseCheckpointManager,
    ModelWrapper,
    purge_thread,
)
from .dcp import EXPORT_DTYPE_MAP

logger = logging.getLogger(__name__)


DEFAULT_TORCH_CHECKPOINTING_BARRIER_TCPSTORE_PORT = 43001
_DEFAULT_BARRIER_INIT_TIMEOUT_SEC = 60
_DEFAULT_BARRIER_TIMEOUT_SEC = 600

# Index the HF consolidation writes at the root of a final export; the backend
# names it after the checkpoint item it consolidated.
_HF_INDEX_FILE_NAME = f"{MODEL}.safetensors.index.json"

# Logger the backend emits its checkpoint events and metrics on.
CHECKPOINTING_LOGGER_NAME = "torch_checkpointing"

_INSTALL_HINT = (
    "the torch_checkpointing backend is not installed. Install it, or use the "
    "DCP checkpoint manager (components/checkpointer/dcp.py), which needs only "
    "torch.distributed.checkpoint."
)


def _require_torch_checkpointing():
    """Import the backend lazily, or fail with an actionable message.

    torchtitan imports these at module scope. Doing that here would make this
    module -- and everything that re-exports it -- unimportable wherever the
    package is absent, which is the common case. Deferring the import to the
    point of use keeps the failure at the earliest place it is actually needed
    and lets the rest of the checkpointer subpackage load regardless.
    """
    try:
        from torch_checkpointing.barriers import TCPStoreBarrierConfig
        from torch_checkpointing.checkpoint_layout import (
            LayoutInfo,
            SafetensorsSerialization,
        )
        from torch_checkpointing.checkpoint_manager import (
            CheckpointManager as BackendCheckpointManager,
        )
        from torch_checkpointing.checkpoint_writer import CheckpointWriterConfig
        from torch_checkpointing.config import (
            AsyncCheckpointSaverConfig,
            CheckpointSaverConfig,
            SyncCheckpointSaverConfig,
        )
        from torch_checkpointing.default_resharder import DefaultResharder
        from torch_checkpointing.distributed_metadata import (
            METADATA_FILE_NAME as TORCH_CHECKPOINTING_METADATA_FILE_NAME,
        )
        from torch_checkpointing.hf.consolidation import (
            consolidate_hf_safetensors_checkpoint,
        )
        from torch_checkpointing.logging_utils import (
            EventLogger,
            checkpoint_logging_context,
        )
        from torch_checkpointing.schema import ItemSpec
        from torch_checkpointing.staging import CheckpointStagerConfig
        from torch_checkpointing.storage.base_storage import (
            Storage,
            StorageConfig,
        )
        from torch_checkpointing.storage.filesystem import (
            LocalFileSystemStorageConfig,
        )
    except ImportError as error:
        raise ImportError(f"{_INSTALL_HINT} ({error})") from error

    return _Backend(
        TCPStoreBarrierConfig=TCPStoreBarrierConfig,
        LayoutInfo=LayoutInfo,
        SafetensorsSerialization=SafetensorsSerialization,
        BackendCheckpointManager=BackendCheckpointManager,
        CheckpointWriterConfig=CheckpointWriterConfig,
        AsyncCheckpointSaverConfig=AsyncCheckpointSaverConfig,
        CheckpointSaverConfig=CheckpointSaverConfig,
        SyncCheckpointSaverConfig=SyncCheckpointSaverConfig,
        DefaultResharder=DefaultResharder,
        METADATA_FILE_NAME=TORCH_CHECKPOINTING_METADATA_FILE_NAME,
        consolidate_hf_safetensors_checkpoint=consolidate_hf_safetensors_checkpoint,
        checkpoint_logging_context=checkpoint_logging_context,
        EventLogger=EventLogger,
        ItemSpec=ItemSpec,
        CheckpointStagerConfig=CheckpointStagerConfig,
        Storage=Storage,
        StorageConfig=StorageConfig,
        LocalFileSystemStorageConfig=LocalFileSystemStorageConfig,
    )


@dataclass(frozen=True, slots=True)
class _Backend:
    """The lazily imported ``torch_checkpointing`` names, bundled for transit.

    A frozen dataclass rather than a dict so every attribute access is a static
    name -- a typo fails at import time instead of midway through a save.
    """

    TCPStoreBarrierConfig: Any
    LayoutInfo: Any
    SafetensorsSerialization: Any
    BackendCheckpointManager: Any
    CheckpointWriterConfig: Any
    AsyncCheckpointSaverConfig: Any
    CheckpointSaverConfig: Any
    SyncCheckpointSaverConfig: Any
    DefaultResharder: Any
    METADATA_FILE_NAME: str
    consolidate_hf_safetensors_checkpoint: Any
    checkpoint_logging_context: Any
    EventLogger: Any
    ItemSpec: Any
    CheckpointStagerConfig: Any
    Storage: Any
    StorageConfig: Any
    LocalFileSystemStorageConfig: Any


class _BackendCheckpointStorage:
    """``CheckpointStorage`` backed by a ``torch_checkpointing`` ``Storage``.

    Path probes go through the same ``Storage`` the backend saves and loads
    with, or a caller-supplied remote storage would be written by the backend
    and read by something else.

    ``Storage`` has no ``isfile``, so it is composed from the two probes it does
    have: ``exists`` covers files and directories alike, so excluding directories
    leaves exactly the existing non-directory entries.
    """

    def __init__(self, storage) -> None:
        self._storage = storage

    def isdir(self, path: str) -> bool:
        return self._storage.isdir(Path(path))

    def isfile(self, path: str) -> bool:
        target = Path(path)
        return self._storage.exists(target) and not self._storage.isdir(target)

    def listdir(self, path: str) -> list[str]:
        return self._storage.ls(Path(path))

    def remove(self, path: str) -> None:
        self._storage.rmdir(Path(path))


def _init_subprocess_logging(
    init_fn: Callable[..., None] | None,
    init_args: tuple[Any, ...],
) -> None:
    """Re-establish logging inside the async save subprocess.

    The subprocess does not inherit the parent's logging handlers, so the
    backend's checkpoint records would otherwise be lost.
    """
    if init_fn is not None:
        init_fn(*init_args)

    backend_logger = logging.getLogger(CHECKPOINTING_LOGGER_NAME)
    if backend_logger.level == logging.NOTSET and not backend_logger.isEnabledFor(
        logging.INFO
    ):
        backend_logger.setLevel(logging.INFO)


def _item_specs(backend: _Backend) -> dict[str, Any]:
    resharder = backend.DefaultResharder()
    return {
        MODEL: backend.ItemSpec(
            requires_copy=True,
            resharder=resharder,
            required=False,
        ),
        OPTIMIZER: backend.ItemSpec(
            requires_copy=True,
            resharder=resharder,
            required=False,
        ),
    }


def _writer_config(backend: _Backend, *, use_barrier: bool):
    return backend.CheckpointWriterConfig(
        checkpoint_write_barrier_timeout_sec=_DEFAULT_BARRIER_TIMEOUT_SEC,
        barrier_config=(
            backend.TCPStoreBarrierConfig(
                master_address=os.environ.get("MASTER_ADDR", "localhost"),
                tcpstore_port=DEFAULT_TORCH_CHECKPOINTING_BARRIER_TCPSTORE_PORT,
                timeout_barrier_init_sec=_DEFAULT_BARRIER_INIT_TIMEOUT_SEC,
                use_checkpoint_barrier_tcpstore_libuv=True,
            )
            if use_barrier
            else None
        ),
    )


def _async_save_config(backend: _Backend):
    return backend.AsyncCheckpointSaverConfig(
        writer_config=_writer_config(backend, use_barrier=True),
        staging_config=backend.CheckpointStagerConfig(use_pinned_memory=True),
        wait_timeout_secs=_DEFAULT_BARRIER_TIMEOUT_SEC,
    )


def _sync_save_config(backend: _Backend, *, use_barrier: bool = True):
    return backend.SyncCheckpointSaverConfig(
        writer_config=_writer_config(backend, use_barrier=use_barrier),
        wait_timeout_secs=_DEFAULT_BARRIER_TIMEOUT_SEC,
    )


def _default_backend_config(
    backend: _Backend,
    save_config,
    *,
    storage_config=None,
    items: dict[str, Any] | None = None,
    subprocess_init_fn: Callable[..., None] | None = None,
    subprocess_init_args: tuple[Any, ...] = (),
    pre_finalize_callback: Callable[[str, Any], None] | None = None,
):
    """Assemble the backend's own ``Config``.

    torchtitan routes the async-save subprocess through
    ``structured_logger``'s subprocess-init hook so the subprocess can log. There
    is no structured logger here, so a caller-supplied ``subprocess_init_fn`` is
    passed through unchanged, wrapped only in ``_init_subprocess_logging`` to
    re-level the backend logger -- which is what makes the subprocess emit
    anything at all.
    """
    if isinstance(save_config, backend.AsyncCheckpointSaverConfig):
        subprocess_init_args = (subprocess_init_fn, subprocess_init_args)
        subprocess_init_fn = _init_subprocess_logging
    return backend.BackendCheckpointManager.Config(
        items=_item_specs(backend) if items is None else items,
        default=backend.ItemSpec(requires_copy=False),
        save=save_config,
        storage_config=storage_config,
        subprocess_init_fn=subprocess_init_fn,
        subprocess_init_args=subprocess_init_args,
        pre_finalize_callback=pre_finalize_callback,
    )


class TorchCheckpointingManager(BaseCheckpointManager):
    """Checkpoint manager backed by ``torch_checkpointing``.

    Args:
        storage_config: backend storage for reading and writing checkpoints.
            Defaults to the local filesystem. An init parameter rather than a
            ``Config`` field because the backend storage object is not a
            command-line surface; callers that need remote storage pass it
            programmatically.
    """

    def __init__(
        self,
        config: CheckpointConfig,
        *,
        model_parts: list[Any],
        optimizer: Any,
        lr_scheduler: Any,
        states: dict[str, Any],
        folder: str,
        sd_adapter: Any | None = None,
        storage_config: Any | None = None,
    ) -> None:
        self.enable = config.enable
        if not self.enable:
            return

        backend = _require_torch_checkpointing()
        self._backend = backend

        self.save_future: Future[Any] | None = None
        self.staging_future: Future[Any] | None = None
        self.purge_thread: threading.Thread | None = None

        self.folder = filesystem.join(folder, config.folder)
        # Checked here, not only in the storage adapter: a save runs no path
        # probe when retention is off, so it would otherwise reach the backend
        # and be mangled by ``Path`` rather than failing.
        for label, candidate in (
            ("checkpoint.folder", self.folder),
            ("checkpoint.initial_load_path", config.initial_load_path),
        ):
            if candidate and filesystem.is_remote(candidate):
                raise ValueError(
                    f"{label} is a remote URI ({candidate!r}); remote URIs are "
                    "not yet supported by torch_checkpointing. Use the DCP "
                    "checkpoint manager for remote storage."
                )

        self.interval = config.interval
        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizer,
                # After OPTIMIZER, deliberately: DCP loads in this order, and the
                # scheduler's restore reads the optimizers' ``base_lrs``. Without
                # it a resumed run's fresh scheduler restarts ``last_epoch`` at 0,
                # so a warmup or decay curve restarts on the step after a resume.
                LR_SCHEDULER: lr_scheduler,
            }
        )

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
        self.keep_latest_k = config.keep_latest_k
        self.purge_exempt = config.purge_exempt

        save_config = (
            _sync_save_config(backend, use_barrier=False)
            if self.load_only
            else _async_save_config(backend)
        )
        manager_config = _default_backend_config(
            backend,
            save_config,
            storage_config=storage_config,
        )
        self._manager_config = manager_config
        storage_config = (
            self._manager_config.storage_config
            or backend.LocalFileSystemStorageConfig()
        )
        self._storage = _BackendCheckpointStorage(storage_config.create_storage())
        self._prewarmed = False

        self.sd_adapter = sd_adapter
        if self.last_save_in_hf and self.sd_adapter is None:
            raise ValueError(
                "checkpoint.last_save_in_hf is True, but sd_adapter is not provided."
            )

        self._manager = self._manager_config.build()

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

        # Last: the backend manager is built above and ``__del__`` runs ``close``
        # on whatever was left behind if any of that raised.
        self._initialized = True

    def __del__(self) -> None:
        # __init__ can fail before the backend manager is built. In that case
        # this object owns no backend resources to close.
        if hasattr(self, "_manager"):
            self.close()

    def _load_checkpoint(
        self,
        states: dict[str, Any],
        checkpoint_id: str,
        *,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        if from_hf:
            raise ValueError(
                "TorchCheckpointingManager does not yet support loading "
                "Hugging Face checkpoints."
            )
        if not self._is_valid_checkpoint(checkpoint_id):
            raise ValueError(
                f"Checkpoint {checkpoint_id!r} is not a native "
                "torch_checkpointing checkpoint."
            )
        # strict: the backend skips by default anything the checkpoint does not
        # carry, which would silently leave parameters at their initialized
        # values and resume from a model that is not the one that was saved.
        # exclude_from_loading is applied by _states_to_load, so anything still
        # in ``states`` here is genuinely required.
        state_dict = self._stateful_to_state_dict(states)
        loaded = self._manager.load(
            checkpoint_id,
            into=state_dict,
            strict=True,
        )
        for key, target in states.items():
            if self._is_stateful(target):
                target.load_state_dict(state_dict[key])
            elif loaded[key] is not target:
                raise TypeError(
                    f"Cannot restore non-Stateful checkpoint state {key!r} of type "
                    f"{type(target).__name__}"
                )

    @staticmethod
    def _is_stateful(obj: Any) -> bool:
        from torch.distributed.checkpoint.stateful import Stateful

        return isinstance(obj, Stateful)

    @staticmethod
    def _stateful_to_state_dict(states: dict[str, Any]) -> dict[str, Any]:
        from torch.distributed.checkpoint.state_dict_saver import (
            _stateful_to_state_dict,
        )

        return _stateful_to_state_dict(states)

    def _save(self, curr_step: int, last_step: bool = False) -> bool:
        should_save = self._should_save(curr_step, last_step)
        # Prewarm on a step we are not saving, so the first real save does not
        # pay for pinned-buffer allocation.
        if not should_save and self._should_prewarm():
            self._manager.prewarm_staging(self._stateful_to_state_dict(self.states))
            self._prewarmed = True
        if not should_save:
            return False

        # The backend stamps its own events from this context and carries it
        # into the async save subprocess, so without it every forwarded backend
        # metric reports step=None.
        self._backend.checkpoint_logging_context.update(step=curr_step)
        self.maybe_wait_for_saving()
        # Always preserve the current step's published and staging directories.
        self._purge_stale_checkpoints(
            saving_step=curr_step,
            staging_dir_prefix=(
                self._manager_config.save.writer_config.temp_dir_prefix
            ),
        )

        if last_step:
            self._save_last_step(curr_step)
        else:
            self.save_future = self._manager.save(
                self._create_checkpoint_id(curr_step),
                self._stateful_to_state_dict(self.states),
            )
            self._prewarmed = True

        return True

    def _is_resumable_checkpoint(self, checkpoint_dir: str) -> bool:
        """Whether automatic loading may select ``checkpoint_dir``.

        Unlike ``_is_valid_checkpoint``, this excludes final Hugging Face
        exports: HF loading has not landed in torch_checkpointing, so automatic
        loading cannot select those exports.
        """
        return self._storage.isfile(
            filesystem.join(checkpoint_dir, self._backend.METADATA_FILE_NAME)
        )

    def _is_valid_checkpoint(self, checkpoint_dir: str) -> bool:
        # Either published layout counts as valid:
        #
        #   step-3/                     step-5/  (final HF export)
        #   |-- metadata.pkl            |-- model.safetensors.index.json
        #   |-- model_0.pt              |-- model-00001-of-00002.safetensors
        #   |-- model_1.pt              |-- model-00002-of-00002.safetensors
        #   |-- optimizer_0.pt          +-- sharded/
        #   +-- optimizer_1.pt              |-- metadata.pkl
        #                                   |-- model_0.safetensors
        #                                   +-- model_1.safetensors
        #
        # Without the HF shape, a finished export looks abandoned and retention
        # deletes it on the next run.
        return self._is_resumable_checkpoint(checkpoint_dir) or self._storage.isfile(
            filesystem.join(checkpoint_dir, _HF_INDEX_FILE_NAME)
        )

    def _maybe_wait_for_staging(self) -> None:
        # BaseCheckpointManager.close() calls this to wait for in-flight staging.
        # If _save_last_step already closed the backend manager, that close
        # drained staging but left this lock usable, so acquiring it cannot hang.
        with self._manager.lock():
            pass

    def _wait_for_saving(self) -> None:
        # Clear the active save before waiting so close() does not retry a
        # failure.
        save_future = self.save_future
        assert save_future is not None
        self.save_future = None
        save_future.result(timeout=self._manager_config.save.wait_timeout_secs)

    def _close(self) -> None:
        if self.staging_future is not None:
            self.staging_future.result()
            self.staging_future = None
        try:
            self.maybe_wait_for_saving()
        finally:
            try:
                if self.purge_thread is not None and self.purge_thread.is_alive():
                    self.purge_queue.put(None)
                    self.purge_thread.join()
            finally:
                # _save_last_step may already have closed the manager; the
                # backend's close() returns immediately when it has.
                self._manager.close()

    def _save_last_step(self, curr_step: int) -> None:
        if self.last_save_model_only:
            model_state = self.states[MODEL].state_dict()
            # Cast floating-point tensors to the export dtype, preserve other
            # buffers.
            model_state = {
                key: value.to(self.export_dtype)
                if isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and value.dtype != self.export_dtype
                else value
                for key, value in model_state.items()
            }
            states: dict[str, Any] = {MODEL: model_state}
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            states = self.states
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")

        # The final save must land before the process exits, so retire the async
        # manager and write synchronously through a fresh one.
        checkpoint_id = self._create_checkpoint_id(curr_step)
        self._manager.close()
        storage_config = self._manager_config.storage_config
        input_checkpoint_id = checkpoint_id
        item_specs: dict[str, Any] | None = None
        pre_finalize_callback: Callable[[str, Any], None] | None = None
        if self.last_save_in_hf:
            assert self.sd_adapter is not None
            states = {MODEL: self.sd_adapter.to_hf(states[MODEL])}
            # Ranks write safetensors shards into a nested directory; the
            # pre-finalize callback consolidates them up into checkpoint_id, so
            # the published checkpoint is HF-layout rather than sharded.
            input_checkpoint_id = filesystem.join(checkpoint_id, "sharded")
            item_specs = _item_specs(self._backend)
            model_spec = item_specs[MODEL]
            item_specs[MODEL] = self._backend.ItemSpec(
                requires_copy=model_spec.requires_copy,
                layout=self._backend.LayoutInfo(
                    f"{MODEL}_{{rank}}.safetensors",
                    self._backend.SafetensorsSerialization(),
                ),
                resharder=model_spec.resharder,
                required=model_spec.required,
            )
            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping
            hf_storage_config = (
                storage_config
                or self._backend.LocalFileSystemStorageConfig(use_direct_io=False)
            )

            # The backend invokes the callback after each rank finishes writing
            # but before the write barrier and the atomic rename of the temp dir
            # to its final path, passing the directory the shards were actually
            # written to. Consolidation repacks the per-rank shards into
            # HF-layout files in the checkpoint directory.
            def pre_finalize_callback(staged: str, _event_logger) -> None:  # noqa: F811
                from ...accelerator import dist_utils

                # All ranks must finish writing before any rank consolidates.
                dist_utils.barrier()
                self._backend.consolidate_hf_safetensors_checkpoint(
                    staged,
                    output_dir=checkpoint_id,
                    item_key=MODEL,
                    fqn_to_index_mapping=fqn_to_index_mapping,
                    storage_config=hf_storage_config,
                )

        manager_config = _default_backend_config(
            self._backend,
            _sync_save_config(self._backend),
            storage_config=storage_config,
            items=item_specs,
            pre_finalize_callback=pre_finalize_callback,
        )
        manager = manager_config.build()
        try:
            manager.save(
                input_checkpoint_id,
                self._stateful_to_state_dict(states),
            )
        finally:
            manager.close()
        GarbageCollection.collect("GC collection invoked by checkpointer.")

    def _should_prewarm(self) -> bool:
        return self.enable and not self._prewarmed and not self.load_only
