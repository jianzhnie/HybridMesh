# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The checkpoint contract: what a checkpoint holds, and the policies over it.

Vendored from torchtitan's ``components/checkpointer/base.py``. The class
structure is kept -- a backend-agnostic ``BaseCheckpointManager`` with the
shared policies (retention, step discovery, async draining, load selection) and
concrete subclasses that only implement how bytes are read and written -- because
that split is what makes the DCP and torch_checkpointing managers share 90% of
their logic in torchtitan, and it is what will let hpmesh do the same.

Four deliberate departures, all subtractions:

* **The config is not defined here.** torchtitan's ``BaseCheckpointManager`` is
  a ``Configurable`` whose nested ``Config.build()`` constructs ``Config._owner``,
  which is what keeps a manager's defaults next to the manager. hpmesh keeps all
  configuration in one module (``hpmesh.trainer.config``), so the managers take
  an explicit ``config`` argument of the type defined there.

* **No tyro.** ``purge_exempt`` was
  ``Annotated[Function.Config | None, tyro.conf.Suppress]`` -- a CLI-suppressed
  pluggable predicate. It is typed as a plain ``Callable[[int], bool] | None``
  here; nothing parses it off a command line.

* **No ``structured_logger`` spans.** torchtitan wraps load/save in
  ``sl.log_trace_span`` and stamps ``sl.add_step_tag``. hpmesh has no structured
  logger; the ``logger.info`` lines that carry the same information are kept.

* **``GarbageCollection`` is local.** torchtitan's version carries
  structured-logger tags in ``run``; hpmesh's ``utils/gc.py`` keeps the
  collection and drops the tags. Its ``run`` is likewise not wired into the
  training loop yet -- the checkpointer uses only ``collect``.

One addition: ``OptimizerWrapper`` and ``init_optim_state``. torchtitan has no
equivalent here because its ``components/optimizer.py`` supplies both, as part
of an ``OptimizersContainer`` that also re-keys state dicts by FQN to survive
pipeline parallelism. hpmesh keeps the two concerns in this wrapper instead:
the *materialization* is load-bearing in both directions, and the FQN re-keying
is an optional mode (``model_parts=...``) the trainer turns on under PP -- see
``OptimizerWrapper``.

``MODEL`` / ``OPTIMIZER`` / ``LR_SCHEDULER`` / ``DATALOADER`` / ``TRAIN_STATE``
are the top-level state keys a checkpoint is keyed by. hpmesh shares none of
torchtitan's component containers, so which of them a run actually populates
differs -- see ``components/checkpointer/__init__.py`` for the mapping.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

from ...utils import filesystem
from ...utils.gc import GarbageCollection

logger = logging.getLogger(__name__)


MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


def purge_thread(
    purge_queue: queue.Queue[str | None],
    remove_path: Callable[[str], None],
) -> None:
    """Thread to purge the old checkpoints.

    Only used when ``keep_latest_k > 0``.

    Args:
        purge_queue: receives paths to purge, and the ``None`` shutdown sentinel.
        remove_path: how to delete one path, supplied by the manager's storage.
    """
    try:
        while True:
            path = purge_queue.get()
            if path is None:
                return
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            # A single failed deletion (a transient remote error, say) must not
            # kill this daemon thread; otherwise keep_latest_k would silently
            # stop purging for the rest of the run.
            try:
                remove_path(path)
            except Exception as error:  # noqa: BLE001 - one path must not stop the loop
                logger.warning(
                    "Checkpointer failed to delete %s: %s. Skipping.", path, error
                )
                continue
            logger.info(
                "Checkpointer deleted %s in %.2f seconds.",
                path,
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the purge thread.")


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``a`` and ``b`` are backed by the same storage.

    For ``DTensor`` the local shard is compared via ``_local_tensor`` rather
    than ``to_local()``, which is autograd-aware. The dispatcher-level alias
    check also supports wrapper subclasses without directly accessible storage.
    """
    if isinstance(a, DTensor):
        a = a._local_tensor
    if isinstance(b, DTensor):
        b = b._local_tensor
    return torch._C._is_alias_of(a, b)


def init_optim_state(optimizer: torch.optim.Optimizer) -> None:
    """Materialize per-parameter optimizer state without changing anything.

    Two callers need this, for the same underlying reason: PyTorch creates Adam's
    ``exp_avg``/``exp_avg_sq`` lazily, on the first ``step()``.

    * **Saving.** DCP reads whatever tensors the state dict reports. An
      optimizer that has not stepped reports none, so the checkpoint would carry
      model weights and no optimizer state -- silently, since the save succeeds.
    * **Loading.** DCP writes *into* the tensors a state dict reports rather
      than calling ``load_state_dict``. A fresh optimizer reports none, so there
      is nothing to write into and the restored run trains with a cold
      optimizer. ``OptimizerWrapper.state_dict`` calls this for that case.

    Vendored from torchtitan's ``components/optimizer/utils.init_optim_state``.
    The step runs with zero gradients and ``lr=0`` so parameters are untouched;
    the materialized state is then reset, so the first real update is still Adam
    step 1. Existing gradients and optimizer state are preserved.

    No-op when every parameter that requires a gradient already has state.
    """
    params = [param for group in optimizer.param_groups for param in group["params"]]
    missing = [
        param
        for param in params
        if param.requires_grad and not optimizer.state.get(param)
    ]
    if not missing:
        return

    saved_grads = [param.grad for param in params]
    for param in params:
        param.grad = None
    for param in missing:
        param.grad = torch.zeros_like(param)

    # Some optimizers update parameters from lr alone, independent of the
    # gradient, so lr is zeroed for the duration of the step.
    saved_lrs: list[Any] = []
    for group in optimizer.param_groups:
        if "lr" not in group:
            continue
        saved_lrs.append(group["lr"])
        group["lr"] = (
            torch.tensor(0.0) if isinstance(group["lr"], torch.Tensor) else 0.0
        )
    optimizer.step()

    # A zero lr leaves parameters alone, but Adam still advances its step count,
    # and coupled weight decay can move its moments. Reset the state that was
    # just materialized so the first real update is Adam step 1.
    if isinstance(optimizer, torch.optim.Adam | torch.optim.AdamW):
        for param in missing:
            state = optimizer.state[param]
            state["step"].zero_()
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            if "max_exp_avg_sq" in state:
                state["max_exp_avg_sq"].zero_()

    for group, lr in zip(
        (g for g in optimizer.param_groups if "lr" in g), saved_lrs, strict=True
    ):
        group["lr"] = lr
    for param, grad in zip(params, saved_grads, strict=True):
        param.grad = grad


class ModelWrapper(Stateful):
    """A ``Stateful`` view over one module or a list of them.

    Serves two purposes:

    1. **Flattening.** Combines the state dicts of several modules (individual
       chunks under pipeline parallelism) into one flat view, so the
       checkpointing code interacts with them through a single interface.
    2. **Stable-storage caching.** Caches the flattened state dict and, on every
       ``state_dict()`` call, returns tensors backed by the *same* storage.
       Async DCP staging may cache pinned host buffers keyed by the source
       storage, so keeping storage stable lets it reuse those buffers across
       saves -- this is the fast checkpoint path. Parameter tensors already
       satisfy this, since the cached view shares the parameter's storage.
       Tensors produced by module ``state_dict`` hooks (one that splits a fused
       parameter, say) may be freshly allocated each call, so they are refreshed
       in place: their storage stays put while their values track the parameters.

    Notes:
        - ``load_state_dict`` updates the underlying modules and refreshes the
          cache.
        - The module tree must not be structurally modified (keys changing,
          tensor references replaced) after wrapping, or the cache goes stale.
    """

    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cached_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        return {k: v for model in self.model for k, v in model.state_dict().items()}

    def state_dict(self) -> dict[str, Any]:
        # Recompute so hook-produced tensors reflect the current parameters,
        # then merge into the cache without changing storage objects.
        for key, value in self._get_state_dict().items():
            cached = self.cached_state_dict.get(key)
            if (
                cached is None
                or cached.shape != value.shape
                or cached.dtype != value.dtype
            ):
                self.cached_state_dict[key] = value
            elif not _shares_storage(cached, value):
                cached.copy_(value)
        return self.cached_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # strict=False because this is the flattened checkpoint dict, which
        # mixes model FQN keys with non-model keys (optimizer, lr_scheduler, ...).
        for model in self.model:
            model.load_state_dict(state_dict, strict=False)
        # Refresh the cache so state_dict() reflects the freshly loaded values.
        self.cached_state_dict = self._get_state_dict()


class OptimizerWrapper(Stateful):
    """A ``Stateful`` view over one optimizer that survives a fresh load.

    ``torch.optim.Optimizer`` already satisfies ``Stateful``, and DCP writes
    straight into the tensors a ``Stateful`` reports -- which is why a *plain*
    optimizer works for saving and for loading into an optimizer that has
    already taken a step. It does not work for the case that matters: a resumed
    run builds a fresh optimizer, whose Adam moments do not exist until its
    first ``step()``, so DCP finds no ``exp_avg`` to write into and the run
    silently restarts from a cold optimizer under warm weights.

    The fix is to give DCP the tensors to load into before it plans the load.
    ``load_state_dict`` therefore materializes the state first (via
    ``init_optim_state``, a zero-gradient, zero-lr step) and only then hands the
    state dict to the optimizer, which -- with the state present -- restores in
    place.

    PyTorch expects this of a ``Stateful`` wrapper: it calls ``load_state_dict``
    on the *object it was given*, and that object is responsible for pushing the
    values into whatever it manages. hpmesh hands DCP this wrapper instead of
    the bare optimizer, so the contract is met.

    Args:
        optimizer: the optimizer to wrap.
        model_parts: when given, optimizer state is keyed by parameter FQN
            rather than by positional index. Pipeline parallelism needs this:
            every stage's optimizer numbers its own parameters from 0, so the
            positional keys of two stages collide in one shared checkpoint
            (torchtitan solves the same collision with its
            ``OptimizersContainer``'s FQN flattening). The FQNs are read off
            ``model_parts`` in parameter order, so the caller must have built
            the optimizer over exactly those parameters in that order -- the
            trainer does (``chain(*(part.parameters() ...))``). ``None`` keeps
            the positional format, which every non-PP checkpoint already on
            disk uses.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_parts: list[nn.Module] | None = None,
    ) -> None:
        self.optimizer = optimizer
        self._fqns = (
            None
            if model_parts is None
            else [name for part in model_parts for name, _ in part.named_parameters()]
        )

    def state_dict(self) -> dict[str, Any]:
        # Materialize first, on both directions. On a save, DCP reads whatever
        # tensors this reports -- an optimizer that has never stepped reports
        # none, so the checkpoint would quietly carry weights and no optimizer
        # state. On a load, DCP calls ``state_dict()`` to learn where the values
        # are going and only afterwards calls ``load_state_dict()``, so a
        # version that materialized lazily would be a step too late for the
        # planner to have anywhere to put ``exp_avg``.
        init_optim_state(self.optimizer)
        state_dict = self.optimizer.state_dict()
        if self._fqns is None:
            return state_dict
        return {
            "state": {
                # ``state_dict`` packs state positionally; the FQN order is the
                # same parameter order, so the re-keying is a rename only.
                self._fqns[index]: state
                for index, state in state_dict["state"].items()
            },
            "param_groups": state_dict["param_groups"],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Already materialized by the ``state_dict()`` call that precedes this
        # one on the DCP path; idempotent here, and load-bearing for any caller
        # that restores without going through DCP.
        init_optim_state(self.optimizer)
        if self._fqns is not None:
            fqn_to_index = {fqn: i for i, fqn in enumerate(self._fqns)}
            state_dict = {
                "state": {
                    fqn_to_index[fqn]: state
                    for fqn, state in state_dict["state"].items()
                },
                "param_groups": state_dict["param_groups"],
            }
        self.optimizer.load_state_dict(state_dict)


@runtime_checkable
class CheckpointStorage(Protocol):
    """The path operations a checkpoint manager needs from its storage.

    Managers differ in how they read and write checkpoint bytes, but they ask
    the same handful of questions about paths: is this a checkpoint directory,
    did this metadata file land, which steps are on disk, delete this one. This
    protocol is the whole of that surface, so policies like retention and
    latest-step discovery can live on ``BaseCheckpointManager`` without knowing
    which backend answers them.

    Paths are ``str`` rather than ``Path`` because a checkpoint id may be a
    remote URI (``gs://...``) that ``Path`` would mangle -- it collapses the
    double slash. Carrying ``str`` keeps the vocabulary lossless; whether a
    given implementation can actually reach a remote URI is up to that
    implementation, which should reject what it cannot address rather than
    silently rewrite it.

    ``runtime_checkable`` so implementations can assert conformance in their
    tests. It only checks that the method names exist, which is enough to catch
    a rename that would otherwise surface as an ``AttributeError`` mid-save.
    """

    def isdir(self, path: str) -> bool:
        """Whether ``path`` is an existing directory."""
        ...

    def isfile(self, path: str) -> bool:
        """Whether ``path`` is an existing entry that is not a directory."""
        ...

    def listdir(self, path: str) -> list[str]:
        """The entry names directly under the directory ``path``."""
        ...

    def remove(self, path: str) -> None:
        """Recursively delete the directory ``path``."""
        ...


class BaseCheckpointManager(ABC):
    """Contract every checkpoint manager implements.

    The config every manager takes is a ``CheckpointConfig``, defined in
    ``hpmesh.trainer.config`` alongside every other config in the package.
    Nothing here introspects it -- a manager receives a built instance and reads
    fields off it.
    """

    enable: bool
    load_only: bool
    interval: int
    enable_first_step_checkpoint: bool
    staging_future: Future | None
    save_future: Future | None
    folder: str
    keep_latest_k: int
    states: dict[str, Any]
    exclude_from_loading: list[str]
    initial_load_path: str | None
    initial_load_model_only: bool
    initial_load_in_hf: bool
    initial_load_in_hf_quantized: bool
    sd_adapter: Any | None
    purge_exempt: Callable[[int], bool] | None = None
    purge_thread: threading.Thread | None
    purge_queue: queue.Queue[str | None]
    _storage: CheckpointStorage
    _initialized: bool = False
    """Whether ``__init__`` ran to completion.

    Set last, by each subclass. A subclass that raises partway through leaves
    the object uninitialized, and its ``__del__`` then calls ``close`` -- so
    every public method has to be prepared for attributes that were never
    assigned, and this flag is how it knows. ``enable`` cannot serve: it is
    assigned first, so a manager that failed later still reads as enabled. That
    partial state is not hypothetical: it is exactly what the HF-options
    rejection in ``dcp.CheckpointManager.__init__`` produces.
    """

    _STEP_DIR_PATTERN = r"step-(0|[1-9]\d*)"
    """The canonical checkpoint directory name, e.g. ``step-100``.

    Only non-negative integers without leading zeros match, so a stray
    ``step-007`` or ``step-x`` is ignored by step discovery rather than
    parsed into something that would then collide with ``step-7``.
    """

    # A disabled manager returns early from ``__init__``, and a failed manager
    # raises partway through it, so in neither case do the attributes below
    # exist. Public entry points must check before touching manager state; the
    # overrides keep the check by calling ``super()``.
    #
    # The check must come first in each method -- reading ``self.enable`` on an
    # object that never assigned it raises the very AttributeError it guards
    # against.

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Restore state from ``step``, or the latest checkpoint when ``-1``."""
        if not getattr(self, "_initialized", False) or not self.enable:
            return False

        model_only = False
        from_hf = False
        from_quantized = False

        has_checkpoint_folder = self._storage.isdir(self.folder)
        load_step = -1
        if has_checkpoint_folder:
            load_step = self._find_load_step() if step == -1 else step

        if step != -1 and not has_checkpoint_folder:
            raise FileNotFoundError(
                f"--checkpoint.load_step={step} not found because "
                f"checkpoint.folder {self.folder} does not exist"
            )

        if load_step == -1:
            # Nothing on disk: fall back to whatever initial weights the config
            # names, or to a fresh start.
            model_only = self.initial_load_model_only
            from_hf = self.initial_load_in_hf
            from_quantized = self.initial_load_in_hf_quantized

            if from_hf:
                assert model_only, (
                    "Only model can be loaded when loading from HF's checkpoint."
                )
            if from_quantized:
                assert from_hf, "Quantized checkpoint can only be loaded from HF format"

            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                if not self._storage.isdir(checkpoint_id):
                    raise ValueError(
                        f"Checkpoint.initial_load_path is invalid: {checkpoint_id}"
                    )
                if from_hf:
                    logger.info(
                        "Loading from HF safetensors from "
                        f"--checkpoint.initial_load_path: {checkpoint_id}"
                    )
            elif from_hf:
                assert self.sd_adapter and self.sd_adapter.hf_assets_path, (
                    "from_hf=True requires sd_adapter and hf_assets_path."
                )
                checkpoint_id = self.sd_adapter.hf_assets_path
                if not self._storage.isdir(checkpoint_id):
                    raise ValueError(
                        "model.hf_assets_path is being used to load HF weights "
                        "but the path is not valid. Either make sure hf_assets_path "
                        "is correct or provide a valid checkpoint.initial_load_path"
                    )
                logger.info(
                    f"Loading HF safetensors from --model.hf_assets_path: "
                    f"{checkpoint_id}"
                )
            else:
                logger.info("No checkpoint was provided, this is a fresh start.")
                return False
        else:
            step = load_step
            # Step 0 is a seed checkpoint, which holds model state only.
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)
            if not self._storage.isdir(checkpoint_id):
                raise FileNotFoundError(
                    f"--checkpoint.load_step={step} not found at {checkpoint_id}"
                )

        logger.info("Loading the checkpoint from %s.", checkpoint_id)
        begin = time.monotonic()
        self._load_checkpoint(
            self._states_to_load(model_only),
            checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            "Finished loading the checkpoint in %.2f seconds.",
            time.monotonic() - begin,
        )
        return True

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> bool:
        """Persist state for ``curr_step``."""
        if not getattr(self, "_initialized", False) or not self.enable:
            return False
        return self._save(curr_step, last_step)

    def maybe_wait_for_staging(self) -> None:
        """Block until asynchronous staging for the last save completes."""
        if not getattr(self, "_initialized", False) or not self.enable:
            return
        self._maybe_wait_for_staging()

    def close(self) -> None:
        """Release background threads and other resources.

        Safe to call at any point in the object's life: ``__del__`` routes here,
        and it can run on a partially constructed object whose ``__init__``
        raised. ``_close`` is a no-op when that happened, so implementations of
        it may assume their own attributes exist.
        """
        if not getattr(self, "_initialized", False):
            return
        try:
            self.maybe_wait_for_staging()
            self.maybe_wait_for_saving()
        finally:
            self._close()

    def maybe_wait_for_saving(self) -> None:
        """Block until the last asynchronous save completes.

        A manager with no asynchronous save in flight leaves ``save_future`` at
        ``None`` and never reaches ``_wait_for_saving``.
        """
        if not getattr(self, "_initialized", False) or not self.enable:
            return
        if getattr(self, "save_future", None) is None:
            return
        self._wait_for_saving()

    @abstractmethod
    def _wait_for_saving(self) -> None:
        """Await ``save_future`` and clear it. Only called when it is set."""

    # -- policies shared by every manager --------------------------------------
    # These depend only on config fields, not on how a backend reads or writes
    # bytes, so they live here rather than once per backend.

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        """Whether ``curr_step`` is a checkpointing step."""
        if not self.enable or self.load_only:
            return False
        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True
        return last_step or curr_step % self.interval == 0

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        """Standardized checkpoint path, e.g. ``checkpoints/step-100``."""
        folder = folder or self.folder
        return filesystem.join(folder, f"step-{step}")

    @abstractmethod
    def _load_checkpoint(
        self,
        states: dict[str, Any],
        checkpoint_id: str,
        *,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        """Restore ``states`` from a resolved checkpoint source."""

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        """Select the live state objects that must be restored."""
        if model_only:
            return {MODEL: self.states[MODEL]}

        for exclude_key in self.exclude_from_loading:
            if exclude_key not in self.states:
                raise ValueError(f"{exclude_key} not found in state_dict.")
        return {
            key: value
            for key, value in self.states.items()
            if key not in self.exclude_from_loading
        }

    @abstractmethod
    def _save(self, curr_step: int, last_step: bool = False) -> bool:
        """Implement ``save``. Only called when checkpointing is enabled."""

    @abstractmethod
    def _maybe_wait_for_staging(self) -> None:
        """Implement ``maybe_wait_for_staging``. Only called when enabled."""

    @abstractmethod
    def _close(self) -> None:
        """Implement ``close``. Only called when checkpointing is enabled."""

    def _should_purge(self) -> bool:
        """Whether this rank should purge stale checkpoints.

        Rank 0 only: retention is a global policy, and N ranks racing to delete
        the same directories would at best duplicate work and at worst have one
        delete a directory another is still reading.
        """
        return (
            self.keep_latest_k > 0
            and (
                not dist.is_available()
                or not dist.is_initialized()
                or dist.get_rank() == 0
            )
            and self._storage.isdir(self.folder)
        )

    def _is_purge_exempt(self, step: int) -> bool:
        """Whether the configured exemption protects ``step`` from deletion."""
        return self.purge_exempt is not None and self.purge_exempt(step)

    def _parse_step(self, dirname: str) -> int | None:
        """Parse a canonical ``step-N`` checkpoint directory name."""
        match = re.fullmatch(self._STEP_DIR_PATTERN, dirname)
        return None if match is None else int(match.group(1))

    @abstractmethod
    def _is_valid_checkpoint(self, checkpoint_dir: str) -> bool:
        """Whether ``checkpoint_dir`` holds a completed checkpoint.

        This includes model-only exports that retention must preserve even
        when they cannot restore the full training state.
        """

    @abstractmethod
    def _is_resumable_checkpoint(self, checkpoint_dir: str) -> bool:
        """Whether automatic loading may select ``checkpoint_dir``."""

    def _find_load_step(self, folder: str = "") -> int:
        """The highest step in ``folder`` that can actually be loaded.

        Args:
            folder: directory to scan. Defaults to ``self.folder``.

        Returns:
            The step number, or -1 when the folder holds no loadable checkpoint.

        Note:
            Not remote friendly: one ``listdir`` plus a metadata probe per step
            folder, each a network round trip on remote (fsspec) storage rather
            than a single batched listing. Acceptable because it runs once, at
            load time.
        """
        folder = folder or self.folder
        if not self._storage.isdir(folder):
            return -1

        resumable_steps = []
        for dirname in self._storage.listdir(folder):
            step = self._parse_step(dirname)
            if step is None:
                continue
            if self._is_resumable_checkpoint(filesystem.join(folder, dirname)):
                resumable_steps.append(step)
        return max(resumable_steps) if resumable_steps else -1

    def _purge_stale_checkpoints(
        self,
        *,
        saving_step: int,
        staging_dir_prefix: str | None = None,
    ) -> None:
        """Delete abandoned entries, and reserve one retained slot for this save.

        Two kinds of entry are collected:

        * **abandoned** -- a directory matching the step pattern that is not a
          valid checkpoint (a save that died midway), or a staging directory
          left behind by an async writer. Both are deleted outright.
        * **complete** -- a valid checkpoint past the retention horizon, deleted
          only if ``purge_exempt`` does not protect its step.

        ``keep_latest_k - 1`` rather than ``keep_latest_k`` is the retention
        count because the save this call precedes has not landed yet: the slot
        it is about to fill is reserved here, so the horizon is computed as if
        it were already on disk.
        """
        if self._should_purge():
            saving_dirnames = {f"step-{saving_step}"}
            if staging_dir_prefix:
                saving_dirnames.add(f"{staging_dir_prefix}step-{saving_step}")

            staging_pattern = (
                re.compile(rf"{re.escape(staging_dir_prefix)}step-(0|[1-9]\d*)")
                if staging_dir_prefix
                else None
            )
            checkpoints: list[tuple[int, str]] = []
            abandoned: list[str] = []

            for dirname in self._storage.listdir(self.folder):
                if dirname in saving_dirnames:
                    continue

                checkpoint_dir = filesystem.join(self.folder, dirname)
                # torch_checkpointing uses this pattern for staging directories.
                if staging_pattern and staging_pattern.fullmatch(dirname):
                    abandoned.append(checkpoint_dir)
                    continue

                step = self._parse_step(dirname)
                if step is None:
                    continue
                if self._is_valid_checkpoint(checkpoint_dir):
                    checkpoints.append((step, checkpoint_dir))
                else:
                    abandoned.append(checkpoint_dir)

            checkpoints.sort()
            num_to_keep = self.keep_latest_k - 1
            num_to_purge = max(0, len(checkpoints) - num_to_keep)
            for step, checkpoint_dir in checkpoints[:num_to_purge]:
                if self._is_purge_exempt(step):
                    logger.info(
                        "Checkpointer is preserving checkpoint %s outside "
                        "keep_latest_k.",
                        checkpoint_dir,
                    )
                    continue
                assert self.purge_thread is not None
                self.purge_queue.put(checkpoint_dir)

            for checkpoint_dir in abandoned:
                assert self.purge_thread is not None
                self.purge_queue.put(checkpoint_dir)
