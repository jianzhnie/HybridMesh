"""Checkpointing config."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from hpmesh.utils import filesystem
from hpmesh.utils.checkpoint_keys import LR_SCHEDULER, MODEL, OPTIMIZER
from hpmesh.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass(kw_only=True)
class CheckpointConfig:
    """What the checkpoint manager keeps, where, and when.

    ``HfArgumentParser`` cannot nest a dataclass behind a named flag -- a nested
    field surfaces as one opaque ``--checkpoint CHECKPOINT`` string -- so this is
    a flat dataclass with every field becoming its own flag (``--enable``,
    ``--interval``, ...). Its ``__post_init__`` therefore runs on the parsed
    values, which is where the cross-field checks belong.

    No ``slots=True``, deliberately: the re-created class breaks the zero-arg
    ``super()`` cell in a subclass's ``__post_init__``. ``ParallelConfig``
    omits it for the same reason.
    """

    enable: bool = False
    """Whether to enable checkpointing."""

    folder: str = "checkpoint"
    """Checkpoint folder, relative to the trainer dump folder."""

    interval: int = 500
    """Checkpointing interval in steps."""

    initial_load_path: str | None = None
    """Optional checkpoint path used when the output checkpoint folder is empty."""

    initial_load_model_only: bool = True
    """Whether an initial checkpoint restores only model state.

    Only consulted on the initial-load path, i.e. when ``initial_load_path``
    names a checkpoint; with no initial checkpoint there is nothing to load
    either way.
    """

    initial_load_in_hf: bool = False
    """Whether the initial checkpoint uses Hugging Face safetensors."""

    initial_load_in_hf_quantized: bool = False
    """Whether the initial Hugging Face checkpoint uses quantized keys."""

    last_save_model_only: bool = True
    """Whether the final checkpoint contains only model state."""

    last_save_in_hf: bool = False
    """Whether the final model-only checkpoint uses Hugging Face safetensors."""

    export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    """Model dtype used by a final model-only checkpoint."""

    keep_latest_k: int = 10
    """Number of recent checkpoints to retain, or zero to retain all."""

    purge_exempt: Callable[[int], bool] | None = None
    """Optional predicate that exempts checkpoint steps from purging."""

    load_step: int = -1
    """Load the checkpoint at the specified step. If -1, load the latest one."""

    exclude_from_loading: list[str] = field(default_factory=list)
    """Non-model state keys excluded from loading."""

    enable_first_step_checkpoint: bool = False
    """Whether to save immediately after the first training step."""

    create_seed_checkpoint: bool = False
    """Whether to initialize and save an unsharded seed checkpoint."""

    load_only: bool = False
    """Whether to permit loads while disabling all saves."""

    async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
    """DCP save mode: synchronous, threaded async, or pinned-memory async.

    Only the DCP backend reads it; the torch_checkpointing one saves
    synchronously. Kept here anyway because it is a checkpointing policy like
    every other field on this class, and splitting one field off into a
    subclass per backend is what this file's single-config design avoids.
    """

    def __post_init__(self) -> None:
        if not self.folder.strip():
            raise ValueError("The 'folder' field cannot be empty.")
        if self.interval < 1:
            raise ValueError("Checkpoint interval needs to be at least 1 step.")
        if self.load_step < -1:
            raise ValueError("load_step must be -1 or non-negative.")
        if self.keep_latest_k < 0:
            raise ValueError("keep_latest_k cannot be negative.")
        if self.keep_latest_k == 1:
            raise ValueError(
                "We need to maintain at least 2 checkpoint replicas, "
                "as the last one may be in the process of being saved."
            )
        if MODEL in self.exclude_from_loading:
            raise ValueError(f"{MODEL} key shouldn't be in exclude_from_loading.")
        if (
            OPTIMIZER in self.exclude_from_loading
            and LR_SCHEDULER not in self.exclude_from_loading
        ):
            raise ValueError(
                f"{LR_SCHEDULER} must be excluded when {OPTIMIZER} is excluded."
            )

        if self.initial_load_path:
            self.initial_load_path = self.initial_load_path.strip()
            if not (
                self.initial_load_path.startswith("/")
                or filesystem.is_remote(self.initial_load_path)
            ):
                raise ValueError(
                    "initial_load_path must be an absolute path or a remote "
                    f"URI (e.g. gs://...): {self.initial_load_path}"
                )
        if self.initial_load_in_hf and not self.initial_load_model_only:
            raise ValueError("initial_load_in_hf requires initial_load_model_only.")
        if self.initial_load_in_hf_quantized and not (
            self.initial_load_in_hf and self.initial_load_path
        ):
            raise ValueError(
                "initial_load_in_hf_quantized requires initial_load_in_hf "
                "and initial_load_path."
            )
        if self.last_save_in_hf and not self.last_save_model_only:
            raise ValueError("last_save_in_hf requires last_save_model_only=True.")

        async_lowered = self.async_mode.lower()
        if async_lowered not in ("disabled", "async", "async_with_pinned_mem"):
            raise ValueError(f"Invalid async_mode: {async_lowered}")
        self.async_mode = async_lowered

        # Remote (fsspec) checkpoint IO supports only the native DCP format. HF
        # safetensors read/write to a remote URI is not implemented, so reject
        # the combination up front instead of failing deep inside DCP.
        if self.last_save_in_hf and filesystem.is_remote(self.folder):
            raise ValueError(
                "last_save_in_hf is not supported with a remote "
                f"checkpoint.folder: {self.folder}"
            )
        if (
            self.initial_load_in_hf
            and self.initial_load_path
            and filesystem.is_remote(self.initial_load_path)
        ):
            raise ValueError(
                "initial_load_in_hf is not supported with a remote "
                f"initial_load_path: {self.initial_load_path}"
            )

        if self.load_only and self.enable_first_step_checkpoint:
            logger.warning(
                "checkpoint.load_only is True; enable_first_step_checkpoint "
                "will be ignored."
            )
        # Note torchtitan's sibling warning for ``initial_load_model_only``
        # without an ``initial_load_path`` is deliberately not ported: hpmesh
        # builds a default Config on every run (``--help`` included), so that
        # warning would fire on runs that never load anything.
