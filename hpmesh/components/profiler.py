"""Kineto traces and allocator memory snapshots, over one training run.

Vendored from torchtitan's ``observability/profiler.py``. The shape is kept: a
``Profiler`` that is entered once around the training loop and stepped once per
iteration, owning a torch profiler for traces and a ``MemoryProfiler`` for
periodic snapshots, plus the OOM path that forces a final snapshot before the
process dies.

The memory side is the part worth having even when traces are off: a run that
OOMs at step 400 leaves no evidence of what the allocator was holding, and the
exit snapshot is what answers that.

Departures from torchtitan, each a subtraction or a device-portability fix:

* **No ``structured_logger``.** The ``log_trace_span`` wrappers around
  ``profiler.step()`` go with it, which hpmesh does not have; a profiler step is
  already the finest thing the trace shows.

* **No ``active()`` builder.** torchtitan splits ``build`` (returns a configured
  profiler) from ``active`` (supplies the runtime step and folder), because a
  config there has to be buildable with no arguments. hpmesh constructs the
  ``Profiler`` where the trainer knows both, so they collapse into the
  constructor.

* **No CUDA-graph annotations.** Those annotate a trace with the module FQNs
  captured inside ``torch.cuda.graph`` regions. hpmesh does not capture CUDA
  graphs, so ``get_cudagraph_annotations`` could only ever return an empty dict
  and the version probe around ``export_chrome_trace`` guards a branch that
  cannot be taken.

* **No XPU activity.** The device is resolved once in ``accelerator/device.py`` as
  ``cuda`` or ``cpu``; there is no third case to ask about.

* **Memory history is recorded through the device module, not ``torch``.** See
  ``utils/monitoring.record_memory_history`` -- torchtitan's non-CUDA branch
  calls ``torch.memory``, which does not exist.

One addition: the rank comes from ``utils/logger_utils``, so a single-process
run has a rank without a process group, the same way metrics does it.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..trainer.config import ProfilerConfig

from ..accelerator.device import device_type
from ..accelerator.monitoring import read_memory_snapshot, record_memory_history
from ..utils.logger_utils import get_distributed_rank, get_logger

logger = get_logger(__name__)

__all__ = ["MemoryProfiler", "Profiler"]

# Directory layout, kept as plain format strings rather than derived at the
# point of use so the trace/snapshot tree is legible in one place.
PROFILE_DIR = "profiling/traces"  # the ProfilerConfig default
PROFILE_ITER_DIR = "iteration_{step}"  # PROFILE_DIR/{PROFILE_ITER_DIR}
PROFILE_FILE = "rank{rank}_trace.json.gz"  # .../{PROFILE_FILE}

MEMORY_DIR = "profiling/memory_snapshot"  # the ProfilerConfig default
MEMORY_STEP_DIR = "step_{step:012d}"  # MEMORY_DIR/{MEMORY_STEP_DIR}
MEMORY_EXIT_DIR = "step_{step:012d}_exit"  # the OOM variant of the same
MEMORY_FILE = "{rank:06d}_step_{step}.pickle"

# A snapshot needs only the Python frames to be useful, and protocol 4 is what
# pytorch.org/memory_viz reads. See MemoryProfiler.step.
_SNAPSHOT_PICKLE_PROTOCOL = 4


class MemoryProfiler:
    """Writes periodic allocator snapshots while training runs.

    Built by :meth:`Profiler.build_memory_profiler` when snapshots are enabled.
    Call :meth:`step` once per iteration; the frequency is checked here rather
    than by the caller, so the loop does not have to know the policy.
    """

    def __init__(
        self,
        step_num: int,
        freq: int,
        snapshot_dir: str,
        leaf_folder: str,
        rank: int,
        max_entries: int,
    ) -> None:
        self._records_history = record_memory_history(max_entries=max_entries)
        if not self._records_history:
            # Not an error: the device has no allocator history to keep. Said
            # once, at the point the run asked for snapshots, rather than
            # silently writing empty files for the rest of the run.
            logger.warning(
                "Memory snapshots are enabled but the %s allocator keeps no "
                "memory history; no snapshots will be written.",
                device_type,
            )
        self.step_num = step_num
        self.freq = freq
        self._snapshot_dir = snapshot_dir
        self._leaf_folder = leaf_folder
        self._rank = rank

    def step(self, *, exit_ctx: bool = False) -> None:
        """Write a snapshot if this iteration calls for one.

        ``exit_ctx`` forces one regardless of the frequency, and names it for
        the *previous* step: the snapshot is taken while unwinding an exception,
        so the step counter has not advanced past the one that failed.

        On a device whose allocator keeps no history (CPU) there is nothing to
        write: returning here is what keeps a pickled ``None`` out of the
        snapshot directory on the OOM path, where the frequency check below is
        deliberately bypassed. Upstream writes the file unconditionally.
        """
        self.step_num += 1
        if not self._records_history:
            return
        if not exit_ctx and self.step_num % self.freq != 0:
            return

        if exit_ctx:
            curr_step = self.step_num - 1
            dir_name = MEMORY_EXIT_DIR.format(step=curr_step)
        else:
            curr_step = self.step_num
            dir_name = MEMORY_STEP_DIR.format(step=curr_step)

        directory = os.path.join(self._snapshot_dir, dir_name, self._leaf_folder)
        os.makedirs(directory, exist_ok=True)

        logger.info("Dumping memory snapshot at step %d", curr_step)
        begin = time.monotonic()
        output_file = os.path.join(
            directory, MEMORY_FILE.format(rank=self._rank, step=curr_step)
        )
        with open(output_file, "wb") as output:
            # Protocol 4 rather than the interpreter's default: memory_viz's JS
            # parser reads that one.
            pickle.dump(
                read_memory_snapshot(), output, protocol=_SNAPSHOT_PICKLE_PROTOCOL
            )
        logger.info(
            "Finished dumping memory snapshot in %.2f seconds", time.monotonic() - begin
        )


class Profiler:
    """Owns the trace and memory-snapshot lifecycle for a training run.

    Example::

        with Profiler(config, global_step=step, base_folder=folder) as prof:
            for step in training_loop:
                ...
                prof.step()

    Args:
        config: a :class:`ProfilerConfig` instance.
        global_step: the step profiling begins at. When resuming from a
            checkpoint this is the loaded step, so trace directories are named
            for where the run actually is (``iteration_100``, not
            ``iteration_0``) and the snapshot frequency stays aligned with it.
        base_folder: root directory for both trace and snapshot output.
    """

    def __init__(
        self,
        config: ProfilerConfig,
        *,
        global_step: int = 0,
        base_folder: str = "",
    ) -> None:
        self._config = config
        self._global_step = global_step
        self._base_folder = base_folder
        self.torch_profiler = None
        self.memory_profiler = None

    def __enter__(self) -> Profiler:
        self.torch_profiler = self.build_torch_profiler()
        self.memory_profiler = self.build_memory_profiler()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.torch_profiler is not None:
            self.torch_profiler.__exit__(exc_type, exc_val, exc_tb)
            self.torch_profiler = None
        if self.memory_profiler is not None:
            if _caused_by_oom(exc_val):
                # The one snapshot that matters most, and the only chance to
                # take it: the allocator still holds the state that failed.
                self.memory_profiler.step(exit_ctx=True)
            self.memory_profiler = None
        return False

    def step(self) -> None:
        """Advance every active profiler by one training iteration."""
        if self.torch_profiler is not None:
            self.torch_profiler.step()
        if self.memory_profiler is not None:
            self.memory_profiler.step()

    def build_torch_profiler(self):
        """Create, start, and return the torch profiler, or ``None`` if off.

        Entered here rather than by the caller so the returned handle is
        already collecting; :meth:`__exit__` is what stops it.
        """
        cfg = self._config
        if not cfg.enable_profiling:
            return None

        trace_dir = os.path.join(self._base_folder, cfg.save_traces_folder)
        rank = get_distributed_rank()

        def trace_handler(prof):
            directory = os.path.join(
                trace_dir,
                PROFILE_ITER_DIR.format(step=prof.step_num),
            )
            os.makedirs(directory, exist_ok=True)

            logger.info("Dumping profiler traces at step %d", prof.step_num)
            begin = time.monotonic()
            prof.export_chrome_trace(
                os.path.join(directory, PROFILE_FILE.format(rank=rank))
            )
            logger.info(
                "Finished dumping profiler traces in %.2f seconds",
                time.monotonic() - begin,
            )

        logger.info("Profiling active. Traces will be saved at %s", trace_dir)
        os.makedirs(trace_dir, exist_ok=True)

        # Left out of the schedule rather than passed as None, because
        # ``torch.profiler.schedule`` treats an explicit None differently from
        # an absent argument.
        optional_params = {
            key: value
            for key, value in [
                ("repeat", cfg.profiler_repeat),
                ("skip_first", cfg.profiler_skip_first),
                ("skip_first_wait", cfg.profiler_skip_first_wait),
            ]
            if value is not None
        }

        # What remains of the cycle after warmup and active. The config's
        # validation guarantees this is not negative.
        wait = cfg.profile_freq - (cfg.profiler_active + cfg.profiler_warmup)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device_type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        torch_profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=wait,
                warmup=cfg.profiler_warmup,
                active=cfg.profiler_active,
                **optional_params,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
        )
        torch_profiler.__enter__()
        # The schedule counts from zero. Starting it at the resumed step is what
        # makes the trace directories line up with the run's step numbers.
        torch_profiler.step_num = self._global_step
        return torch_profiler

    def build_memory_profiler(self):
        """Create a :class:`MemoryProfiler`, or ``None`` if snapshots are off.

        Its constructor starts recording immediately; that is why this is built
        on entry rather than at the first dump.
        """
        cfg = self._config
        if not cfg.enable_memory_snapshot:
            return None

        freq = (
            cfg.profile_freq
            if cfg.memory_snapshot_freq is None
            else cfg.memory_snapshot_freq
        )
        if freq <= 0:
            raise ValueError(
                "Memory snapshot frequency must be greater than zero; set "
                "profiler.memory_snapshot_freq or profiler.profile_freq to a "
                "positive value."
            )

        snapshot_dir = os.path.join(self._base_folder, cfg.save_memory_snapshot_folder)
        os.makedirs(snapshot_dir, exist_ok=True)
        rank = get_distributed_rank()

        logger.info(
            "Memory profiler active. Snapshots will be saved at %s", snapshot_dir
        )
        return MemoryProfiler(
            self._global_step,
            freq,
            snapshot_dir,
            "",
            rank,
            cfg.memory_snapshot_max_entries,
        )


def _caused_by_oom(exc: BaseException | None) -> bool:
    """Whether an ``OutOfMemoryError`` appears anywhere in the cause chain.

    The chain matters rather than just the exception type: pipeline parallelism
    does not re-raise the OOM it catches, it wraps the failure in a plain
    ``RuntimeError`` to attach the stage's shapes, which hides the type.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, torch.OutOfMemoryError):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False
