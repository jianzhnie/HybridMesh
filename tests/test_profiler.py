# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The profiler's directory layout, frequency policy, and OOM handling.

The traces and snapshots themselves need a GPU to be interesting, but almost
none of the code here is about the device: it decides *when* to dump, *where*
to put the file, and whether a failure was an OOM. Those are the parts that go
wrong quietly -- a snapshot written to the wrong step, or an OOM whose wrapper
exception hides the type -- so they are tested on CPU, where the module has to
degrade gracefully anyway.
"""

from __future__ import annotations

import os
import pickle

import pytest
import torch

from hpmesh.components import profiler as profiler_module
from hpmesh.components.profiler import (
    MEMORY_EXIT_DIR,
    MEMORY_STEP_DIR,
    PROFILE_ITER_DIR,
    MemoryProfiler,
    Profiler,
    _caused_by_oom,
)
from hpmesh.utils.monitoring import record_memory_history

Config = Profiler.Config


# -- config -------------------------------------------------------------------


def test_the_defaults_disable_everything() -> None:
    """Profiling is opt-in: it changes what the run does and costs real time."""
    config = Config()

    assert not config.enable_profiling
    assert not config.enable_memory_snapshot


def test_a_cycle_that_cannot_fit_its_parts_is_rejected() -> None:
    """The schedule's wait would go negative. torch accepts a negative wait and
    then never activates, producing a run that profiles nothing."""
    with pytest.raises(ValueError, match="warmup \\+ profiler_active"):
        Config(enable_profiling=True, profile_freq=2, profiler_warmup=3)


def test_the_cycle_length_only_has_to_fit_when_profiling_is_on() -> None:
    """The values are still validated -- they are just not enforced when the
    schedule they configure is never built."""
    config = Config(profile_freq=1, profiler_warmup=10)

    assert config.profile_freq == 1


def test_a_zero_snapshot_frequency_is_rejected_at_build_time(tmp_path) -> None:
    profiler = Profiler(
        Config(enable_memory_snapshot=True, memory_snapshot_freq=0),
        base_folder=str(tmp_path),
    )

    with pytest.raises(ValueError, match="greater than zero"):
        profiler.build_memory_profiler()


def test_the_snapshot_frequency_falls_back_to_the_trace_frequency(tmp_path) -> None:
    profiler = Profiler(
        Config(enable_memory_snapshot=True, profile_freq=7, memory_snapshot_freq=None),
        base_folder=str(tmp_path),
    )

    assert profiler.build_memory_profiler().freq == 7


def test_an_explicit_snapshot_frequency_wins(tmp_path) -> None:
    profiler = Profiler(
        Config(enable_memory_snapshot=True, profile_freq=7, memory_snapshot_freq=2),
        base_folder=str(tmp_path),
    )

    assert profiler.build_memory_profiler().freq == 2


# -- lifecycle ----------------------------------------------------------------


def test_nothing_is_built_when_nothing_is_enabled(tmp_path) -> None:
    """Every entry point still has to work: the trainer enters and steps the
    profiler unconditionally."""
    with Profiler(Config(), base_folder=str(tmp_path)) as profiler:
        assert profiler.torch_profiler is None
        assert profiler.memory_profiler is None
        profiler.step()


def test_exit_clears_both_handles(tmp_path) -> None:
    """A second __exit__ must be a no-op rather than stopping a profiler that
    is no longer running."""
    profiler = Profiler(Config(), base_folder=str(tmp_path))
    profiler.__enter__()

    profiler.__exit__(None, None, None)

    assert profiler.torch_profiler is None
    assert profiler.memory_profiler is None


def test_exit_does_not_swallow_the_exception(tmp_path) -> None:
    """The profiler observes the failure; the loop still has to see it."""
    with Profiler(Config(), base_folder=str(tmp_path)):
        pass
    profiler = Profiler(Config(), base_folder=str(tmp_path))
    profiler.__enter__()

    assert profiler.__exit__(ValueError, ValueError("boom"), None) is False


# -- OOM detection ------------------------------------------------------------


def test_an_oom_is_detected_directly() -> None:
    assert _caused_by_oom(torch.OutOfMemoryError("CUDA out of memory"))


def test_an_oom_wrapped_in_another_error_is_still_detected() -> None:
    """Pipeline parallelism re-raises the OOM it catches as a plain
    RuntimeError to attach the stage's shapes, so the type alone is not enough
    and the snapshot that matters most would be skipped."""
    try:
        try:
            raise torch.OutOfMemoryError("CUDA out of memory")
        except torch.OutOfMemoryError as oom:
            raise RuntimeError("Failure at stage 2") from oom
    except RuntimeError as wrapped:
        assert _caused_by_oom(wrapped)


def test_an_oom_reached_through_implicit_context_is_detected() -> None:
    """A bare ``raise`` inside an except block chains via __context__, not
    __cause__."""
    try:
        try:
            raise torch.OutOfMemoryError("CUDA out of memory")
        except torch.OutOfMemoryError:
            # Deliberately without ``from``: that is the chaining shape under
            # test, so the re-raise lint does not apply.
            raise RuntimeError("wrapped without from")  # noqa: B904
    except RuntimeError as wrapped:
        assert wrapped.__cause__ is None
        assert _caused_by_oom(wrapped)


def test_an_unrelated_failure_is_not_an_oom() -> None:
    assert not _caused_by_oom(ValueError("just a bug"))
    assert not _caused_by_oom(RuntimeError("Failure at stage 2"))
    assert not _caused_by_oom(None)


def test_a_self_referential_chain_terminates() -> None:
    """The walk keeps a seen-set; without it a cycle would hang the exit path,
    which is the one moment the process cannot afford to hang."""
    error = RuntimeError("cycled")
    error.__cause__ = error

    assert not _caused_by_oom(error)


def test_the_exit_snapshot_is_forced_on_an_oom(tmp_path) -> None:
    """The frequency must not gate it: this is the only chance to capture the
    allocator state that failed."""
    profiler = Profiler(
        Config(enable_memory_snapshot=True, memory_snapshot_freq=1000),
        base_folder=str(tmp_path),
    )
    profiler.__enter__()
    recorder = _RecordingMemoryProfiler(freq=1000)
    profiler.memory_profiler = recorder

    profiler.__exit__(torch.OutOfMemoryError, torch.OutOfMemoryError("oom"), None)

    assert recorder.calls == [True]


class _RecordingMemoryProfiler(MemoryProfiler):
    """A MemoryProfiler that records step() calls instead of writing files."""

    def __init__(self, freq: int) -> None:
        self.step_num = 0
        self.freq = freq
        self._records_history = True
        self.calls: list[bool] = []

    def step(self, *, exit_ctx: bool = False) -> None:
        self.calls.append(exit_ctx)


# A zero warmup is what makes a trace fire on the very first cycle, which is
# what these tests need to stay fast. torch warns because warmup is what keeps
# the first active iteration from being dominated by lazy initialization.
_NO_WARMUP = pytest.mark.filterwarnings("ignore:Profiler won't be using warmup")


@pytest.fixture
def writable_snapshots(monkeypatch):
    """Report a device that keeps allocator history, so the write path runs.

    CPU has no history to record, so without this every snapshot test would
    skip. The two functions patched here are the whole device dependency of
    ``MemoryProfiler``; the naming, frequency and file format around them are
    what these tests are for.
    """
    monkeypatch.setattr(profiler_module, "record_memory_history", lambda **_: True)
    monkeypatch.setattr(profiler_module, "read_memory_snapshot", lambda: {"fake": 1})


def test_exit_takes_no_forced_snapshot_on_an_ordinary_failure(tmp_path) -> None:
    profiler = Profiler(
        Config(enable_memory_snapshot=True, memory_snapshot_freq=1000),
        base_folder=str(tmp_path),
    )
    profiler.__enter__()
    recorder = _RecordingMemoryProfiler(freq=1000)
    profiler.memory_profiler = recorder

    profiler.__exit__(ValueError, ValueError("just a bug"), None)

    assert recorder.calls == []


# -- the snapshot frequency policy -------------------------------------------


def _snapshot_dirs(root: str) -> list[str]:
    """The step directories the profiler wrote, in order."""
    found: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        if filenames:
            found.append(os.path.basename(dirpath))
    return sorted(found)


def test_snapshots_are_written_on_the_frequency(tmp_path, writable_snapshots) -> None:
    profiler = MemoryProfiler(
        step_num=0,
        freq=2,
        snapshot_dir=str(tmp_path),
        leaf_folder="",
        rank=0,
        max_entries=100,
    )

    for _ in range(4):
        profiler.step()

    assert _snapshot_dirs(str(tmp_path)) == [
        MEMORY_STEP_DIR.format(step=2),
        MEMORY_STEP_DIR.format(step=4),
    ]


def test_the_exit_snapshot_names_the_step_that_failed(
    tmp_path, writable_snapshots
) -> None:
    """step_num is incremented before the check, so at the moment of failure
    the counter has already moved past the failing step."""
    profiler = MemoryProfiler(
        step_num=10,
        freq=1000,
        snapshot_dir=str(tmp_path),
        leaf_folder="",
        rank=0,
        max_entries=100,
    )

    profiler.step(exit_ctx=True)

    assert _snapshot_dirs(str(tmp_path)) == [MEMORY_EXIT_DIR.format(step=10)]


def test_the_snapshot_file_is_named_for_the_rank_and_step(
    tmp_path, writable_snapshots
) -> None:
    profiler = MemoryProfiler(
        step_num=0,
        freq=1,
        snapshot_dir=str(tmp_path),
        leaf_folder="",
        rank=3,
        max_entries=100,
    )

    profiler.step()

    written = os.path.join(
        str(tmp_path), MEMORY_STEP_DIR.format(step=1), "000003_step_1.pickle"
    )
    assert os.path.exists(written)


def test_the_snapshot_uses_protocol_4(tmp_path, writable_snapshots) -> None:
    """memory_viz's JS parser reads protocol 4. The interpreter's default is
    higher, and a snapshot it cannot read is the same as no snapshot."""
    profiler = MemoryProfiler(
        step_num=0,
        freq=1,
        snapshot_dir=str(tmp_path),
        leaf_folder="",
        rank=0,
        max_entries=100,
    )

    profiler.step()

    written = os.path.join(
        str(tmp_path), MEMORY_STEP_DIR.format(step=1), "000000_step_1.pickle"
    )
    with open(written, "rb") as handle:
        # The first bytes are the protocol magic: \x80 <protocol>.
        assert handle.read(2) == b"\x80\x04"
        assert pickle.load(handle) == {"fake": 1}


def test_a_worker_directory_is_nested_under_the_step(
    tmp_path, writable_snapshots
) -> None:
    """The leaf folder is how a fault-tolerant replica keeps its snapshots
    apart from its siblings'."""
    profiler = MemoryProfiler(
        step_num=0,
        freq=1,
        snapshot_dir=str(tmp_path),
        leaf_folder="worker_1",
        rank=0,
        max_entries=100,
    )

    profiler.step()

    written = os.path.join(
        str(tmp_path),
        MEMORY_STEP_DIR.format(step=1),
        "worker_1",
        "000000_step_1.pickle",
    )
    assert os.path.exists(written)


# -- the device probes underneath --------------------------------------------


def test_a_cpu_run_reports_that_it_cannot_record_history() -> None:
    """torchtitan's non-CUDA branch calls ``torch.memory``, which is not a real
    module -- so this raises AttributeError there rather than returning False."""
    started = record_memory_history(max_entries=100)

    assert started == torch.cuda.is_available()


def test_a_memory_profiler_off_cuda_warns_and_writes_nothing(tmp_path, caplog) -> None:
    """Said once, when the run asked for snapshots, rather than silently
    producing empty files for the rest of it."""
    profiler = MemoryProfiler(
        step_num=0,
        freq=1,
        snapshot_dir=str(tmp_path),
        leaf_folder="",
        rank=0,
        max_entries=100,
    )

    for _ in range(3):
        profiler.step()

    if torch.cuda.is_available():
        assert profiler._records_history
    else:
        assert not profiler._records_history
        assert not _snapshot_dirs(str(tmp_path))
        assert any("no memory history" in r.message for r in caplog.records)


def test_the_trace_directory_is_named_for_the_step(tmp_path) -> None:
    """With the schedule disabled it never fires, so the naming rule is checked
    directly against the format the handler uses."""
    assert PROFILE_ITER_DIR.format(step=42) == "iteration_42"


# -- trace collection ---------------------------------------------------------


@_NO_WARMUP
def test_a_trace_is_written_at_the_end_of_the_cycle(tmp_path) -> None:
    """A one-iteration cycle with no warmup, so the run below produces exactly
    one trace directory."""
    profiler = Profiler(
        Config(
            enable_profiling=True,
            profile_freq=1,
            profiler_warmup=0,
            profiler_active=1,
        ),
        global_step=0,
        base_folder=str(tmp_path),
    )

    with profiler:
        torch.mm(torch.randn(16, 16), torch.randn(16, 16))
        profiler.step()

    assert os.path.exists(
        os.path.join(str(tmp_path), "profiling", "traces", "iteration_1")
    )


@_NO_WARMUP
def test_the_schedule_restarts_from_the_resumed_step(tmp_path) -> None:
    """A resumed run must name its traces for where it actually is. Starting
    the schedule at zero would label every trace of a resumed run from
    ``iteration_1``, which collides with the pre-checkpoint ones."""
    profiler = Profiler(
        Config(enable_profiling=True, profile_freq=1, profiler_warmup=0),
        global_step=100,
        base_folder=str(tmp_path),
    )

    with profiler as entered:
        assert entered.torch_profiler.step_num == 100


def test_profiling_is_off_by_default_and_builds_nothing(tmp_path) -> None:
    profiler = Profiler(Config(), base_folder=str(tmp_path))

    assert profiler.build_torch_profiler() is None


# -- the exported format ------------------------------------------------------


@_NO_WARMUP
def test_the_trace_is_gzipped_json(tmp_path) -> None:
    """The extension is what the trace viewer dispatches on."""
    profiler = Profiler(
        Config(enable_profiling=True, profile_freq=1, profiler_warmup=0),
        global_step=0,
        base_folder=str(tmp_path),
    )

    with profiler:
        torch.mm(torch.randn(16, 16), torch.randn(16, 16))
        profiler.step()

    written = os.path.join(
        str(tmp_path), "profiling", "traces", "iteration_1", "rank0_trace.json.gz"
    )
    assert os.path.exists(written)
    with open(written, "rb") as handle:
        # gzip magic bytes.
        assert handle.read(2) == b"\x1f\x8b"


def test_pickling_a_none_snapshot_is_still_a_valid_file(tmp_path) -> None:
    """On a device that reports no history the guard stops the write, but if
    one ever slipped through, the file has to at least be readable back."""
    written = tmp_path / "snapshot.pickle"
    with open(written, "wb") as handle:
        pickle.dump(None, handle, protocol=4)

    with open(written, "rb") as handle:
        assert pickle.load(handle) is None
