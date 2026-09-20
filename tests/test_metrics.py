# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The metrics surface: derivations, sinks, and the probes underneath.

Worth testing outside a training run because these are the numbers that are
*not* checked by the loss curve. A throughput denominator, an MFU denominator,
and a tag applied twice all produce output that looks fine and is wrong, and no
loss comparison would notice. Everything here runs on CPU with no process group.
"""

from __future__ import annotations

import logging

import pytest

from hpmesh.components import metrics as metrics_module
from hpmesh.components.metrics import (
    BaseLogger,
    DeviceMemoryMonitor,
    LoggerContainer,
    MetricsProcessor,
    TensorBoardLogger,
    WandBLogger,
    ensure_pp_loss_visible,
    get_metrics_rank,
)
from hpmesh.models.hf_wrapper import num_flops_per_token
from hpmesh.trainer.config import HybridMeshConfig, ModelArguments, TrainingArguments
from hpmesh.utils.monitoring import Color, NoColor, colors_enabled, get_peak_flops

Config = MetricsProcessor.Config


class _RecordingLogger(BaseLogger):
    """A sink that keeps what it was given, so a test can assert on it."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, int]] = []
        self.closed = False

    def log(self, metrics: dict, step: int) -> None:
        # Copied: the processor does not reuse the dict today, and a test that
        # held a live reference would silently depend on that.
        self.calls.append((dict(metrics), step))

    def close(self) -> None:
        self.closed = True


class _TaggingLogger(_RecordingLogger):
    """The prefixing behaviour that TensorBoardLogger and WandBLogger share.

    Calling the real ones would drag in a run directory or a live W&B run to
    test a string operation both do identically -- see the two tests at the end
    of the sinks section for those.
    """

    def __init__(self, tag: str | None) -> None:
        super().__init__()
        self.tag = tag

    def log(self, metrics: dict, step: int) -> None:
        prefixed = {
            key if self.tag is None else f"{self.tag}/{key}": value
            for key, value in metrics.items()
        }
        super().log(prefixed, step)


class _FakeParallelDims:
    """The subset of ParallelDims that the metrics rank logic reads."""

    def __init__(self, *, pp: int = 1, world_size: int = 1, non_data_parallel: int = 1):
        self.pp = pp
        self.world_size = world_size
        self.non_data_parallel_size = non_data_parallel

    @property
    def pp_enabled(self) -> bool:
        return self.pp > 1


def _processor(**config_overrides) -> MetricsProcessor:
    return MetricsProcessor(
        Config(**config_overrides), parallel_dims=None, num_flops_per_token=1000
    )


def _derive(processor: MetricsProcessor, step: int = 1):
    """``_derive`` with the anchor that ``log`` would have established.

    The anchor is ``log``'s job, so a direct call to the derivation has to
    stand in for it -- or ``log``'s ordering guarantee is the thing under test.
    """
    if processor.step_last_log is None:
        processor.step_last_log = step - 1
    return processor._derive(step)


def _capturing():
    """Attach a recording handler to the module's logger and collect records.

    The pipeline-parallel warnings go through ``logger.warning``, not
    ``warnings.warn``, so ``pytest.warns`` cannot see them.
    """
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler()
    metrics_module.logger.addHandler(handler)
    return _LogCapture(records, handler)


class _LogCapture:
    def __init__(self, records, handler) -> None:
        self.records = records
        self.handler = handler

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        metrics_module.logger.removeHandler(self.handler)
        return False

    def messages(self) -> list[str]:
        return [record.getMessage() for record in self.records]


# -- config -------------------------------------------------------------------


def test_config_rejects_a_non_positive_log_freq() -> None:
    for log_freq in (0, -1):
        with pytest.raises(ValueError, match="greater than 0"):
            Config(log_freq=log_freq)


# -- should_log ---------------------------------------------------------------


def test_should_log_fires_on_the_first_step_and_then_on_the_interval() -> None:
    """Step 1 always reports: a long run whose first line never arrives is
    indistinguishable from a hang."""
    processor = _processor(log_freq=3)

    assert [step for step in range(1, 10) if processor.should_log(step)] == [1, 3, 6, 9]


# -- derivations --------------------------------------------------------------


def test_throughput_divides_by_the_non_data_parallel_size() -> None:
    """Tokens/s is per device, so N replicas holding the whole model each count
    1/N of the tokens that their step processed."""
    processor = MetricsProcessor(
        Config(),
        parallel_dims=_FakeParallelDims(non_data_parallel=4),
        num_flops_per_token=1000,
    )
    processor.add_tokens(4000)
    processor.time_last_log -= 1.0  # pretend the window took exactly one second

    assert _derive(processor).tps == pytest.approx(1000, rel=1e-3)


def test_throughput_is_not_divided_without_parallel_dims() -> None:
    processor = _processor()
    processor.add_tokens(4000)
    processor.time_last_log -= 1.0

    assert _derive(processor).tps == pytest.approx(4000, rel=1e-3)


def test_tflops_is_flops_per_token_times_throughput() -> None:
    processor = _processor()
    processor.add_tokens(2000)
    processor.time_last_log -= 1.0

    derived = _derive(processor)

    assert derived.tps == pytest.approx(2000, rel=1e-3)
    assert derived.tflops == pytest.approx(1000 * 2000 / 1e12, rel=1e-3)


def test_mfu_is_none_when_the_device_peak_is_unknown() -> None:
    """A ratio against an unknown denominator is not a measurement."""
    processor = _processor()
    assert processor.gpu_peak_flops == 0.0

    assert _derive(processor).mfu is None


def test_mfu_is_computed_against_the_device_peak() -> None:
    processor = _processor()
    processor.gpu_peak_flops = 1000.0
    processor.add_tokens(1000)
    processor.time_last_log -= 1.0

    # mfu = 100 * flops_per_token * tps / peak = 100 * 1000 * 1000 / 1000
    assert _derive(processor).mfu == pytest.approx(100_000.0, rel=1e-3)


def test_time_per_step_averages_over_the_window() -> None:
    """end_to_end is per step, not per window: a run logging every 5 steps
    would otherwise report a number 5x its step time."""
    processor = _processor()
    processor.add_tokens(100)
    processor.step_last_log = 1  # as the step-1 log would have left it
    processor.time_last_log -= 0.5

    # 0.5s elapsed between step 1 and step 6 is five steps' worth.
    assert _derive(processor, 6).time_end_to_end == pytest.approx(0.1, rel=1e-3)


def test_data_loading_is_averaged_and_also_reported_as_a_share() -> None:
    processor = _processor()
    processor.time_last_log -= 1.0
    processor.add_data_loading_time(0.1)
    processor.add_data_loading_time(0.3)

    derived = _derive(processor)

    assert derived.time_data_loading == pytest.approx(0.2, rel=1e-3)
    assert derived.time_data_loading_pct == pytest.approx(40.0, rel=1e-3)


def test_a_window_with_no_recorded_fetch_reports_zero_not_a_division() -> None:
    """Upstream divides by the sample count unguarded, so a caller that does
    not time its loader -- or a validation pass before step 1 -- raises."""
    processor = _processor()
    processor.time_last_log -= 1.0

    assert _derive(processor).time_data_loading == 0.0


# -- the reported metrics -----------------------------------------------------


def test_log_reports_the_core_metrics_and_advances_the_window() -> None:
    processor = _processor()
    sink = _RecordingLogger()
    processor.logger = sink
    processor.add_tokens(900)
    processor.add_data_loading_time(0.01)

    processor.log(1, global_avg_loss=4.5, global_max_loss=4.75, grad_norm=2.0)

    assert len(sink.calls) == 1
    reported, step = sink.calls[0]
    assert step == 1
    assert reported["loss_metrics/global_avg_loss"] == 4.5
    assert reported["loss_metrics/global_max_loss"] == 4.75
    assert reported["grad_norm"] == 2.0
    assert "throughput(tps)" in reported
    assert reported["memory/num_ooms"] == 0

    # The window is closed: the next report measures from here.
    assert processor.ntokens_since_last_log == 0
    assert processor.data_loading_times == []
    assert processor.step_last_log == 1


def test_log_anchors_itself_without_a_prior_should_log() -> None:
    """``should_log`` and ``log`` must not have to be called in that order."""
    processor = _processor()
    sink = _RecordingLogger()
    processor.logger = sink
    processor.add_tokens(100)

    processor.log(5, global_avg_loss=1.0, global_max_loss=1.0, grad_norm=1.0)

    # Anchored at step 4, so the end-to-end time is one step's worth, not five's.
    assert "time_metrics/end_to_end(s)" in sink.calls[0][0]


def test_mfu_is_reported_when_it_is_known() -> None:
    processor = _processor()
    processor.gpu_peak_flops = 1000.0
    sink = _RecordingLogger()
    processor.logger = sink
    processor.add_tokens(1000)
    processor.time_last_log -= 1.0

    processor.log(1, global_avg_loss=1.0, global_max_loss=1.0, grad_norm=1.0)

    # mfu = 100 * flops_per_token * tps / peak = 100 * 1000 * 1000 / 1000
    assert sink.calls[0][0]["mfu(%)"] == pytest.approx(100_000.0, rel=1e-3)


def test_mfu_is_omitted_from_the_report_when_it_is_not_known() -> None:
    """MFU is not merely shown as N/A: the key is absent, so a reader charting
    it gets a gap rather than a zero."""
    processor = _processor()
    assert processor.gpu_peak_flops == 0.0
    sink = _RecordingLogger()
    processor.logger = sink

    processor.log(1, global_avg_loss=1.0, global_max_loss=1.0, grad_norm=1.0)

    assert "mfu(%)" not in sink.calls[0][0]


def test_extra_metrics_are_merged_into_the_report() -> None:
    processor = _processor()
    sink = _RecordingLogger()
    processor.logger = sink

    processor.log(
        1,
        global_avg_loss=1.0,
        global_max_loss=1.0,
        grad_norm=1.0,
        extra_metrics={"custom/thing": 7},
    )

    assert sink.calls[0][0]["custom/thing"] == 7


def test_the_tag_is_applied_by_the_sink_and_only_by_the_sink() -> None:
    """Applied once: a key carrying the tag from the processor *and* the sink
    records as ``baseline/baseline/throughput(tps)``."""
    processor = MetricsProcessor(
        Config(),
        parallel_dims=None,
        num_flops_per_token=1000,
        tag="baseline",
    )
    untagged, tagged = _TaggingLogger(None), _TaggingLogger("baseline")
    container = LoggerContainer()
    container.add_logger(untagged)
    container.add_logger(tagged)
    processor.logger = container

    processor.log(1, global_avg_loss=1.0, global_max_loss=1.0, grad_norm=1.0)

    # What the processor handed over is untagged...
    assert "throughput(tps)" in untagged.calls[0][0]
    # ...and what a sink makes of it is tagged exactly once.
    assert set(tagged.calls[0][0]) == {
        f"baseline/{key}" for key in untagged.calls[0][0]
    }


# -- the validation path ------------------------------------------------------


def test_log_validation_reports_under_its_own_prefix() -> None:
    processor = _processor()
    sink = _RecordingLogger()
    processor.logger = sink
    processor.add_tokens(500)

    processor.log_validation(loss=3.5, step=7)

    reported, step = sink.calls[0]
    assert step == 7
    assert reported["validation_metrics/loss"] == 3.5
    assert "validation_metrics/throughput(tps)" in reported
    # Training keys must not appear: a reader splitting on the prefix would
    # otherwise see validation points mixed into the training loss curve.
    assert not any(key.startswith("loss_metrics/") for key in reported)


# -- sinks --------------------------------------------------------------------


def test_logger_container_fans_out_and_closes_every_sink() -> None:
    container = LoggerContainer()
    first, second = _RecordingLogger(), _RecordingLogger()
    container.add_logger(first)
    container.add_logger(second)

    assert container.number_of_loggers == 2
    container.log({"a": 1}, step=3)
    assert first.calls == [({"a": 1}, 3)]
    assert second.calls == [({"a": 1}, 3)]

    container.close()
    assert first.closed and second.closed


def test_a_processor_with_nothing_enabled_logs_to_a_no_op() -> None:
    """Every rank calls log(); only some ranks record. A no-op logger rather
    than None keeps that one code path."""
    processor = _processor()
    assert isinstance(processor.logger, BaseLogger)


def test_only_the_metrics_rank_records_unless_all_ranks_are_asked_for() -> None:
    processor = MetricsProcessor(
        Config(enable_tensorboard=True),
        parallel_dims=_FakeParallelDims(pp=1, world_size=1),
        num_flops_per_token=0,
    )
    # Single process, metrics rank 0, so this rank does record.
    assert isinstance(processor.logger, LoggerContainer)

    off_rank = MetricsProcessor(
        Config(enable_tensorboard=True),
        parallel_dims=_FakeParallelDims(pp=2, world_size=4),
        num_flops_per_token=0,
    )
    # Metrics rank is 2 under pp=2; this process is rank 0, so it does not.
    assert isinstance(off_rank.logger, BaseLogger)
    assert not isinstance(off_rank.logger, LoggerContainer)


class _WandBLoggerRaising(BaseLogger):
    """A WandBLogger whose construction fails -- the only part of its life
    cycle the processor takes part in."""

    error: Exception = ImportError("no wandb")

    def __init__(self, *args, **kwargs) -> None:
        raise type(self).error


def test_enabling_wandb_without_the_package_says_so(monkeypatch) -> None:
    """The one exception that is rewritten is the one the user can install."""
    monkeypatch.setattr(metrics_module, "WandBLogger", _WandBLoggerRaising)
    _WandBLoggerRaising.error = ImportError("No module named 'wandb'")

    with pytest.raises(ImportError, match="wandb is not installed"):
        MetricsProcessor(
            Config(enable_wandb=True),
            parallel_dims=None,
            num_flops_per_token=0,
        )


def test_a_wandb_misconfiguration_is_not_swallowed(monkeypatch) -> None:
    """torchtitan catches every exception from WandBLogger and logs it, so a
    mistyped WANDB_PROJECT silently disables tracking for the whole run. Only
    a missing package is rewritten here; anything else -- a missing API key, a
    bad entity -- reaches the caller."""
    from wandb.errors import UsageError

    monkeypatch.setattr(metrics_module, "WandBLogger", _WandBLoggerRaising)
    _WandBLoggerRaising.error = UsageError("No API key configured.")

    with pytest.raises(UsageError, match="No API key"):
        MetricsProcessor(
            Config(enable_wandb=True),
            parallel_dims=None,
            num_flops_per_token=0,
        )


def test_tensorboard_writes_one_scalar_per_metric(tmp_path) -> None:
    """The one sink with a real writer behind it, checked for the tagged key
    name rather than for the file landing on disk."""
    writer = TensorBoardLogger(str(tmp_path), tag="run1")

    seen: list[tuple[str, float, int]] = []
    writer.writer.add_scalar = lambda tag, value, step: seen.append((tag, value, step))
    writer.log({"loss": 1.5, "tps": 10.0}, step=4)
    writer.close()

    assert seen == [("run1/loss", 1.5, 4), ("run1/tps", 10.0, 4)]


class _FakeWandB:
    """Just the surface WandBLogger touches: ``log``, ``run``, ``finish``."""

    def __init__(self, run: object = "a-run") -> None:
        self.run = run
        self.sent: list[tuple[dict, int]] = []
        self.finished = False

    def log(self, metrics: dict, step: int) -> None:
        self.sent.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


def test_wandb_logger_prefixes_what_it_sends() -> None:
    logger = WandBLogger.__new__(WandBLogger)  # skip init: it starts a run
    logger.tag = "run1"
    logger.wandb = _FakeWandB()

    logger.log({"loss": 1.5}, step=4)

    assert logger.wandb.sent == [({"run1/loss": 1.5}, 4)]


def test_wandb_logger_sends_untagged_keys_without_a_tag() -> None:
    logger = WandBLogger.__new__(WandBLogger)
    logger.tag = None
    logger.wandb = _FakeWandB()

    logger.log({"loss": 1.5}, step=4)

    assert logger.wandb.sent == [({"loss": 1.5}, 4)]


def test_wandb_close_tolerates_an_already_finished_run() -> None:
    """``run`` is None once finish has run; calling it again raises, and close()
    tends to be called during interpreter shutdown."""
    logger = WandBLogger.__new__(WandBLogger)
    logger.wandb = _FakeWandB(run=None)

    logger.close()

    assert not logger.wandb.finished


def test_wandb_close_finishes_a_live_run() -> None:
    logger = WandBLogger.__new__(WandBLogger)
    logger.wandb = _FakeWandB()

    logger.close()

    assert logger.wandb.finished


# -- device memory ------------------------------------------------------------


def test_device_memory_monitor_reports_zeros_on_cpu() -> None:
    """CPU has no allocator to sample; zeros are the honest answer, and the
    percentage is suppressed rather than divided by a zero capacity."""
    monitor = DeviceMemoryMonitor("cpu")

    stats = monitor.get_peak_stats()

    assert (stats.max_active_gib, stats.max_reserved_gib) == (0.0, 0.0)
    assert (stats.max_active_pct, stats.max_reserved_pct) == (0.0, 0.0)
    assert (stats.num_alloc_retries, stats.num_ooms) == (0, 0)
    monitor.reset_peak_stats()  # must not raise

    assert monitor._to_pct(1234) == 0.0


def test_gib_is_binary_gigabytes() -> None:
    """Device memory is reported in GiB. Using 1e9 here understates a 80GiB
    card as 85.9."""
    assert DeviceMemoryMonitor._to_gib(1024**3) == 1.0
    assert DeviceMemoryMonitor._to_gib(0) == 0.0


# -- metrics rank -------------------------------------------------------------


def test_metrics_rank_is_zero_without_pipeline_parallelism() -> None:
    assert get_metrics_rank(parallel_dims=_FakeParallelDims(), pp_schedule="1F1B") == 0


def test_metrics_rank_is_the_first_rank_of_the_last_stage() -> None:
    """The loss only exists on the last stage, so that is the rank that reports."""
    dims = _FakeParallelDims(pp=4, world_size=16)

    assert get_metrics_rank(parallel_dims=dims, pp_schedule="1F1B") == 12


def test_metrics_rank_is_zero_for_a_v_block_schedule() -> None:
    """ZBV returns loss on rank 0 regardless of how many stages there are."""
    dims = _FakeParallelDims(pp=4, world_size=16)

    assert get_metrics_rank(parallel_dims=dims, pp_schedule="ZBVZeroBubble") == 0


def test_ensure_pp_loss_visible_warns_when_log_rank_misses_the_loss(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LOG_RANK", "0")
    dims = _FakeParallelDims(pp=4, world_size=16)

    with _capturing() as capture:
        ensure_pp_loss_visible(parallel_dims=dims, pp_schedule="1F1B", color=NoColor())

    assert any("loss is not visible" in m for m in capture.messages())


def test_ensure_pp_loss_visible_is_silent_when_the_rank_is_watched(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LOG_RANK", "0,12")
    dims = _FakeParallelDims(pp=4, world_size=16)

    with _capturing() as capture:
        ensure_pp_loss_visible(parallel_dims=dims, pp_schedule="1F1B", color=NoColor())

    assert not any("loss is not visible" in m for m in capture.messages())


def test_ensure_pp_loss_visible_says_nothing_without_pipeline_parallelism(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LOG_RANK", "0")

    with _capturing() as capture:
        ensure_pp_loss_visible(
            parallel_dims=_FakeParallelDims(), pp_schedule="1F1B", color=NoColor()
        )

    assert capture.messages() == []


# -- colour -------------------------------------------------------------------


def test_no_color_and_color_expose_the_same_fields() -> None:
    """The module asserts this at import; pinning it here records why it
    matters -- a caller formatting with ``color.orange`` works under one and
    raises under the other."""
    from dataclasses import fields

    assert {f.name for f in fields(Color)} == {f.name for f in fields(NoColor)}
    assert all(getattr(NoColor, f.name) == "" for f in fields(NoColor))


def test_colors_are_off_when_the_config_asks_for_it() -> None:
    assert colors_enabled(disable_color_printing=True) is False


def test_no_color_env_vetoes_escapes(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert colors_enabled(disable_color_printing=False) is False


def test_force_color_env_asks_for_them_anyway(monkeypatch) -> None:
    """FORCE_COLOR is how a user inside a multiplexer says they want escapes
    despite stdout not being a tty."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert colors_enabled(disable_color_printing=False) is True


def test_a_dumb_terminal_is_not_given_escapes(monkeypatch) -> None:
    """A tty under TERM=dumb accepts the isatty check and then prints the
    escapes literally."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert colors_enabled(disable_color_printing=False) is False


def test_the_config_switch_wins_over_a_willing_terminal(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert colors_enabled(disable_color_printing=True) is False


# -- peak flops ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("NVIDIA A100-SXM4-80GB", 312e12),
        ("NVIDIA A6000", 154.85e12),
        ("NVIDIA H100 NVL", 835e12),
        ("NVIDIA H100 PCIe", 756e12),
        ("NVIDIA H100 80GB HBM3", 989e12),
        ("NVIDIA H200", 989e12),
        ("NVIDIA H20", 148e12),
        ("NVIDIA GB200", 2.5e15),
        ("NVIDIA GB300", 2.5e15),
        ("NVIDIA B300", 2.25e15),
        ("NVIDIA B200", 2.25e15),
        ("AMD MI355X", 2500e12),
        ("AMD MI300X", 1300e12),
        ("AMD MI325X", 1300e12),
        ("AMD MI250X", 191.5e12),
        ("NVIDIA L40S", 362e12),
        ("TPU v6e", 918e12),
        ("TPU v7", 2307e12 / 2),
    ],
)
def test_peak_flops_are_known_for_the_common_accelerators(name, expected) -> None:
    assert get_peak_flops(name) == expected


def test_a_more_specific_name_beats_its_own_substring() -> None:
    """The ordering that a plain substring match gets wrong: "GB300" contains
    "B300", and "MI250X" contains "MI250"."""
    assert get_peak_flops("NVIDIA GB300") != get_peak_flops("NVIDIA B300")
    assert get_peak_flops("AMD MI250X") == 191.5e12


def test_h100_variants_are_distinguished_by_their_suffix() -> None:
    assert get_peak_flops("NVIDIA H100 NVL") != get_peak_flops("NVIDIA H100 PCIe")
    # An undifferentiated name selects SXM, the default the module documents.
    assert get_peak_flops("NVIDIA H100") == 989e12


def test_peak_flops_are_zero_for_an_unknown_device() -> None:
    """Zero rather than a plausible substitute: MFU is a ratio against this,
    and a guess would be reported as a measurement."""
    assert get_peak_flops("Apple M2") == 0.0
    assert get_peak_flops("cpu") == 0.0


def test_peak_flops_ignore_case() -> None:
    assert get_peak_flops("nvidia h100") == get_peak_flops("NVIDIA H100")


def test_a_tpu_name_must_start_with_tpu() -> None:
    """The verticals are matched with startswith, so a device that merely
    mentions a TPU version does not borrow its peak."""
    assert get_peak_flops("not a tpu v6e") == 0.0


# -- num_flops_per_token ------------------------------------------------------


def _cfg(**model_kwargs) -> HybridMeshConfig:
    return HybridMeshConfig(model=ModelArguments(**model_kwargs))


def test_num_flops_per_token_is_positive_and_grows_with_the_model() -> None:
    small = _cfg(hidden_size=64, num_hidden_layers=2)
    large = _cfg(hidden_size=64, num_hidden_layers=4)

    assert num_flops_per_token(small) > 0
    assert num_flops_per_token(large) > num_flops_per_token(small)


def test_num_flops_per_token_is_affine_in_the_layer_count() -> None:
    """Every term scales with the layer count except the output projection, so
    the marginal cost of a layer is constant and the FLOPs are affine in the
    depth. Reading it as proportional would hide the lm_head offset."""
    one = num_flops_per_token(_cfg(hidden_size=64, num_hidden_layers=1))
    two = num_flops_per_token(_cfg(hidden_size=64, num_hidden_layers=2))
    four = num_flops_per_token(_cfg(hidden_size=64, num_hidden_layers=4))

    assert four - two == 2 * (two - one)


def test_num_flops_per_token_matches_the_formula_exactly() -> None:
    """A fully specified model, so every term is checked rather than bounded."""
    cfg = HybridMeshConfig(
        model=ModelArguments(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        ),
        training=TrainingArguments(max_seq_len=4),
    )
    # _cfg() above cannot set training, so max_seq_len is the default 64.
    assert cfg.max_seq_len == 4

    hidden, intermediate, vocab, heads, kv_heads = 8, 16, 32, 2, 1
    head_dim, seq_len, layers = hidden // heads, 4, 2
    per_layer = (
        2 * hidden * heads * head_dim  # q
        + 2 * hidden * kv_heads * head_dim  # k
        + 2 * hidden * kv_heads * head_dim  # v
        + 2 * heads * head_dim * hidden  # o
        + 3 * 2 * hidden * intermediate  # gate, up, down
    )
    attention = 6 * heads * 2 * head_dim * seq_len

    assert num_flops_per_token(cfg) == (
        3 * (layers * per_layer + 2 * vocab * hidden) + layers * attention
    )


def test_the_attention_term_grows_with_sequence_length() -> None:
    """The parameter term does not depend on the sequence length, so this
    isolates the attention term."""
    short = HybridMeshConfig(
        model=ModelArguments(hidden_size=8),
        training=TrainingArguments(max_seq_len=8),
    )
    long = HybridMeshConfig(
        model=ModelArguments(hidden_size=8),
        training=TrainingArguments(max_seq_len=32),
    )

    assert num_flops_per_token(long) > num_flops_per_token(short)


def test_num_flops_per_token_is_zero_when_the_config_is_incomplete(
    monkeypatch,
) -> None:
    """A missing size suppresses MFU and tflops rather than producing a number
    derived from guessed geometry."""
    from hpmesh.models import hf_wrapper

    class _Bare:
        seq_len = 4

    monkeypatch.setattr(hf_wrapper, "build_model_config_for", lambda cfg: _Bare())

    assert hf_wrapper.num_flops_per_token(HybridMeshConfig()) == 0


def test_num_flops_per_token_defaults_kv_heads_to_the_head_count(
    monkeypatch,
) -> None:
    """Missing GQA metadata must fall back to full multi-head, not crash. The
    failure mode is a silently wrong magnitude, so it is pinned here."""
    from hpmesh.models import hf_wrapper

    def _config(num_key_value_heads):
        class _Arch:
            vocab_size = 32
            hidden_size = 8
            intermediate_size = 16
            num_hidden_layers = 1
            num_attention_heads = 2
            head_dim = 4

        _Arch.num_key_value_heads = num_key_value_heads
        return _Arch()

    cfg = HybridMeshConfig(
        model=ModelArguments(hidden_size=8),
        training=TrainingArguments(max_seq_len=4),
    )

    monkeypatch.setattr(hf_wrapper, "build_model_config_for", lambda _: _config(None))
    unset = hf_wrapper.num_flops_per_token(cfg)

    monkeypatch.setattr(hf_wrapper, "build_model_config_for", lambda _: _config(2))
    explicit = hf_wrapper.num_flops_per_token(cfg)

    assert unset == explicit
