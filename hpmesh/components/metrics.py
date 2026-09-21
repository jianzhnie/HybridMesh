"""Training metrics: collect them, scale them, and send them somewhere.

Vendored from torchtitan's ``observability/metrics.py``. The shape is kept --
text loggers behind a common ``BaseLogger``, a ``MetricsProcessor`` that owns
the timing and derivations, device-memory sampling via a
``DeviceMemoryMonitor`` -- because the split is what makes the derivations
(throughput, MFU, time per step, data-loading share) testable without a
TensorBoard writer or a GPU in the room.

Departures from torchtitan, all subtractions:

* **No fault tolerance.** torchtitan threads ``ft_enable``/``ft_replica_id``
  into both the log directory layout and the metrics rank. hpmesh has no
  replica concept, so both are gone rather than carried as dead flags.

* **No ``OptimizersContainer`` / ``model_parts``.** torchtitan reaches into
  both to compute a per-optimizer-steps metric. hpmesh's loop tracks one step
  counter, so the metric has no source and no consumer here.

* **No wandb failure swallowing.** torchtitan catches every exception from
  ``WandBLogger`` and logs it, on the grounds that losing experiment tracking
  should not kill a run. hpmesh lets it raise -- the checkpointer's config
  rejects options it cannot honour for the same reason, and a mistyped
  ``WANDB_PROJECT`` that silently disables tracking is the failure mode this
  avoids. Only the missing-package case is caught, because that one is an
  installable dependency rather than a misconfiguration.

* **No ``has_quantization``.** torchtitan suppresses MFU when the run is
  quantized, since the peak it divides by is a dense BF16 figure. hpmesh has no
  quantization path, so the flag has no producer and a value it could never be
  set to is a branch nothing can exercise.

* **Colour is vetoed by the terminal, not only by the config.** See
  ``utils/monitoring.colors_enabled``.

* **MFU is suppressed when the device is unknown**, rather than assuming A100
  peak. A ratio measured against the wrong denominator is worse than no ratio,
  and on a laptop CPU the denominator is not a number at all.

* **A window with no recorded data-loading time reports zero**, rather than
  dividing by the number of samples it has (which is zero). torchtitan's
  trainer times every fetch, so it never meets the empty list; anything that
  logs without one -- a validation pass before the first training step, or a
  caller that does not instrument its loader -- crashes there.

One addition: ``MetricsProcessor.log`` never calls into ``torch.distributed``.
torchtitan's ``_get_metrics_rank`` returns rank 0 unconditionally outside the
PP case, and its logger construction calls ``torch.distributed.get_rank()``
regardless -- so every rank of a gloo-rendezvoused CPU run builds the same
logger. hpmesh reads the rank from ``utils/logger_utils``, which already has to
answer "which rank am I" for the log prefix and handles the single-process case.
"""

from __future__ import annotations

import os
import time
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch

from ..utils.logger_utils import get_distributed_rank, get_logger
from ..utils.monitoring import (
    Color,
    NoColor,
    colors_enabled,
    get_device_capacity_bytes,
    get_device_name,
    get_peak_flops,
)

if TYPE_CHECKING:
    # Annotation-only. ``parallel_dims`` imports the training config, which
    # imports the checkpointer, so importing it for real here would close a
    # cycle back into the components package that only exists to name a type.
    from ..parallel.parallel_dims import ParallelDims
    from ..trainer.config import MetricsConfig

# hpmesh configures handlers per module, rather than on the root logger the
# way torchtitan does, so this has to be get_logger for the metrics lines to
# reach the console at all.
logger = get_logger(__name__)

__all__ = [
    "BaseLogger",
    "DeviceMemStats",
    "DeviceMemoryMonitor",
    "LoggerContainer",
    "MetricsProcessor",
    "TensorBoardLogger",
    "WandBLogger",
    "build_device_memory_monitor",
    "ensure_pp_loss_visible",
    "get_metrics_rank",
]


DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class DeviceMemoryMonitor:
    """Peak device-memory sampling over a logging window.

    The trainer resets the peaks after every log, so each report covers exactly
    the steps since the last one. That is what makes ``max_reserved`` a useful
    number rather than a high-water mark set once during startup and never
    revisited.
    """

    def __init__(self, device_type: str) -> None:
        self.device_type = device_type
        self.device_name = get_device_name()
        self.device_capacity = get_device_capacity_bytes()
        self.device_capacity_gib = self._to_gib(self.device_capacity)
        self.reset_peak_stats()

    @staticmethod
    def _to_gib(memory_in_bytes: float) -> float:
        # GiB (gibibyte) is 1024^3, GB is 1000^3. Device memory is reported in
        # the former.
        return memory_in_bytes / (1024 * 1024 * 1024)

    def _to_pct(self, memory: float) -> float:
        if self.device_capacity == 0:
            return 0.0
        return 100 * memory / self.device_capacity

    def _device_module(self):
        if self.device_type == "cuda":
            return torch.cuda
        return None

    def get_peak_stats(self) -> DeviceMemStats:
        device_module = self._device_module()
        if device_module is None:
            # CPU has no allocator to ask, so every field is a real zero rather
            # than a placeholder: there genuinely is no device memory in use.
            return DeviceMemStats(0.0, 0.0, 0.0, 0.0, 0, 0)

        device_info = device_module.memory_stats()
        max_active = device_info.get("active_bytes.all.peak", -1)
        max_reserved = device_info.get("reserved_bytes.all.peak", -1)
        num_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(
                "%d %s memory allocation retries.",
                num_retries,
                self.device_type.upper(),
            )
        if num_ooms > 0:
            logger.warning(
                "%d %s OOM errors thrown.", num_ooms, self.device_type.upper()
            )

        return DeviceMemStats(
            self._to_gib(max_active),
            self._to_pct(max_active),
            self._to_gib(max_reserved),
            self._to_pct(max_reserved),
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self) -> None:
        device_module = self._device_module()
        if device_module is not None:
            device_module.reset_peak_memory_stats()


def build_device_memory_monitor() -> DeviceMemoryMonitor:
    """Build a monitor and say which device it is watching."""
    from ..utils.device import device_type

    monitor = DeviceMemoryMonitor(device_type)
    if device_type == "cuda":
        logger.info(
            "%s capacity: %s with %.2fGiB memory",
            device_type.upper(),
            monitor.device_name,
            monitor.device_capacity_gib,
        )
    return monitor


class BaseLogger:
    """A sink that takes metrics. Logging nothing is a valid implementation."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Writes scalars to a TensorBoard event directory."""

    def __init__(self, log_dir: str, tag: str | None = None):
        # Imported here rather than at module scope: tensorboard is a large
        # import, and a run that logs to stdout only should not pay for it.
        from torch.utils.tensorboard import SummaryWriter

        self.tag = tag
        self.writer = SummaryWriter(log_dir, max_queue=1000)
        logger.info("TensorBoard logging enabled. Logs will be saved at %s", log_dir)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            tag = key if self.tag is None else f"{self.tag}/{key}"
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(BaseLogger):
    """Streams metrics to Weights & Biases.

    Every run parameter comes from the environment, matching torchtitan. That is
    deliberate for a launcher-driven tool: ``WANDB_RUN_ID`` and ``WANDB_RESUME_FROM``
    are how a requeued job rejoins its own run, and reading them from the
    environment means a launcher can set them without touching the config.
    """

    def __init__(
        self,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        # Deferred so that a run without wandb enabled does not import it.
        import wandb

        self.wandb = wandb
        self.tag = tag

        os.makedirs(log_dir, exist_ok=True)
        self.wandb.init(
            entity=os.getenv("WANDB_TEAM"),
            project=os.getenv("WANDB_PROJECT", "hpmesh"),
            name=os.getenv("WANDB_RUN_NAME"),
            id=os.getenv("WANDB_RUN_ID"),
            notes=os.getenv("WANDB_RUN_NOTES"),
            tags=os.getenv("WANDB_RUN_TAGS"),
            group=os.getenv("WANDB_RUN_GROUP"),
            job_type=os.getenv("WANDB_RUN_JOB_TYPE"),
            resume_from=os.getenv("WANDB_RESUME_FROM"),
            fork_from=os.getenv("WANDB_FORK_FROM"),
            dir=log_dir,
            config=config_dict,
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        prefixed = {
            key if self.tag is None else f"{self.tag}/{key}": value
            for key, value in metrics.items()
        }
        self.wandb.log(prefixed, step=step)

    def close(self) -> None:
        # ``run`` is None when init failed or ``finish`` already ran; calling
        # finish again would raise during interpreter shutdown.
        if self.wandb.run is not None:
            self.wandb.finish()


class LoggerContainer(BaseLogger):
    """Fans one ``log`` call out to every enabled sink."""

    def __init__(self) -> None:
        self._loggers: list[BaseLogger] = []

    def add_logger(self, logger_instance: BaseLogger) -> None:
        self._loggers.append(logger_instance)

    @property
    def number_of_loggers(self) -> int:
        return len(self._loggers)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log(metrics, step)

    def close(self) -> None:
        for logger_instance in self._loggers:
            logger_instance.close()


def get_metrics_rank(*, parallel_dims: ParallelDims, pp_schedule: str) -> int:
    """The rank whose loss is the reportable one.

    Rank 0, except under pipeline parallelism, where the loss exists only on the
    last stage. A V-block schedule is the exception twice over: it returns loss
    on rank 0 like an ordinary run.
    """
    if not parallel_dims.pp_enabled:
        return 0
    if pp_schedule == "ZBVZeroBubble":
        return 0
    # First rank of the last pipeline stage. Ranks are laid out
    # [dp_replicate, dp_shard, cp, tp] within a stage, so this is the first
    # rank of the final stage block.
    pp_size = parallel_dims.pp
    return (parallel_dims.world_size // pp_size) * (pp_size - 1)


def ensure_pp_loss_visible(
    *, parallel_dims: ParallelDims, pp_schedule: str, color: Color | NoColor
) -> None:
    """Warn when the loss will be computed on a rank nobody is watching.

    Under pipeline parallelism the loss lives on one rank, and ``LOG_RANK``
    decides which ranks print. Getting that wrong produces a run that trains
    correctly and reports nothing -- which reads exactly like a hang.
    """
    if not parallel_dims.pp_enabled:
        return
    if pp_schedule == "ZBVZeroBubble":
        return

    loss_visible_rank = get_metrics_rank(
        parallel_dims=parallel_dims, pp_schedule=pp_schedule
    )
    env_logged_ranks = [r for r in os.environ.get("LOG_RANK", "").split(",") if r]
    if str(loss_visible_rank) not in env_logged_ranks:
        logger.warning(
            "%sPipeline Parallel loss is not visible. Please add %srank %d%s to "
            "the LOG_RANK environment variable in the launcher.%s",
            color.red,
            color.yellow,
            loss_visible_rank,
            color.red,
            color.reset,
        )


@dataclass(kw_only=True, slots=True)
class _Derived:
    """The numbers a log call computes from its window, before naming."""

    tps: float
    tflops: float
    mfu: float | None
    time_end_to_end: float
    time_data_loading: float
    time_data_loading_pct: float


class MetricsProcessor:
    """Turns per-step counters into throughput, MFU, and memory, and reports them.

    Args:
        config: which sinks to build and how often to log.
        parallel_dims: the resolved parallelism, for the throughput divisor and
            the metrics rank. ``None`` in the single-process case.
        dump_folder: base directory for the TensorBoard event files.
        pp_schedule: the pipeline schedule name, which decides the metrics rank.
        num_flops_per_token: model FLOPs per token, used for tflops and MFU. The
            caller sets this once the model exists. ``0`` -- the value
            :func:`~hpmesh.models.hf_wrapper.num_flops_per_token` returns for a
            config whose geometry it cannot read -- suppresses MFU rather than
            reporting ``0.00%``; ``tflops`` is then ``0.0`` of its own accord.
        config_dict: the full job config, handed to wandb. Only wandb reads it.
        tag: prefix applied to every recorded key, so two runs can share one
            project or event directory. The console line is not tagged.
    """

    def __init__(
        self,
        config: MetricsConfig,
        *,
        parallel_dims: ParallelDims | None,
        dump_folder: str = "./outputs",
        pp_schedule: str = "1F1B",
        num_flops_per_token: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        self.config = config
        self.parallel_dims = parallel_dims
        self.logger = self._build_metric_logger(
            config=config,
            parallel_dims=parallel_dims,
            dump_folder=dump_folder,
            pp_schedule=pp_schedule,
            config_dict=config_dict,
            tag=tag,
        )
        self.device_memory_monitor = build_device_memory_monitor()
        self.color = (
            Color()
            if colors_enabled(disable_color_printing=config.disable_color_printing)
            else NoColor()
        )
        self.gpu_peak_flops = get_peak_flops(self.device_memory_monitor.device_name)
        self.num_flops_per_token = num_flops_per_token

        self.ntokens_since_last_log = 0
        self.data_loading_times: list[float] = []
        self.time_last_log = time.perf_counter()
        self.step_last_log: int | None = None
        self.device_memory_monitor.reset_peak_stats()

    def set_num_flops_per_token(self, num_flops_per_token: int) -> None:
        """Set the FLOPs-per-token figure once the model is built.

        Separate from ``__init__`` because the metrics processor is built before
        the model, and the number cannot be known until it exists.
        """
        self.num_flops_per_token = num_flops_per_token

    @property
    def _non_data_parallel_size(self) -> int:
        """Ranks that hold a copy of the whole model rather than a data shard.

        Throughput is per device, so the token count is divided by how many
        devices contributed to it -- one per data-parallel replica group.
        """
        if self.parallel_dims is None:
            return 1
        return self.parallel_dims.non_data_parallel_size

    def should_log(self, step: int) -> bool:
        """Whether ``step`` is a logging step. The first step always is."""
        return step == 1 or step % self.config.log_freq == 0

    def add_data_loading_time(self, seconds: float) -> None:
        """Record how long this step's data fetch blocked."""
        self.data_loading_times.append(seconds)

    def add_tokens(self, num_tokens: int) -> None:
        """Record tokens processed since the last log."""
        self.ntokens_since_last_log += num_tokens

    def _build_metric_logger(
        self,
        *,
        config: MetricsConfig,
        parallel_dims: ParallelDims | None,
        dump_folder: str,
        pp_schedule: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> BaseLogger:
        """Build the sink this rank should write to.

        Returns a no-op logger -- rather than ``None`` -- when nothing is
        enabled or this is not the metrics rank, so the caller has one code path.
        """
        if not (config.enable_tensorboard or config.enable_wandb):
            return BaseLogger()

        if not config.save_for_all_ranks:
            if parallel_dims is None:
                metrics_rank = 0
            else:
                metrics_rank = get_metrics_rank(
                    parallel_dims=parallel_dims, pp_schedule=pp_schedule
                )
            if get_distributed_rank() != metrics_rank:
                return BaseLogger()

        base_log_dir = os.path.join(
            dump_folder,
            config.save_tb_folder,
            datetime.now().strftime("%Y%m%d-%H%M"),
        )
        if config.save_for_all_ranks:
            base_log_dir = os.path.join(base_log_dir, f"rank_{get_distributed_rank()}")

        logger_container = LoggerContainer()

        if config.enable_wandb:
            try:
                logger_container.add_logger(
                    WandBLogger(base_log_dir, config_dict=config_dict, tag=tag)
                )
            except ImportError as error:
                # Only the missing-package case: that is a dependency the user
                # can install. A credential or project error is a
                # misconfiguration they should see, not something to swallow.
                raise ImportError(
                    "metrics.enable_wandb is set but wandb is not installed. "
                    "Install it with 'pip install wandb', or unset it."
                ) from error

        if config.enable_tensorboard:
            logger_container.add_logger(TensorBoardLogger(base_log_dir, tag))

        return logger_container

    def _derive(self, step: int) -> _Derived:
        """Compute the windowed numbers that both log paths share.

        The keys these become carry no tag: a sensor is what the tag belongs to,
        and it applies it once on the way out. Applying it here as well would
        produce ``tag/tag/throughput(tps)`` in the recorded output.
        """
        time_delta = time.perf_counter() - self.time_last_log

        tps = self.ntokens_since_last_log / (time_delta * self._non_data_parallel_size)
        tflops = self.num_flops_per_token * tps / 1e12
        # MFU is a ratio against a datasheet dense-BF16 peak, so it is only
        # meaningful where the hardware actually achieves that. Where the peak
        # is unknown the measurement is suppressed rather than reported against
        # a guess -- see get_peak_flops.
        # The model's own geometry is the other input, and it is missing in the
        # same way: a config that does not expose the sizes the formula needs
        # makes num_flops_per_token 0, and 0 into a non-zero peak reports a
        # confident 0.00% rather than an unknown. Suppress both -- a wrong
        # number in a dashboard is worse than an absent one.
        # https://arxiv.org/abs/2204.02311 for the definition.
        if self.gpu_peak_flops == 0:
            mfu = None
        elif self.num_flops_per_token == 0:
            mfu = None
        else:
            mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops

        assert self.step_last_log is not None, "should_log must run before log"
        time_end_to_end = time_delta / (step - self.step_last_log)
        # Zero rather than a division: a window with no recorded fetch spent no
        # time fetching, which is the number. torchtitan reports the mean only,
        # so a caller that does not time its loader -- or a log before the first
        # training step -- divides by zero there.
        time_data_loading = (
            sum(self.data_loading_times) / len(self.data_loading_times)
            if self.data_loading_times
            else 0.0
        )
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

        return _Derived(
            tps=tps,
            tflops=tflops,
            mfu=mfu,
            time_end_to_end=time_end_to_end,
            time_data_loading=time_data_loading,
            time_data_loading_pct=time_data_loading_pct,
        )

    def _end_window(self, step: int) -> None:
        """Reset the counters that the next window measures over."""
        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.step_last_log = step
        self.device_memory_monitor.reset_peak_stats()

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Report one training step.

        Args:
            step: the current training step.
            global_avg_loss: total loss over total valid tokens, so it is
                independent of how the batch was split across ranks.
            global_max_loss: the worst rank's per-token loss. Equal to
                ``global_avg_loss`` when averaging over rank means rather than
                tokens -- see the note in the trainer.
            grad_norm: gradient norm, measured after clipping.
            extra_metrics: additional numbers the caller wants recorded.
        """
        if self.step_last_log is None:
            # The first log has no previous step to measure against. Anchoring
            # here rather than in should_log keeps the two entry points
            # (should_log then log) order-independent.
            self.step_last_log = step - 1

        derived = self._derive(step)
        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        metrics: dict[str, Any] = {
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "grad_norm": grad_norm,
            "throughput(tps)": derived.tps,
            "tflops": derived.tflops,
            "time_metrics/end_to_end(s)": derived.time_end_to_end,
            "time_metrics/data_loading(s)": derived.time_data_loading,
            "time_metrics/data_loading(%)": derived.time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }
        if derived.mfu is not None:
            metrics["mfu(%)"] = derived.mfu
        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        mfu = metrics.get("mfu(%)")
        mfu_str = "N/A" if mfu is None else f"{mfu:.2f}%"
        # A single pre-built message rather than a format string plus arguments:
        # the colour codes are already interpolated, so passing them as logging
        # arguments would leave a line no log handler could re-format.
        logger.info(
            f"{color.red}step: {step:2d}  "
            f"{color.green}loss: {global_avg_loss:8.5f}  "
            f"{color.orange}grad_norm: {grad_norm:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(derived.tps):,}  "
            f"{color.cyan}tflops: {derived.tflops:,.2f}  "
            f"{color.magenta}mfu: {mfu_str}{color.reset}"
        )

        self._end_window(step)

    def log_validation(
        self, loss: float, step: int, extra_metrics: dict[str, Any] | None = None
    ) -> None:
        """Report a validation pass.

        Uses the same window as ``log``: a validation pass that ran since the
        last log is inside this window's token count and elapsed time, so the
        throughput here is the one including validation.
        """
        if self.step_last_log is None:
            self.step_last_log = step - 1

        time_delta = time.perf_counter() - self.time_last_log
        device_mem_stats = self.device_memory_monitor.get_peak_stats()
        tps = self.ntokens_since_last_log / (time_delta * self._non_data_parallel_size)

        metrics: dict[str, Any] = {
            "validation_metrics/loss": loss,
            "validation_metrics/throughput(tps)": tps,
            "validation_metrics/memory/max_active(GiB)": (
                device_mem_stats.max_active_gib
            ),
            "validation_metrics/memory/max_active(%)": device_mem_stats.max_active_pct,
            "validation_metrics/memory/max_reserved(GiB)": (
                device_mem_stats.max_reserved_gib
            ),
            "validation_metrics/memory/max_reserved(%)": (
                device_mem_stats.max_reserved_pct
            ),
        }
        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.yellow}validate step: {step:2d}  "
            f"{color.green}loss: {loss:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}{color.reset}"
        )

        self._end_window(step)

    def close(self) -> None:
        self.logger.close()
