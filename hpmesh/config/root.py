"""HybridMeshConfig: the aggregate config root."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from transformers import AutoConfig

from hpmesh.config.checkpoint import CheckpointConfig
from hpmesh.config.data import DataloaderConfig
from hpmesh.config.model import ModelConfig
from hpmesh.config.optimizer import LRSchedulerConfig, OptimizerConfig
from hpmesh.config.parallel import ParallelConfig
from hpmesh.config.training import (
    MetricsConfig,
    ProfilerConfig,
    TrainingConfig,
    ValidationConfig,
)
from hpmesh.errors import ConfigError
from hpmesh.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class HybridMeshConfig:
    """Single entry point: composes the argument groups (no multiple inheritance).

    Each group validates itself in its own __post_init__ (run by default_factory).
    Nested groups (``parallel``, ``training``) are reachable both as themselves and
    through the flat property view below.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        # Cross-group check: CP must divide the sequence length.
        if self.training.max_seq_len % self.parallel.cp != 0:
            raise ConfigError(
                f"max_seq_len ({self.training.max_seq_len}) must be divisible by "
                f"cp ({self.parallel.cp})"
            )
        # Async TP is a compiled-TP optimization: without compile there is no
        # inductor pass to pipeline the collectives, and without TP there are
        # no collectives. Reject both halves here so the error names the flag
        # rather than surfacing deep inside the compile step.
        if self.training.compile_config.enable_async_tensor_parallel:
            if not self.training.compile:
                raise ConfigError(
                    "training.compile_config.enable_async_tensor_parallel "
                    "requires training.compile=True: async TP is an inductor "
                    "pass over compiled regions, so without compile it would "
                    "silently do nothing."
                )
            if self.parallel.tp < 2:
                raise ConfigError(
                    "training.compile_config.enable_async_tensor_parallel "
                    "requires tensor_parallel_size > 1 (got "
                    f"{self.parallel.tp}): it pipelines the TP collectives, "
                    "and there are none at tp=1."
                )

    # -- Flat view: lets the trainer read cfg.lr / cfg.steps / ... uniformly. --
    # The parallel degrees (dp/tp/pp/cp/ep) are deliberately NOT here: the
    # parallel layer takes ``cfg.parallel`` (a ParallelConfig) directly, so a
    # second flat spelling of the same numbers could only drift apart.
    @property
    def hf_model(self) -> str:
        return self.model.model_name_or_path

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.model.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.model.intermediate_size

    @property
    def num_hidden_layers(self) -> int:
        return self.model.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.model.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.model.num_key_value_heads

    @property
    def arch_overrides(self) -> dict[str, Any]:
        return self.model.arch_overrides

    @property
    def experts_implementation(self) -> str:
        return self.model.experts_implementation

    @property
    def compute_dtype(self) -> str | None:
        return self.model.compute_dtype

    @property
    def lr(self) -> float:
        return self.optimizer.learning_rate

    @property
    def weight_decay(self) -> float:
        return self.optimizer.weight_decay

    @property
    def betas(self) -> tuple[float, float]:
        return self.optimizer.betas

    @property
    def eps(self) -> float:
        return self.optimizer.eps

    @property
    def lr_scheduler_config(self) -> LRSchedulerConfig:
        return self.optimizer.lr_scheduler_config

    @property
    def global_batch_size(self) -> int:
        return self.training.global_batch_size

    @property
    def max_seq_len(self) -> int:
        return self.training.max_seq_len

    @property
    def steps(self) -> int:
        return self.training.steps

    @property
    def seed(self) -> int:
        return self.training.seed

    @property
    def deterministic(self) -> bool:
        return self.training.deterministic

    @property
    def pipeline_parallel_schedule(self) -> str:
        return self.parallel.pipeline_parallel_schedule

    @property
    def max_norm(self) -> float:
        return self.training.max_norm

    @property
    def dump_folder(self) -> str:
        return self.training.dump_folder

    @property
    def gc_freq(self) -> int:
        return self.training.gc_freq

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.training.gradient_accumulation_steps

    @property
    def checkpoint(self) -> CheckpointConfig:
        return self.training.checkpoint

    @property
    def metrics(self) -> MetricsConfig:
        return self.training.metrics

    @property
    def dataloader(self) -> DataloaderConfig:
        return self.training.dataloader

    @property
    def profiler(self) -> ProfilerConfig:
        return self.training.profiler

    @property
    def validation(self) -> ValidationConfig | None:
        return self.training.validation

    def derive_dp(self, world_size: int) -> int:
        """Flat passthrough so callers use cfg.derive_dp(world_size) uniformly."""
        return self.parallel.derive_dp(world_size)

    def auto_fill_model(self) -> None:
        """Fill model arch fields from a HF hub config when given a hub id.

        Called explicitly (not in __post_init__) so offline runs never hit the network.
        """
        mp = self.model.model_name_or_path
        if mp.count("/") != 1:
            return  # offline architecture name; explicit sizes are authoritative
        try:
            hf_config = AutoConfig.from_pretrained(mp)
        except Exception as e:  # noqa: BLE001 - warn, keep explicit values
            logger.warning("Could not load AutoConfig for '%s': %s", mp, e)
            return
        m = self.model
        m.vocab_size = getattr(hf_config, "vocab_size", m.vocab_size)
        m.hidden_size = getattr(hf_config, "hidden_size", m.hidden_size)
        m.intermediate_size = getattr(
            hf_config, "intermediate_size", m.intermediate_size
        )
        m.num_hidden_layers = getattr(
            hf_config, "num_hidden_layers", m.num_hidden_layers
        )
        m.num_attention_heads = getattr(
            hf_config, "num_attention_heads", m.num_attention_heads
        )
        m.num_key_value_heads = getattr(
            hf_config, "num_key_value_heads", m.num_key_value_heads
        )
