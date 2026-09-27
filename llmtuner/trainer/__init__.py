"""Trainer subpackage: the training loop and CLI entry point.

The config re-exports below are a compatibility alias kept for existing
callers; the canonical path is ``llmtuner.config`` (and ``llmtuner.LLMTunerConfig``
at the package root). ``Trainer`` lives at the root as ``llmtuner.Trainer``
(lazy); the CLI entry point is ``llmtuner.trainer.train:main``.
"""

from llmtuner.config import (
    DataloaderConfig,
    LLMTunerConfig,
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)

__all__ = [
    "DataloaderConfig",
    "LLMTunerConfig",
    "LRSchedulerConfig",
    "ModelConfig",
    "OptimizerConfig",
    "ParallelConfig",
    "TrainingConfig",
]
