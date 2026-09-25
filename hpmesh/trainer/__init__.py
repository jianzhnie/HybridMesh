"""Trainer subpackage: the training loop and CLI entry point.

The config re-exports below are a compatibility alias kept for existing
callers; the canonical path is ``hpmesh.config`` (and ``hpmesh.HybridMeshConfig``
at the package root). ``Trainer`` lives at the root as ``hpmesh.Trainer``
(lazy); the CLI entry point is ``hpmesh.trainer.train:main``.
"""

from hpmesh.config import (
    DataloaderConfig,
    HybridMeshConfig,
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)

__all__ = [
    "DataloaderConfig",
    "HybridMeshConfig",
    "LRSchedulerConfig",
    "ModelConfig",
    "OptimizerConfig",
    "ParallelConfig",
    "TrainingConfig",
]
