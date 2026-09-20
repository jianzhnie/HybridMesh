"""Trainer subpackage: configuration dataclasses (and the training loop/CLI).

Only the config is re-exported here. ``Trainer`` and the CLI entry point live in
``hpmesh.trainer.trainer`` / ``hpmesh.trainer.train`` and are imported lazily to
avoid a circular import with the model layer.
"""

from .config import (
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
