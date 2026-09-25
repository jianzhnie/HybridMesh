"""hpmesh configuration: every dataclass a run is described by.

This is the one place configuration lives, and every class here is named the
same way -- ``*Config``. What varies is only whether a config is a top-level
parser group or a nested one:

* Top-level groups, handed to ``HfArgumentParser`` in ``train.py`` --
  ``ModelConfig``, ``ParallelConfig``, ``OptimizerConfig``, ``TrainingConfig``.
* Nested configs, each also parsed as its own group and then grafted onto the
  group that holds it -- ``CheckpointConfig``, ``DataloaderConfig``,
  ``MetricsConfig``, ``ProfilerConfig``, ``LRSchedulerConfig``.

These are descriptions, not builders. Nothing here constructs the runtime object
it describes -- the loader in ``datasets/build.py``, the scheduler in
``components/optimizer/lr_scheduler.py`` -- for two reasons. A config that
carries its own builder suggests the built object is one of its fields, and it
is not: building takes arguments the config does not have (which rank am I, how
many tokens per batch). And a builder has to name the type it produces, which
for a config nested under a component would mean importing the component into
this module while the component imports this one back.

So the seam is a factory function that takes the config as its first argument.
``trainer/trainer.py`` calls those; this module is only ever read.

A component -- the checkpointer, the metrics processor, the profiler, the
learning-rate schedule -- does not define its own config class next to itself
either; it takes an instance from here and reads fields off it. It names the
type only under ``TYPE_CHECKING``, for the same cycle reason.

Design notes (see docs/hybridmesh_design.md): grouped by concern, then COMPOSED
-- not mixed in via multiple inheritance -- so each group's ``__post_init__``
validation runs automatically via ``default_factory``, with no fragile manual
chaining. The single entry point is ``HybridMeshConfig``.

Kept deliberately small for the learning path: one optimizer (adamw), one LR
schedule, deterministic seeding. Add knobs only when a learning step needs them.

The classes live one domain per file under this package
(``model.py`` / ``parallel.py`` / ...); everything is re-exported here so
``from hpmesh.config import X`` resolves for every public name.
"""

from hpmesh.config.checkpoint import CheckpointConfig
from hpmesh.config.data import DataloaderConfig
from hpmesh.config.model import ModelConfig
from hpmesh.config.optimizer import (
    EMAConfig,
    LRSchedulerConfig,
    OptimizerConfig,
    ParamGroupConfig,
)
from hpmesh.config.parallel import ParallelConfig
from hpmesh.config.root import HybridMeshConfig
from hpmesh.config.training import (
    CompileConfig,
    MemoryBudgetACConfig,
    MetricsConfig,
    ProfilerConfig,
    SelectiveACConfig,
    TrainingConfig,
    ValidationConfig,
)

__all__ = [
    "CheckpointConfig",
    "CompileConfig",
    "DataloaderConfig",
    "EMAConfig",
    "ValidationConfig",
    "HybridMeshConfig",
    "LRSchedulerConfig",
    "MemoryBudgetACConfig",
    "MetricsConfig",
    "ModelConfig",
    "OptimizerConfig",
    "ParallelConfig",
    "ParamGroupConfig",
    "ProfilerConfig",
    "SelectiveACConfig",
    "TrainingConfig",
]
