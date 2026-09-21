"""The data pipeline: a Grain dataset graph plus the loaders that drive it.

The public surface is the substrate below plus ``build_dataloader``, which
assembles them from a ``trainer.config.DataloaderConfig``. The concrete dataset
catalogs
(``datasets.hf.text``, ``datasets.hf.multimodal``) are deliberately not
re-exported: they pull in optional dependencies -- torchvision, and a video
backend behind them -- that a run using none of them should not have to install.
"""

from .build import build_dataloader
from .collators import HAS_PIN_MEMORY, Collator, TextCollator, TrainerBatch
from .dataset import (
    DatasetConcatConfig,
    DatasetConfig,
    DatasetMixConfig,
    GrainDataset,
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
    WeightedDataset,
)
from .loader import (
    BaseDataLoader,
    DataloaderExhaustedError,
    GrainDataLoader,
    GrainDataLoaderConfig,
    build_dataset_iteration_policy,
)
from .packing import ConcatThenSplitPackingConfig, FirstFitPackingConfig
from .sources import (
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
    IndexedJsonlSource,
    RandomAccessDataSource,
    SourceConfig,
)
from .types import DatasetBuildContext, DatasetIterationPolicy

__all__ = [
    "BaseDataLoader",
    "Collator",
    "ConcatThenSplitPackingConfig",
    "DatasetBuildContext",
    "DatasetConcatConfig",
    "DatasetConfig",
    "DatasetIterationPolicy",
    "DatasetMixConfig",
    "DataloaderExhaustedError",
    "FirstFitPackingConfig",
    "GrainDataLoader",
    "GrainDataLoaderConfig",
    "GrainDataset",
    "HAS_PIN_MEMORY",
    "HuggingFaceRandomAccessSource",
    "HuggingFaceStreamingSource",
    "IndexedJsonlSource",
    "RandomAccessDataSource",
    "SampleProcessor",
    "SingleDatasetConfig",
    "SourceConfig",
    "TextCollator",
    "TextSequence",
    "TrainerBatch",
    "WeightedDataset",
    "build_dataloader",
    "build_dataset_iteration_policy",
]
