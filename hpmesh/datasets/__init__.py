"""The data pipeline: a Grain dataset graph plus the loaders that drive it.

The public surface is the substrate below plus ``build_dataloader``, which
assembles them from a ``hpmesh.config.DataloaderConfig``. The concrete dataset
catalogs (``datasets.text.text``, ``datasets.multimodal.mm_datasets``) are
deliberately not re-exported: they pull in optional dependencies -- torchvision,
and a video backend behind them -- that a run using none of them should not have
to install. Neither subpackage's ``__init__`` imports its children, for the same
reason; ``build_dataloader`` is the only entry point that decides.
"""

from .build import build_dataloader
from .collators import HAS_PIN_MEMORY, Collator, TextCollator, TrainerBatch
from .dataset import (
    DatasetConcat,
    DatasetMix,
    GrainDataset,
    SampleProcessor,
    SingleDataset,
    TextSequence,
    WeightedDataset,
    build_dataset,
)
from .loader import BaseDataLoader, DataloaderExhaustedError, GrainDataLoader
from .packing import (
    build_concat_then_split_packing,
    build_first_fit_packing,
)
from .sources import (
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
    IndexedJsonlSource,
    RandomAccessDataSource,
    build_source,
)
from .types import DatasetBuildContext, DatasetIterationPolicy

__all__ = [
    "BaseDataLoader",
    "Collator",
    "DatasetBuildContext",
    "DatasetConcat",
    "DatasetIterationPolicy",
    "DatasetMix",
    "DataloaderExhaustedError",
    "GrainDataLoader",
    "GrainDataset",
    "HAS_PIN_MEMORY",
    "HuggingFaceRandomAccessSource",
    "HuggingFaceStreamingSource",
    "IndexedJsonlSource",
    "RandomAccessDataSource",
    "SampleProcessor",
    "SingleDataset",
    "TextCollator",
    "TextSequence",
    "TrainerBatch",
    "WeightedDataset",
    "build_concat_then_split_packing",
    "build_dataloader",
    "build_dataset",
    "build_first_fit_packing",
    "build_source",
]
