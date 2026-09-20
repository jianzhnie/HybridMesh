# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The data pipeline: a Grain dataset graph plus the loaders that drive it.

The public surface is the substrate below. The concrete dataset catalogs
(``datasets.hf.text``, ``datasets.hf.multimodal``) are deliberately not
re-exported: they pull in optional dependencies -- torchvision, and a video
backend behind them -- that a run using none of them should not have to install.
"""

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
    "build_dataset_iteration_policy",
]
