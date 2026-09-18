"""Console entry point: parse the config groups and run the Trainer.

HfArgumentParser is given the four argument GROUPS (not the composed config), so
each field becomes a clean flat CLI flag (--steps, --dp, --learning_rate, ...)
and each group runs its own __post_init__ validation. We then compose them into
the single HybridMeshConfig. A YAML/JSON file can also be passed positionally.

Single process (step 0):
    python -m hpmesh --steps 20
    # or, after `pip install -e .`:  hpmesh-train --steps 20

Data parallel, 2 ranks (step 1):
    torchrun --nproc_per_node=2 -m hpmesh --dp 2
"""

from __future__ import annotations

from transformers import HfArgumentParser

from .config import (
    HybridMeshConfig,
    ModelArguments,
    OptimizerArguments,
    ParallelArguments,
    TrainingArguments,
)
from .trainer import Trainer


def parse_config() -> HybridMeshConfig:
    parser = HfArgumentParser(
        [ModelArguments, ParallelArguments, OptimizerArguments, TrainingArguments]
    )
    model, parallel, optimizer, training = parser.parse_args_into_dataclasses()
    cfg = HybridMeshConfig(
        model=model, parallel=parallel, optimizer=optimizer, training=training
    )
    cfg.auto_fill_model()  # pull arch from a HF hub id when given one (no-op offline)
    return cfg


def main() -> None:
    Trainer(parse_config()).train()


if __name__ == "__main__":
    main()
