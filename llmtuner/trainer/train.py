"""Console entry point: parse the config groups and run the Trainer.

Deciding which group a config belongs to

HfArgumentParser is given the config GROUPS (not the composed config), so each
field becomes a clean flat CLI flag (--steps, --data_parallel_shard_size,
--learning_rate, --dump_folder, ...) and each group runs its own __post_init__
validation. The nine groups live in ``llmtuner.config`` and are composed into the
single LLMTunerConfig by ``LLMTunerConfig.from_groups``, which owns the
graft table (which nested group lands on which top-level group). A YAML/JSON
file can also be passed positionally.

Grafting happens after parsing, so a nested config's fields still reach the user
as bare flags: --enable, --interval, --log_freq, --dataset, --profile_freq.
HfArgumentParser has no way to prefix one group's fields, and llmtuner's top-level
groups are flat too (--tensor_parallel_size, --learning_rate).

Single process (step 0):
    python -m llmtuner --steps 20
    # or, after `pip install -e .`:  llmtuner-train --steps 20

Data parallel, 2 ranks (step 1):
    torchrun --nproc_per_node=2 -m llmtuner --data_parallel_shard_size -1

Checkpointing (disabled unless --enable is passed):
    python -m llmtuner --steps 20 --enable --interval 10 --dump_folder ./outputs

Metrics (stdout always; TensorBoard and WandB are opt-in):
    python -m llmtuner --steps 20 --enable_tensorboard
    python -m llmtuner --steps 20 --enable_wandb --tag baseline

Data (synthetic random tokens unless a corpus is named):
    python -m llmtuner --steps 20 --dataset local_jsonl \
        --dataset_path ./corpus.jsonl --tokenizer_path ./tokenizer

Profiling (off unless enabled; both write under --dump_folder):
    python -m llmtuner --steps 20 --enable_profiling --profile_freq 4
    python -m llmtuner --steps 20 --enable_memory_snapshot --memory_snapshot_freq 5
"""

from __future__ import annotations

import os

from transformers import HfArgumentParser

from llmtuner.config import (
    CheckpointConfig,
    DataloaderConfig,
    LLMTunerConfig,
    LRSchedulerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    ProfilerConfig,
    TrainingConfig,
)
from llmtuner.errors import ConfigError

from .trainer import Trainer

# Parser order is significant: parse_args_into_dataclasses() returns one
# instance per class in this exact order.
_GROUP_CLASSES = (
    ModelConfig,
    ParallelConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    TrainingConfig,
    CheckpointConfig,
    DataloaderConfig,
    MetricsConfig,
    ProfilerConfig,
)


def parse_config() -> LLMTunerConfig:
    parser = HfArgumentParser(list(_GROUP_CLASSES))
    # Each group is its own parser group, so every scalar field becomes a flag.
    parsed = dict(
        zip(_GROUP_CLASSES, parser.parse_args_into_dataclasses(), strict=True)
    )
    cfg = LLMTunerConfig.from_groups(
        model=parsed[ModelConfig],
        parallel=parsed[ParallelConfig],
        optimizer=parsed[OptimizerConfig],
        lr_scheduler=parsed[LRSchedulerConfig],
        training=parsed[TrainingConfig],
        checkpoint=parsed[CheckpointConfig],
        dataloader=parsed[DataloaderConfig],
        metrics=parsed[MetricsConfig],
        profiler=parsed[ProfilerConfig],
    )
    cfg.auto_fill_model()  # pull arch from a HF hub id when given one (no-op offline)
    return cfg


def main() -> None:
    cfg = parse_config()
    trainer = Trainer(cfg)
    if cfg.checkpoint.create_seed_checkpoint:
        # Mirrors torchtitan ``train.py``: a seed checkpoint is the unsharded
        # step-0 model, so it must be written from a single process (any
        # sharding would bake one rank's shard layout into the artifact), and
        # loading treats step-0 as model-only (see ``checkpointer/base.py``).
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ConfigError(
                "Must create a seed checkpoint using a single device, to "
                "disable sharding."
            )
        if not cfg.checkpoint.enable:
            raise ConfigError(
                "Must enable checkpointing when creating a seed checkpoint."
            )
        try:
            if trainer.checkpointer.save(curr_step=0, last_step=True):
                print("Created seed checkpoint at step 0")
        finally:
            trainer.close()
        return
    trainer.train()


if __name__ == "__main__":
    main()
