"""Console entry point: parse the config groups and run the Trainer.

Deciding which group a config belongs to

HfArgumentParser is given the config GROUPS (not the composed config), so each
field becomes a clean flat CLI flag (--steps, --data_parallel_shard_size,
--learning_rate, --dump_folder, ...) and each group runs its own __post_init__
validation. We then compose them into the single HybridMeshConfig. A YAML/JSON
file can also be passed positionally.

All ten classes in ``config.py`` are named ``*Config`` and each is parsed as its
own group here. Five of them are nested inside another config (reachable as
``cfg.training.checkpoint`` and friends), so which one they graft onto is a
deliberate choice, and this is the table of it:

  model          ModelConfig            top level
  parallel       ParallelConfig         top level
  optimizer      OptimizerConfig        top level (LRSchedulerConfig grafts here)
  training       TrainingConfig         top level (the rest graft here)
                 CheckpointConfig
                 DataloaderConfig
                 MetricsConfig
                 ProfilerConfig

Grafting happens after parsing, so a nested config's fields still reach the user
as bare flags: --enable, --interval, --log_freq, --dataset, --profile_freq.
HfArgumentParser has no way to prefix one group's fields, and hpmesh's top-level
groups are flat too (--tensor_parallel_size, --learning_rate).

Single process (step 0):
    python -m hpmesh --steps 20
    # or, after `pip install -e .`:  hpmesh-train --steps 20

Data parallel, 2 ranks (step 1):
    torchrun --nproc_per_node=2 -m hpmesh --data_parallel_shard_size -1

Checkpointing (disabled unless --enable is passed):
    python -m hpmesh --steps 20 --enable --interval 10 --dump_folder ./outputs

Metrics (stdout always; TensorBoard and WandB are opt-in):
    python -m hpmesh --steps 20 --enable_tensorboard
    python -m hpmesh --steps 20 --enable_wandb --tag baseline

Data (synthetic random tokens unless a corpus is named):
    python -m hpmesh --steps 20 --dataset local_jsonl \
        --dataset_path ./corpus.jsonl --tokenizer_path ./tokenizer

Profiling (off unless enabled; both write under --dump_folder):
    python -m hpmesh --steps 20 --enable_profiling --profile_freq 4
    python -m hpmesh --steps 20 --enable_memory_snapshot --memory_snapshot_freq 5
"""

from __future__ import annotations

import os

from transformers import HfArgumentParser

from hpmesh.config import (
    CheckpointConfig,
    DataloaderConfig,
    HybridMeshConfig,
    LRSchedulerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    ProfilerConfig,
    TrainingConfig,
)

from .trainer import Trainer


def parse_config() -> HybridMeshConfig:
    parser = HfArgumentParser(
        [
            ModelConfig,
            ParallelConfig,
            OptimizerConfig,
            LRSchedulerConfig,
            TrainingConfig,
            CheckpointConfig,
            DataloaderConfig,
            MetricsConfig,
            ProfilerConfig,
        ]
    )
    (
        model_config,
        parallel_config,
        optimizer_config,
        lr_scheduler_config,
        training_config,
        checkpoint_config,
        dataloader_config,
        metrics_config,
        profiler_config,
    ) = parser.parse_args_into_dataclasses()
    # Each group is its own parser group, so every scalar field becomes a flag.
    # The nested configs (reachable as ``training.checkpoint`` and friends) are
    # grafted on here; their __post_init__ already ran as part of the parser's
    # construction. The schedule grafts onto the OPTIMIZER group, not onto
    # ``training``: it scales the learning rate that group sets, and splitting
    # them would let a run halve one without touching the other.
    optimizer_config.lr_scheduler_config = lr_scheduler_config
    training_config.checkpoint_config = checkpoint_config
    training_config.dataloader_config = dataloader_config
    training_config.metrics_config = metrics_config
    training_config.profiler_config = profiler_config
    cfg = HybridMeshConfig(
        model=model_config,
        parallel=parallel_config,
        optimizer=optimizer_config,
        training=training_config,
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
            raise RuntimeError(
                "Must create a seed checkpoint using a single device, to "
                "disable sharding."
            )
        if not cfg.checkpoint.enable:
            raise RuntimeError(
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
