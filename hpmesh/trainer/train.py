"""Console entry point: parse the config groups and run the Trainer.

HfArgumentParser is given the argument GROUPS (not the composed config), so each
field becomes a clean flat CLI flag (--steps, --data_parallel_shard_degree,
--learning_rate, --dump_folder, ...) and each group runs its own __post_init__
validation. We then compose them into the single HybridMeshConfig. A YAML/JSON
file can also be passed positionally.

The checkpoint options are their own group, so every one of the manager
Config's fields becomes a flag and Config.__post_init__ validates the parsed
values. The flags keep their bare field names (--enable, --interval, --folder,
--load_step, --keep_latest_k, ...) because that is what HfArgumentParser derives
them from -- it has no way to prefix one group's fields, and hpmesh's other
groups are flat too (--backend, --learning_rate).

Single process (step 0):
    python -m hpmesh --steps 20
    # or, after `pip install -e .`:  hpmesh-train --steps 20

Data parallel, 2 ranks (step 1):
    torchrun --nproc_per_node=2 -m hpmesh --data_parallel_shard_degree -1

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

from transformers import HfArgumentParser

from .config import (
    CheckpointArguments,
    DataloaderArguments,
    HybridMeshConfig,
    MetricsArguments,
    ModelArguments,
    OptimizerArguments,
    ParallelArguments,
    ProfilerArguments,
    TrainingArguments,
)
from .trainer import Trainer


def parse_config() -> HybridMeshConfig:
    parser = HfArgumentParser(
        [
            ModelArguments,
            ParallelArguments,
            OptimizerArguments,
            TrainingArguments,
            CheckpointArguments,
            DataloaderArguments,
            MetricsArguments,
            ProfilerArguments,
        ]
    )
    (
        model,
        parallel,
        optimizer,
        training,
        checkpoint,
        dataloader,
        metrics,
        profiler,
    ) = parser.parse_args_into_dataclasses()
    # Each group is its own parser group, so every scalar field becomes a flag.
    # The nested configs (reachable as ``training.checkpoint`` and friends) are
    # grafted on here; their __post_init__ already ran as part of the parser's
    # construction.
    training.checkpoint_config = checkpoint
    training.dataloader_config = dataloader
    training.metrics_config = metrics
    training.profiler_config = profiler
    cfg = HybridMeshConfig(
        model=model, parallel=parallel, optimizer=optimizer, training=training
    )
    cfg.auto_fill_model()  # pull arch from a HF hub id when given one (no-op offline)
    return cfg


def main() -> None:
    Trainer(parse_config()).train()


if __name__ == "__main__":
    main()
