"""Qwen3 training -- the whole thing in a few readable lines.

The HuggingFace ecosystem end to end: ``AutoConfig`` builds a tiny Qwen3,
``AutoModelForCausalLM`` provides the model, and hpmesh's ``Trainer`` runs the
loop. No hand-written model code, no network.

Two things are worth reading here. The first is how little it takes to start: a
config, a Trainer, ``.train()``. The second is ``num_key_value_heads`` --
grouped-query attention, where the K/V projections are narrower than the query
projection because several query heads share each K/V head. It is the one
setting that makes this Qwen3 rather than a generic decoder, and
``AutoModelForCausalLM`` builds the shape on its own once the config says so.

Run it (from the repo root, with hpmesh installed or on the path):
    python -m examples.train_qwen3

Note the ``-m``: running the file by path (``python examples/train_qwen3.py``)
puts ``examples/`` on ``sys.path`` instead of the repo root, so ``import hpmesh``
fails. ``-m`` puts the repo root there, which is where the package lives.

or equivalently through the CLI, with the same sizes as flags:
    python -m hpmesh --model_name_or_path qwen3 --num_hidden_layers 2 \
        --num_attention_heads 4 --num_key_value_heads 2 --steps 20 \
        --max_seq_len 32 --global_batch_size 8

On a real corpus / real Qwen3 weights, point ``model_name_or_path`` at a hub id
(e.g. "Qwen/Qwen3-0.6B"): the Hub's config then carries the architecture and the
sizes below are ignored.
"""

from __future__ import annotations

from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from hpmesh.trainer.config import MetricsConfig
from hpmesh.trainer.trainer import Trainer


def qwen3_config() -> HybridMeshConfig:
    """A tiny offline Qwen3 (random init) for the single-device learning step."""
    return HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",  # offline: AutoConfig.for_model("qwen3", ...)
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,  # GQA: half the query heads share each K/V
        ),
        # Parallelism uses the torchtitan-spelled degree fields; -1 on the shard
        # degree means "derive from world_size" (all remaining ranks are DP).
        parallel=ParallelConfig(data_parallel_shard_size=-1),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=8,
            max_seq_len=32,
            steps=20,
            seed=42,
            deterministic=True,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def main() -> None:
    Trainer(qwen3_config()).train()


if __name__ == "__main__":
    main()
