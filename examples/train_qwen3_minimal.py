"""Minimal Qwen3 training -- the whole thing in a few readable lines.

Uses the HuggingFace ecosystem end to end: ``AutoConfig`` builds a tiny Qwen3,
``AutoModelForCausalLM`` provides the model, and hpmesh's ``Trainer`` runs the
loop. No hand-written model code.

Run it:
    python examples/train_qwen3_minimal.py

or equivalently through the CLI:
    python -m hpmesh --model_name_or_path qwen3 --steps 20 --max_seq_len 32

On a real corpus / real Qwen3 weights, point model_name_or_path at a hub id
(e.g. "Qwen/Qwen3-0.6B") instead of the offline "qwen3" architecture name.
"""

from __future__ import annotations

from hpmesh.trainer import (
    HybridMeshConfig,
    ModelArguments,
    OptimizerArguments,
    ParallelArguments,
    TrainingArguments,
)
from hpmesh.trainer.trainer import Trainer


def qwen3_minimal_config() -> HybridMeshConfig:
    """A tiny offline Qwen3 (random init) for the single-device learning step."""
    return HybridMeshConfig(
        model=ModelArguments(
            model_name_or_path="qwen3",  # offline: AutoConfig.for_model("qwen3", ...)
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,  # GQA; set < num_attention_heads to see it
        ),
        # Parallelism uses the torchtitan-spelled degree fields; -1 on the shard
        # degree means "derive from world_size" (all remaining ranks are DP).
        parallel=ParallelArguments(data_parallel_shard_degree=-1),
        optimizer=OptimizerArguments(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingArguments(
            global_batch_size=8,
            max_seq_len=32,
            steps=20,
            seed=42,
            deterministic=True,
            log_freq=1,
        ),
    )


def main() -> None:
    Trainer(qwen3_minimal_config()).train()


if __name__ == "__main__":
    main()
