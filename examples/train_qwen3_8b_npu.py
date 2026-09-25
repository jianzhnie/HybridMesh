"""Train the local pretrained Qwen3-8B checkpoint on Ascend NPUs.

The architecture is read from the local Hugging Face directory:

    /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B

The five safetensors shards are loaded through DCP after FSDP has established
the distributed parameter layout. Key coverage is checked strictly before any
training step, so this cannot silently fall back to random weights.

Launch through ``examples/train_qwen3_8b_npu.sh``. The default topology is
8-way FSDP with full activation checkpointing and BF16 model construction.
"""

from __future__ import annotations

import os

import torch

from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from hpmesh.config import CheckpointConfig, DataloaderConfig, MetricsConfig
from hpmesh.trainer.trainer import Trainer

MODEL_PATH = "/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B"
DATASET_PATH = (
    "/home/jianzhnie/llmtuner/hfhub/datasets/EleutherAI/"
    "hendrycks_math/train.jsonl"
)


def qwen3_8b_npu_config() -> HybridMeshConfig:
    model_path = os.environ.get("HPMESH_QWEN3_8B_PATH", MODEL_PATH)
    dataset_path = os.environ.get("HPMESH_DATASET_PATH", DATASET_PATH)
    save_checkpoint = os.environ.get("HPMESH_SAVE_CHECKPOINT", "1") == "1"
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(f"Qwen3-8B config.json not found under {model_path}")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"training dataset not found: {dataset_path}")

    return HybridMeshConfig(
        model=ModelConfig(model_name_or_path=model_path),
        parallel=ParallelConfig(
            data_parallel_shard_size=-1,
        ),
        optimizer=OptimizerConfig(
            learning_rate=1e-5,
            weight_decay=0.1,
        ),
        training=TrainingConfig(
            global_batch_size=int(os.environ.get("HPMESH_GLOBAL_BATCH_SIZE", "8")),
            max_seq_len=int(os.environ.get("HPMESH_MAX_SEQ_LEN", "4096")),
            steps=int(os.environ.get("HPMESH_STEPS", "100")),
            seed=42,
            deterministic=False,
            activation_checkpoint_mode="full",
            checkpoint_config=CheckpointConfig(
                enable=True,
                load_only=not save_checkpoint,
                initial_load_path=model_path,
                initial_load_model_only=True,
                initial_load_in_hf=True,
                # This is a training checkpoint, not an export artifact: keep
                # optimizer, scheduler, dataloader, and counters so the final
                # step can be resumed safely.
                last_save_model_only=False,
            ),
            dataloader_config=DataloaderConfig(
                dataset="local_jsonl_sft",
                dataset_path=dataset_path,
                tokenizer_path=model_path,
                prompt_field="problem",
                response_field="solution",
                packing="first_fit",
                max_num_documents=1,
            ),
            metrics_config=MetricsConfig(log_freq=1),
            dump_folder=os.environ.get(
                "HPMESH_DUMP_FOLDER",
                "/home/jianzhnie/llmtuner/llm/HybridMesh/outputs/qwen3-8b-npu",
            ),
        ),
    )


def main() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError(
            "This example requires torch-npu and a visible Ascend NPU. "
            "Run it through examples/train_qwen3_8b_npu.sh."
        )
    # Construct the model directly in BF16. Building Qwen3-8B in FP32 first
    # would transiently require roughly twice the intended parameter memory on
    # every rank before FSDP has a chance to shard it.
    torch.set_default_dtype(torch.bfloat16)
    Trainer(qwen3_8b_npu_config()).train()


if __name__ == "__main__":
    main()
