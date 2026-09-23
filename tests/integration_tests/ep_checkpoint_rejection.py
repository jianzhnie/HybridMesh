"""EP>1 checkpointing must reject before model construction.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/ep_checkpoint_rejection.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)
from hpmesh.trainer.config import CheckpointConfig
from hpmesh.trainer.trainer import Trainer


def main() -> None:
    failures: list[str] = []
    cfg = HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(expert_parallel_size=2),
        training=TrainingConfig(
            global_batch_size=8,
            max_seq_len=32,
            steps=1,
            checkpoint_config=CheckpointConfig(enable=True),
        ),
    )
    try:
        Trainer(cfg)
        failures.append("ep=2 with checkpointing built a Trainer instead of raising")
    except NotImplementedError as ex:
        if "checkpoint" not in str(ex):
            failures.append(f"unexpected message: {ex}")

    device_type = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_ok = torch.tensor(
        [0.0 if not failures else 1.0],
        device=f"{device_type}:{local_rank}" if device_type != "cpu" else "cpu",
    )
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        for failure in failures:
            print(f"  FAIL {failure}")
        if not failures:
            print("all checks passed")
    dist.destroy_process_group()
    assert local_ok.cpu().item() == 0, "EP checkpoint rejection check failed"


if __name__ == "__main__":
    main()
