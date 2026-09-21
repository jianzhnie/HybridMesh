"""EP>1 with gradient clipping must refuse loudly, not clip to a wrong norm.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/ep_clip_rejection.py

hpmesh's EP physically partitions experts across the ep ranks; the expert
parameters are plain (or efsdp-sharded) tensors, never DTensors on an "ep"
mesh axis -- the premise torchtitan's ``_clip_grad_norm_with_ep`` asserts. The
dense ``clip_grad_norm_`` would miss the cross-EP sum of expert-gradient
norms, so the Trainer raises instead of clipping against a partial norm.
"""

from __future__ import annotations

import torch.distributed as dist

from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)
from hpmesh.trainer.trainer import Trainer


def _cfg(max_norm: float) -> HybridMeshConfig:
    return HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(expert_parallel_size=2, backend="gloo"),
        training=TrainingConfig(
            global_batch_size=8, max_seq_len=32, steps=1, max_norm=max_norm
        ),
    )


def main() -> None:
    failures: list[str] = []
    try:
        # Trainer.__init__ initializes the process group itself; the rejection
        # must fire before the model is built.
        Trainer(_cfg(max_norm=1.0))
        failures.append("ep=2 with max_norm=1.0 built a Trainer instead of raising")
    except NotImplementedError as ex:
        if "grad" not in str(ex):
            failures.append(f"unexpected message: {ex}")

    import torch

    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")
    dist.destroy_process_group()
    assert local_ok.item() == 0, "EP clip rejection check failed"


if __name__ == "__main__":
    main()
