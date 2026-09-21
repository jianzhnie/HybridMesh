"""Chunked loss with pipeline parallelism must refuse loudly, not silently drift.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/chunked_loss_pp_rejection.py

Under PP the last stage's loss is computed inside the schedule
(``pipeline_parallel/pp.py:_scalar_loss_fn``) on materialized logits; the
chunked path needs hidden states plus a per-chunk backward, which that seam
does not carry. Building a Trainer with both must raise NotImplementedError
before the model is built, rather than run an un-chunked (or wrong) loss.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)
from hpmesh.trainer.trainer import Trainer


def _cfg() -> HybridMeshConfig:
    return HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="llama",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(pipeline_parallel_size=2, backend="gloo"),
        training=TrainingConfig(
            global_batch_size=8,
            max_seq_len=32,
            steps=1,
            chunked_loss_num_chunks=2,
        ),
    )


def main() -> None:
    failures: list[str] = []
    try:
        # Trainer.__init__ initializes the process group itself; the rejection
        # must fire before the model is built.
        Trainer(_cfg())
        failures.append("pp=2 with chunked_loss_num_chunks=2 built a Trainer")
    except NotImplementedError as ex:
        if "pipeline" not in str(ex):
            failures.append(f"unexpected message: {ex}")

    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")
    dist.destroy_process_group()
    assert local_ok.item() == 0, "chunked loss PP rejection check failed"


if __name__ == "__main__":
    main()
