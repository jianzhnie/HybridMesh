"""Loss reporting must not gate its collectives on a local token count.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/loss_report_rank_consistency.py

The reported ``loss``/``max_loss`` are reduced over the loss mesh at logging
time. The reduction used to be skipped on any rank whose accumulation window
held zero valid tokens -- a *local* predicate gating a *collective*, so one
such rank in a DP group would leave the others inside ``all_reduce`` forever.
Here rank 1's batch has every label masked (zero valid tokens) while rank 0's
is normal; both ranks must come back with the same reported numbers.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from hpmesh.components.loss import IGNORE_INDEX
from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from hpmesh.trainer.config import MetricsConfig
from hpmesh.trainer.trainer import Trainer

SEQ = 32
VOCAB = 128


def _cfg() -> HybridMeshConfig:
    return HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(data_parallel_shard_size=-1),
        optimizer=OptimizerConfig(learning_rate=1e-4),
        training=TrainingConfig(
            global_batch_size=8,
            max_seq_len=SEQ,
            steps=1,
            seed=42,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def main() -> None:
    # Trainer.__init__ initializes the process group (gloo, per the config).
    trainer = Trainer(_cfg())
    rank = trainer.rank
    world = trainer.world_size
    assert world == 2, f"this check assumes 2 ranks, got {world}"
    assert dist.is_initialized()

    g = torch.Generator().manual_seed(0)
    inputs = torch.randint(0, VOCAB, (8 * SEQ,), generator=g)
    labels = torch.randint(0, VOCAB, (8 * SEQ,), generator=g)
    if rank == 1:
        # This rank's whole window predicts nothing.
        labels = torch.full_like(labels, IGNORE_INDEX)
    batch = {
        "input": inputs,
        "labels": labels,
        "num_valid_tokens": int((labels != IGNORE_INDEX).sum()),
    }

    trainer.step += 1
    # On the broken gating this call never returns: rank 1 skips the loss-mesh
    # collectives that rank 0 enters.
    metrics = trainer.train_step(iter([batch]))
    assert metrics is not None  # log_freq=1: every step reports

    failures: list[str] = []
    gathered = [None] * world
    dist.all_gather_object(gathered, metrics)
    if not all(m == gathered[0] for m in gathered):
        failures.append(f"metrics disagree across ranks: {gathered}")
    if gathered[0]["loss"] <= 0:
        failures.append(f"rank 0 carried the whole loss; got {gathered[0]}")

    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"dp={world}, rank 1 holds zero valid tokens")
        print(f"reported metrics = {gathered[0]}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    dist.barrier()
    trainer.checkpointer.close()
    trainer.metrics.close()
    dist.destroy_process_group()
    assert local_ok.item() == 0, "loss-report rank-consistency check failed"


if __name__ == "__main__":
    main()
