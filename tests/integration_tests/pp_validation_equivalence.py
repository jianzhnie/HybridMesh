"""PP x validation check: the schedule's eval driver runs the pass.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \\
        tests/integration_tests/pp_validation_equivalence.py

A tiny qwen3 trains under pp=2 with validation enabled (freq=1, steps=1).
Every train step is followed by a validation pass driven through
``schedule.eval`` (upstream's Validator seam). The check pins: the pass runs
on both ranks without hanging, the logged validation loss is finite and
positive, and the model returns to train mode afterwards.

Environment note: needs torch >= 2.12 (the pipelining surface; the eval
driver is gated by the ``pipelining_schedule_eval`` capability). Written
2026-10-02 with the PP x validation unlock; environment not covered on the
development host -- awaiting a multi-rank run on the target torch.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from llmtuner.config import MetricsConfig, ValidationConfig
from llmtuner.trainer import (
    LLMTunerConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.trainer.trainer import Trainer

STEPS = 2
MICROBATCHES = 2
GLOBAL_BATCH = 4
SEQ = 32
VOCAB = 128
PP = 2
WORLD = 2


def _cfg() -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=PP,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=MICROBATCHES,
            data_parallel_shard_size=-1,
        ),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            max_seq_len=SEQ,
            steps=STEPS,
            seed=42,
            deterministic=True,
            metrics_config=MetricsConfig(log_freq=1),
            validation_config=ValidationConfig(freq=1, steps=1),
        ),
    )


def main() -> None:
    cfg = _cfg()
    failures: list[str] = []

    trainer = Trainer(cfg)
    rank = trainer.rank
    assert trainer.world_size == WORLD, (
        f"this check assumes {WORLD} ranks, got {trainer.world_size}"
    )

    # Capture the validation reports through the metrics seam.
    reported: list[float] = []
    original = trainer.metrics.log_validation

    def _capture(loss: float, step: int) -> None:
        reported.append(loss)
        original(loss=loss, step=step)

    trainer.metrics.log_validation = _capture

    data_iterator = trainer.data_iterator()
    for _ in range(STEPS):
        trainer.step += 1
        trainer.train_step(data_iterator)
        if trainer.should_validate(trainer.step):
            trainer.validate(trainer.step)
    trainer.close()

    # The pass ran once per step (freq=1), on the rank(s) that hold the loss.
    if trainer.pp_has_last_stage:
        if len(reported) != STEPS:
            failures.append(
                f"rank {rank}: {len(reported)} validation reports, want {STEPS}"
            )
        if any(not (0.0 < loss < float("inf")) for loss in reported):
            failures.append(f"rank {rank}: non-finite validation loss {reported}")
    if any(not part.training for part in trainer.model_parts):
        failures.append(f"rank {rank}: a model part is still in eval mode")

    if rank == 0:
        print(f"pp={PP} validation freq=1 steps={STEPS}")
        print(f"reported losses = {reported}")
        for f in failures:
            print(f"  FAIL {f}")
        print("all checks passed" if not failures else "CHECKS FAILED")
    verdict = torch.tensor(len(failures), dtype=torch.int64)
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    assert int(verdict) == 0, f"{int(verdict)} check(s) failed -- see rank 0 output"


if __name__ == "__main__":
    main()
