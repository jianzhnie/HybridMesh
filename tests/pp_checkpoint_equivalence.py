"""PP checkpoint round-trip: a resumed pp=2 run must match the uninterrupted one.

Run under torchrun with 2 ranks, in two phases that share a workdir:

    D=$(mktemp -d)
    torchrun --nproc_per_node=2 tests/pp_checkpoint_equivalence.py full "$D"
    torchrun --nproc_per_node=2 tests/pp_checkpoint_equivalence.py resume "$D"

Phase ``full`` trains a tiny offline qwen3 for N+M optimizer steps with
``pp=2`` (schedule 1F1B, 4 micro-batches per step) through the real
``Trainer``, saving a full DCP checkpoint -- model, FQN-keyed optimizer state,
and the trainer's step/token counters -- at step N, and records the
uninterrupted loss trajectory to the workdir.

Phase ``resume`` builds a brand-new Trainer against the same workdir, loads
the step-N checkpoint, and continues for M steps. The resumed losses must
match the uninterrupted run's last M losses BITWISE: the optimizer's Adam
moments enter the update, so a checkpoint that restored weights but not
moments (or moments keyed to the wrong stage's parameters) would diverge
visibly, while a correct round-trip reproduces the uninterrupted run's
arithmetic exactly.

The FQN keying is what pp > 1 exercises that a single-rank round-trip does
not: every stage's optimizer numbers its own parameters from 0, so positional
keys collide across ranks in one shared checkpoint (see ``OptimizerWrapper``).

Known seam, worked around rather than fixed here: the synthetic random corpus
carries no cursor in the checkpoint (``Trainer._build_dataloader`` drops it --
batch k is a pure function of ``(seed, k)``), and ``_data_iterator`` starts a
fresh loader at batch 0. The resume phase therefore fast-forwards the new
iterator by N batches; replaying generation lands on exactly the batches the
uninterrupted run consumed at steps N+1..N+M.

Everything runs in fp32 on CPU/gloo. The workdir is removed by the resume
phase once the comparison is done.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import torch
import torch.distributed as dist

from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from hpmesh.trainer.config import CheckpointConfig, MetricsConfig
from hpmesh.trainer.trainer import Trainer

SPLIT_STEP = 2  # N: checkpoint taken after this many steps
EXTRA_STEPS = 2  # M: steps trained after the checkpoint (both runs)
TOTAL_STEPS = SPLIT_STEP + EXTRA_STEPS
MICROBATCHES = 4
GLOBAL_BATCH = 8
SEQ = 32
VOCAB = 128
SEED = 42

FULL_LOSSES_FILE = "losses_full.json"


def _cfg(workdir: str) -> HybridMeshConfig:
    return HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",  # offline: AutoConfig.for_model("qwen3", ...)
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=MICROBATCHES,
            # -1 derives the shard degree from the world size; with pp=2 on 2
            # ranks that leaves dp=1, so both stages see the whole batch.
            data_parallel_shard_degree=-1,
            backend="gloo",
        ),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            max_seq_len=SEQ,
            steps=TOTAL_STEPS,
            seed=SEED,
            deterministic=True,
            dump_folder=workdir,
            checkpoint_config=CheckpointConfig(
                enable=True,
                folder="checkpoint",
                interval=SPLIT_STEP,
                # No retention policy to exercise: keep the purge thread out.
                keep_latest_k=0,
            ),
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def _train_steps(trainer: Trainer, data_iterator, num_steps: int) -> list[float]:
    """Drive ``num_steps`` optimizer steps; return the last-stage losses."""
    losses = []
    for _ in range(num_steps):
        trainer.step += 1
        metrics = trainer.train_step(data_iterator)
        assert metrics is not None  # log_freq=1: every step reports
        if trainer.pp_has_last_stage:
            losses.append(metrics["loss"])
    return losses


def _phase_full(workdir: str) -> None:
    trainer = Trainer(_cfg(workdir))
    assert trainer.world_size == 2, (
        f"this check assumes 2 ranks, got {trainer.world_size}"
    )

    data_iterator = trainer._data_iterator()
    losses = _train_steps(trainer, data_iterator, SPLIT_STEP)

    # The interval policy agrees this is a checkpointing step; a full state
    # (not the model-only last-step export) because ``last_step`` is False.
    saved = trainer.checkpointer.save(SPLIT_STEP)
    assert saved, "the split-step checkpoint was not written"

    losses += _train_steps(trainer, data_iterator, EXTRA_STEPS)

    checkpoint_dir = os.path.join(workdir, "checkpoint", f"step-{SPLIT_STEP}")
    if trainer.pp_has_last_stage:
        with open(os.path.join(workdir, FULL_LOSSES_FILE), "w") as f:
            json.dump(losses, f)
        assert os.path.isfile(os.path.join(checkpoint_dir, ".metadata")), (
            f"no DCP checkpoint at {checkpoint_dir}"
        )

    print(
        f"[full] rank {trainer.rank}: trained {TOTAL_STEPS} steps, "
        f"saved step-{SPLIT_STEP} checkpoint, losses={losses}",
        flush=True,
    )
    dist.barrier()
    trainer.checkpointer.close()
    trainer.metrics.close()
    dist.destroy_process_group()


def _phase_resume(workdir: str) -> None:
    trainer = Trainer(_cfg(workdir))
    assert trainer.world_size == 2, (
        f"this check assumes 2 ranks, got {trainer.world_size}"
    )
    failures: list[str] = []

    loaded = trainer.checkpointer.load(-1)
    if not loaded:
        failures.append(f"rank {trainer.rank}: no checkpoint was loaded")
    if trainer.step != SPLIT_STEP:
        failures.append(
            f"rank {trainer.rank}: resumed at step {trainer.step}, "
            f"expected {SPLIT_STEP}"
        )

    # Non-vacuity: the optimizer must have come back warm -- a cold Adam (no
    # or zeroed moments) is exactly the failure a loss match would then catch,
    # so assert it directly for a clearer message.
    optim_state = trainer.optimizer.state_dict()["state"]
    if not optim_state:
        failures.append(f"rank {trainer.rank}: optimizer state is empty")
    elif not any(float(s["exp_avg"].abs().sum()) > 0 for s in optim_state.values()):
        failures.append(f"rank {trainer.rank}: optimizer moments are all zero")

    # The synthetic source is not checkpointed (see module docstring): replay
    # the first N batches so step N+1 sees what the uninterrupted run saw.
    data_iterator = trainer._data_iterator()
    for _ in range(SPLIT_STEP):
        next(data_iterator)

    losses = _train_steps(trainer, data_iterator, EXTRA_STEPS)

    with open(os.path.join(workdir, FULL_LOSSES_FILE)) as f:
        full_losses = json.load(f)
    uninterrupted_tail = full_losses[SPLIT_STEP:]

    # The real losses live on the last-stage rank; collect them everywhere.
    gathered = [losses]
    dist.broadcast_object_list(gathered, src=trainer.world_size - 1)
    resumed_losses = gathered[0]

    max_diff = 0.0
    for offset, (got, want) in enumerate(
        zip(resumed_losses, uninterrupted_tail, strict=True), 1
    ):
        diff = abs(got - want)
        max_diff = max(max_diff, diff)
        if got != want:  # bitwise: a correct round-trip reproduces the arithmetic
            failures.append(
                f"rank {trainer.rank}: step {SPLIT_STEP + offset} loss {got!r} "
                f"vs uninterrupted {want!r} (diff {diff:.3e})"
            )

    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if trainer.rank == 0:
        print(
            f"pp=2 checkpoint round-trip: split={SPLIT_STEP} "
            f"extra={EXTRA_STEPS} microbatches={MICROBATCHES}"
        )
        print(f"uninterrupted losses = {[f'{x:.6f}' for x in full_losses]}")
        print(f"resumed losses       = {[f'{x:.6f}' for x in resumed_losses]}")
        print(f"max abs diff         = {max_diff:.3e}")
        for f_ in failures:
            print(f"  FAIL {f_}")
        if not failures:
            print("all checks passed")

    dist.barrier()
    trainer.checkpointer.close()
    trainer.metrics.close()
    dist.destroy_process_group()
    if trainer.rank == 0:
        shutil.rmtree(workdir, ignore_errors=True)

    assert local_ok.item() == 0, "PP checkpoint round-trip check failed"


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else None
    workdir = sys.argv[2] if len(sys.argv) > 2 else None
    if phase not in ("full", "resume") or not workdir:
        raise SystemExit(
            "usage: torchrun --nproc_per_node=2 "
            "tests/pp_checkpoint_equivalence.py {full|resume} WORKDIR"
        )
    if phase == "full":
        os.makedirs(workdir, exist_ok=True)
        _phase_full(workdir)
    else:
        _phase_resume(workdir)


if __name__ == "__main__":
    main()
