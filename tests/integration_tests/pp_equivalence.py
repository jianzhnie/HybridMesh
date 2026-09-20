"""PP>1 check: a split pipeline must train like the unpipelined model.

Run under torchrun with 2 ranks, one schedule per invocation:

    torchrun --nproc_per_node=2 tests/pp_equivalence.py            # 1F1B
    torchrun --nproc_per_node=2 tests/pp_equivalence.py Interleaved1F1B

A tiny offline qwen3 (random init, fixed seed) goes through the real
``Trainer`` with ``pp=2``, 4 micro-batches per step, for 4 optimizer steps,
under either a single-stage schedule (1F1B: one stage per rank) or a looped
one (Interleaved1F1B: two virtual stages per rank, over a deeper model so
every stage holds at least one layer). The loss on the last-stage rank must
track, step for step, a reference computed on the whole model with no
pipeline.

What the reference is, and why it is not just the ``pp=1`` trainer: the
trainer's non-PP body flattens the whole batch into ONE causal sequence, while
the PP body runs each micro-batch as its own sequence (pipeline stages are
separate forward calls; attention cannot cross a micro-batch boundary). The
reference therefore applies the *same* row chunking on one process -- same
batches, same per-micro-batch summed CE, same clip and AdamW -- so any
divergence is attributable to the pipeline machinery (stage split, p2p
activations/grads, schedule, cross-stage grad-norm reduction), not to a
different attention pattern.

Non-vacuity: each rank must hold only its own stages' layers, and the per-rank
parameter counts must sum to the whole model's.

Everything runs in fp32 on CPU/gloo. The tolerance sits above the only
intended difference: the gradient norm is reduced per stage and combined,
which is a different floating-point summation order than one whole-model norm.
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist
import torch.nn as nn

from hpmesh.components.loss import IGNORE_INDEX, cross_entropy_loss
from hpmesh.datasets.random_data import RandomTokenSource, batch_iterator
from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config_for
from hpmesh.parallel.collectives import clip_grad_norm_
from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from hpmesh.trainer.config import MetricsConfig
from hpmesh.trainer.trainer import Trainer

STEPS = 4
MICROBATCHES = 4
GLOBAL_BATCH = 8
SEQ = 32
VOCAB = 128
SEED = 42

# fp32; the only sanctioned divergence is the grad-norm summation order (per
# stage, then combined, vs one whole-model norm), which sits at ~1e-7 relative.
TOL = 1e-5


# Per-schedule scenario: a looped schedule runs two virtual stages per rank
# (4 stages over pp=2), so the model is deepened to give every stage at least
# one layer. The reference trajectory is schedule-independent -- it applies
# the same row chunking either way -- so only the split changes.
SCENARIOS = {
    # schedule: (num_hidden_layers, stages_per_rank)
    "1F1B": (2, 1),
    "Interleaved1F1B": (6, 2),
}


def _cfg(schedule: str) -> HybridMeshConfig:
    num_layers, _ = SCENARIOS[schedule]
    return HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",  # offline: AutoConfig.for_model("qwen3", ...)
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=2,
            pipeline_parallel_schedule=schedule,
            num_pp_microbatches=MICROBATCHES,
            # -1 derives the shard degree from the world size; with pp=2 on 2
            # ranks that leaves dp=1, so both stages see the whole batch.
            data_parallel_shard_size=-1,
            backend="gloo",
        ),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            max_seq_len=SEQ,
            steps=STEPS,
            seed=SEED,
            deterministic=True,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def _reference_trajectory(cfg: HybridMeshConfig) -> list[float]:
    """The same training step with no pipeline: same chunks, one process.

    Mirrors the trainer's step arithmetic exactly -- the same per-row target
    shift (``preprocess_inputs``), the same summed CE per micro-batch, the
    same clip and optimizer -- with the micro-batch loop unrolled locally
    instead of being driven through a schedule.
    """
    torch.manual_seed(cfg.seed)
    model = HFTransformerModel(build_model_config_for(cfg))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    rows_per_mb = cfg.global_batch_size // MICROBATCHES

    batches = batch_iterator(
        RandomTokenSource(
            seed=cfg.seed,
            vocab_size=cfg.vocab_size,
            batch_size=cfg.global_batch_size,
            seq_len=cfg.max_seq_len,
        )
    )
    losses = []
    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        batch = next(batches)
        # The model's own synthetic-path shift: within a row, row ends ignored.
        # Read off the model that actually trains, so the oracle's targets and
        # the PP body's cannot drift apart.
        targets = model.preprocess_inputs(batch, parallel_dims=None)[1].reshape(
            cfg.global_batch_size, cfg.max_seq_len
        )
        num_valid = int((targets != IGNORE_INDEX).sum())

        loss_sum = None
        for mb in range(MICROBATCHES):
            row = slice(mb * rows_per_mb, (mb + 1) * rows_per_mb)
            logits = model(batch.input_ids[row].reshape(-1))
            loss = cross_entropy_loss(logits, targets[row].reshape(-1))
            # Normalized BEFORE backward, as the trainer's PP body does -- the
            # schedule's ``_scalar_loss_fn`` divides by this same count. Not
            # cosmetic: ``clip_grad_norm_`` below reads the gradient, so
            # dividing afterwards would clip a gradient ``num_valid`` times too
            # large against an absolute threshold. The divisor is the whole
            # batch's count, not this micro-batch's, because it has to be the
            # same number for every micro-batch of the step.
            (loss / num_valid).backward()
            loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()

        clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm, foreach=True)
        optimizer.step()
        losses.append(float(loss_sum / num_valid))
    return losses


def main() -> None:
    schedule = sys.argv[1] if len(sys.argv) > 1 else "1F1B"
    if schedule not in SCENARIOS:
        raise ValueError(
            f"unknown schedule {schedule!r}; expected one of {sorted(SCENARIOS)}"
        )
    _, stages_per_rank = SCENARIOS[schedule]
    cfg = _cfg(schedule)
    failures: list[str] = []

    # Trainer init owns the process group (torchrun env); pp=2 over 2 ranks.
    trainer = Trainer(cfg)
    rank = trainer.rank
    assert trainer.world_size == 2, (
        f"this check assumes 2 ranks, got {trainer.world_size}"
    )

    # -- non-vacuity: this rank holds its stages and only their layers -------
    assert len(trainer.model_parts) == stages_per_rank
    num_layers_held = sum(len(part.layers) for part in trainer.model_parts)
    if num_layers_held >= cfg.num_hidden_layers:
        failures.append(
            f"rank {rank}: holds {num_layers_held} of {cfg.num_hidden_layers} "
            "layers -- the model was not split"
        )
    if trainer.pp_has_first_stage == trainer.pp_has_last_stage:
        failures.append(
            f"rank {rank}: has_first={trainer.pp_has_first_stage} "
            f"has_last={trainer.pp_has_last_stage} -- a 2-rank pipeline "
            "assigns exactly one of them"
        )
    # Both layouts keep the first stage first and the last stage last in the
    # rank's part list ("loop" stages ascend; the single-stage case has one).
    if trainer.pp_has_first_stage and isinstance(
        trainer.model_parts[0].tok_embeddings, nn.Identity
    ):
        failures.append(f"rank {rank}: first stage lost its embedding")
    if trainer.pp_has_last_stage and isinstance(
        trainer.model_parts[-1].lm_head, nn.Identity
    ):
        failures.append(f"rank {rank}: last stage lost its lm_head")

    # The stages' parameter sets are disjoint and cover the whole model.
    local_numel = sum(
        p.numel() for part in trainer.model_parts for p in part.parameters()
    )
    total_numel = torch.tensor([local_numel])
    dist.all_reduce(total_numel, op=dist.ReduceOp.SUM)
    reference_numel = sum(
        p.numel() for p in HFTransformerModel(build_model_config_for(cfg)).parameters()
    )
    if total_numel.item() != reference_numel:
        failures.append(
            f"rank {rank}: stage parameters sum to {total_numel.item()}, "
            f"the whole model has {reference_numel}"
        )

    # -- the trajectory ------------------------------------------------------
    data_iterator = trainer._data_iterator()
    pp_losses = []
    for _ in range(STEPS):
        trainer.step += 1
        metrics = trainer.train_step(data_iterator)
        assert metrics is not None  # log_freq=1: every step reports
        if trainer.pp_has_last_stage:
            # Only the last stage holds a real loss; other stages carry the
            # sentinel, which is never logged (the metrics rank is rank 1).
            pp_losses.append(metrics["loss"])

    reference = _reference_trajectory(cfg)

    # The real losses live on the last-stage rank; collect them on rank 0 for
    # the report. In both layouts the last stage sits on the last rank.
    gathered = [pp_losses]
    dist.broadcast_object_list(gathered, src=trainer.world_size - 1)
    pp_losses = gathered[0]

    # Every rank has both series now (the reference is computed locally and
    # identically on each), so every rank runs the comparison.
    max_diff = 0.0
    for step, (got, want) in enumerate(zip(pp_losses, reference, strict=True), 1):
        diff = abs(got - want)
        max_diff = max(max_diff, diff)
        if not torch.isclose(torch.tensor(got), torch.tensor(want), rtol=TOL, atol=TOL):
            failures.append(
                f"rank {rank}: step {step} loss {got:.6f} vs reference "
                f"{want:.6f} (diff {diff:.3e})"
            )

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"pp=2 schedule={schedule} steps={STEPS} microbatches={MICROBATCHES} "
            f"global_batch={GLOBAL_BATCH} seq={SEQ} tol={TOL:.0e}"
        )
        print(f"reference losses = {[f'{x:.6f}' for x in reference]}")
        print(f"pp losses        = {[f'{x:.6f}' for x in pp_losses]}")
        print(f"max abs diff     = {max_diff:.3e}")
        print(f"failed ranks     = {int(local_ok.item())}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "PP equivalence check failed"
    trainer.checkpointer.close()
    trainer.metrics.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
