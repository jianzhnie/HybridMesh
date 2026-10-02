"""PP x CP check: a 2-stage pipeline with 2-way CP must match single-process.

Run under torchrun with 4 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/pp_cp_equivalence.py

The pipeline (pp=2) composes with context parallelism (cp=2, kv_allgather):
every stage's decoder layers run the CP attention over this rank's sequence
shard, and the p2p channel carries the already-sharded activations between
stages at the same CP coordinate. A tiny qwen3 (dense) trains for a few steps
on the synthetic corpus; the per-step losses must track a single-process,
unsharded reference computed in-process.

Non-vacuity: the model is actually split across stages, and the CP mesh is
size 2 (a size-1 CP group would make the comparison vacuous).

Environment note: needs torch >= 2.12 (the pipelining/flex surface). Written
2026-10-02 with the PP x CP unlock; environment not covered on the
development host -- awaiting a multi-rank run on the target torch.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.components.loss import IGNORE_INDEX, cross_entropy_loss
from llmtuner.config import MetricsConfig
from llmtuner.datasets.random_data import RandomTokenSource, batch_iterator
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.trainer import (
    LLMTunerConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.trainer.trainer import Trainer

STEPS = 4
MICROBATCHES = 4
GLOBAL_BATCH = 8
SEQ = 256  # torch's CP BlockMask path requires Q_LEN % (cp * 128) == 0
VOCAB = 128
SEED = 42
PP = 2
CP = 2
WORLD = 4  # pp * cp, dp_shard = 1

# Looser than the dense PP check: CP's all-gathered K/V reassociate the fp32
# attention sums relative to the unsharded reference.
TOL = 1e-4


def _cfg() -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,  # 4 so both stages get 2 layers
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=PP,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=MICROBATCHES,
            context_parallel_size=CP,
            data_parallel_shard_size=-1,
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


def _reference_trajectory(cfg: LLMTunerConfig) -> list[float]:
    """The same steps in one process: no pipeline, no CP (CP is numerically
    transparent: every rank attends the same full K/V)."""
    torch.manual_seed(cfg.seed)
    model = HFTransformerModel(build_model_config_for(cfg))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=cfg.optimizer.implementation == "fused",
        foreach=cfg.optimizer.implementation == "foreach",
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
        targets = model.preprocess_inputs(batch, parallel_dims=None)[1].reshape(
            cfg.global_batch_size, cfg.max_seq_len
        )
        num_valid = int((targets != IGNORE_INDEX).sum())
        loss_sum = None
        for mb in range(MICROBATCHES):
            row = slice(mb * rows_per_mb, (mb + 1) * rows_per_mb)
            logits = model(batch.input_ids[row].reshape(-1))
            loss = cross_entropy_loss(logits, targets[row].reshape(-1))
            (loss / num_valid).backward()
            loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
        clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm, foreach=True)
        optimizer.step()
        losses.append(float(loss_sum / num_valid))
    return losses


def main() -> None:
    cfg = _cfg()
    failures: list[str] = []

    trainer = Trainer(cfg)
    rank = trainer.rank
    assert trainer.world_size == WORLD, (
        f"this check assumes {WORLD} ranks, got {trainer.world_size}"
    )

    # -- non-vacuity: the model is split, and CP is real ----------------------
    num_layers_held = sum(len(part.layers) for part in trainer.model_parts)
    if num_layers_held >= cfg.num_hidden_layers:
        failures.append(
            f"rank {rank}: holds {num_layers_held} of {cfg.num_hidden_layers} "
            "layers -- the model was not split"
        )
    cp_mesh = trainer.parallel_dims.get_optional_mesh("cp")
    if cp_mesh is None or cp_mesh.size() != CP:
        failures.append(
            f"rank {rank}: cp mesh size "
            f"{None if cp_mesh is None else cp_mesh.size()}, want {CP}"
        )

    # -- the trajectory ------------------------------------------------------
    data_iterator = trainer.data_iterator()
    pp_cp_losses = []
    for _step in range(STEPS):
        trainer.step += 1
        metrics = trainer.train_step(data_iterator)
        assert metrics is not None  # log_freq=1: every step reports
        pp_cp_losses.append(metrics["loss"])
    trainer.close()

    reference = _reference_trajectory(cfg)
    for step, (got, want) in enumerate(zip(pp_cp_losses, reference, strict=True)):
        if abs(got - want) > TOL:
            failures.append(f"rank {rank} step {step + 1}: loss {got} vs {want}")

    if rank == 0:
        print(f"pp={PP} cp={CP} world={WORLD} steps={STEPS} tol={TOL:.0e}")
        print(f"pp+cp losses = {[f'{x:.6f}' for x in pp_cp_losses]}")
        print(f"reference    = {[f'{x:.6f}' for x in reference]}")
        for f in failures:
            print(f"  FAIL {f}")
        print("all checks passed" if not failures else "CHECKS FAILED")
    verdict = torch.tensor(len(failures), dtype=torch.int64)
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    assert int(verdict) == 0, f"{int(verdict)} check(s) failed -- see rank 0 output"


if __name__ == "__main__":
    main()
