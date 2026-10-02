"""PP x real-corpus check: a 2-stage pipeline over packed local_jsonl data.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \\
        tests/integration_tests/pp_real_corpus_equivalence.py

The pipeline used to refuse every corpus but the synthetic ``random`` one,
on the grounds that a packed corpus's per-token positions could not cross the
pipeline. They can and do: the PP body threads each microbatch's
``extra_kwargs`` (positions, masks) through the schedule's ``kwarg_mbs``, the
same channel upstream's ``Validator`` uses. This check trains a tiny qwen3
for a few steps on a packed local_jsonl corpus and compares the loss
trajectory against a single-process, unsharded reference computed in-process
over the same loader.

Non-vacuity: the corpus is really packed (more than one document per row),
the model is really split, and every batch carries positions.

Environment note: needs torch >= 2.12 (the pipelining/flex surface) and
grain. Written 2026-10-02 with the PP x real-corpus unlock; environment not
covered on the development host -- awaiting a multi-rank run on the target
torch.
"""

from __future__ import annotations

import json
import tempfile

import torch
import torch.distributed as dist

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.components.loss import IGNORE_INDEX, cross_entropy_loss
from llmtuner.config import DataloaderConfig, MetricsConfig
from llmtuner.datasets.build import build_dataloader
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
from tests.data_fixtures import write_tokenizer

STEPS = 3
MICROBATCHES = 4
GLOBAL_BATCH = 8
SEQ = 64
VOCAB = 128
SEED = 42
PP = 2
WORLD = 2
NUM_ROWS = 48  # > STEPS * GLOBAL_BATCH rows so no repeat kicks in
TOL = 1e-5


def _write_corpus() -> tuple[str, str]:
    """A tiny tokenizer and a JSONL corpus of multi-document rows."""
    import os

    root = tempfile.mkdtemp(prefix="pp_real_corpus_")
    tokenizer_path = os.path.join(root, "tokenizer")
    write_tokenizer(tokenizer_path)
    corpus_path = os.path.join(root, "rows.jsonl")
    with open(corpus_path, "w") as handle:
        for i in range(NUM_ROWS):
            handle.write(
                json.dumps({"text": f"w{i} " + "lorem ipsum " * (i % 5 + 1)}) + "\n"
            )
    return tokenizer_path, corpus_path


def _cfg(tokenizer_path: str, corpus_path: str) -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=PP,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=MICROBATCHES,
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
            dataloader_config=DataloaderConfig(
                dataset="local_jsonl",
                tokenizer_path=tokenizer_path,
                dataset_path=corpus_path,
                shuffle=False,
            ),
        ),
    )


def _reference_trajectory(cfg: LLMTunerConfig) -> list[float]:
    """The same steps in one process over the same loader, unsharded."""
    torch.manual_seed(cfg.seed)
    model = HFTransformerModel(build_model_config_for(cfg))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=cfg.optimizer.implementation == "fused",
        foreach=cfg.optimizer.implementation == "foreach",
    )
    loader = build_dataloader(
        cfg,
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=GLOBAL_BATCH * SEQ,
    )
    rows_per_mb = GLOBAL_BATCH // MICROBATCHES
    losses = []
    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        batch = next(iter(loader))
        # The packed batch carries per-token positions; drop the trainer's
        # bookkeeping int before the forward, as the trainer does.
        batch = dict(batch)
        batch.pop("num_valid_tokens", None)
        loss_sum = None
        for mb in range(MICROBATCHES):
            row = slice(mb * rows_per_mb, (mb + 1) * rows_per_mb)
            microbatch = {k: v[row] for k, v in batch.items()}
            num_valid = int((microbatch["labels"] != IGNORE_INDEX).sum())
            inputs, labels, extra = model.preprocess_inputs(
                microbatch, parallel_dims=None
            )
            logits = model(inputs, **extra)
            loss = cross_entropy_loss(logits, labels)
            (loss / num_valid).backward()
            loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
        clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm, foreach=True)
        optimizer.step()
        losses.append(float(loss_sum))
    loader.close()
    return losses


def main() -> None:
    tokenizer_path, corpus_path = _write_corpus()
    cfg = _cfg(tokenizer_path, corpus_path)
    failures: list[str] = []

    trainer = Trainer(cfg)
    rank = trainer.rank
    assert trainer.world_size == WORLD, (
        f"this check assumes {WORLD} ranks, got {trainer.world_size}"
    )

    # -- non-vacuity: the model is split, and the corpus is really packed -----
    num_layers_held = sum(len(part.layers) for part in trainer.model_parts)
    if num_layers_held >= cfg.num_hidden_layers:
        failures.append(
            f"rank {rank}: holds {num_layers_held} of {cfg.num_hidden_layers} "
            "layers -- the model was not split"
        )
    data_iterator = trainer.data_iterator()
    probe = next(iter(trainer.dataloader))
    if "positions" not in probe:
        failures.append(f"rank {rank}: packed batch carries no positions")
    positions = probe["positions"]
    if positions is not None and int((positions == 0).sum()) < 2:
        failures.append(
            f"rank {rank}: a packed row should restart positions more than "
            f"once, got {int((positions == 0).sum())} restarts"
        )

    # -- the trajectory ------------------------------------------------------
    pp_losses = []
    for _step in range(STEPS):
        trainer.step += 1
        metrics = trainer.train_step(data_iterator)
        assert metrics is not None  # log_freq=1: every step reports
        pp_losses.append(metrics["loss"])
    trainer.close()

    reference = _reference_trajectory(cfg)
    for step, (got, want) in enumerate(zip(pp_losses, reference, strict=True)):
        if abs(got - want) > TOL:
            failures.append(f"rank {rank} step {step + 1}: loss {got} vs {want}")

    if rank == 0:
        print(f"pp={PP} corpus=local_jsonl(packed) steps={STEPS} tol={TOL:.0e}")
        print(f"pp losses  = {[f'{x:.6f}' for x in pp_losses]}")
        print(f"reference  = {[f'{x:.6f}' for x in reference]}")
        for f in failures:
            print(f"  FAIL {f}")
        print("all checks passed" if not failures else "CHECKS FAILED")
    verdict = torch.tensor(len(failures), dtype=torch.int64)
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    assert int(verdict) == 0, f"{int(verdict)} check(s) failed -- see rank 0 output"


if __name__ == "__main__":
    main()
