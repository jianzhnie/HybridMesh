"""Chunked loss check: the chunked trainer must train like the plain reference.

Run under torchrun with 2 ranks (from the repo root):

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/chunked_loss_equivalence.py

A tiny offline llama (random init, fixed seed) goes through the real ``Trainer``
with ``chunked_loss_num_chunks=3`` (uneven chunks over the 128-token per-rank
batch) for 4 optimizer steps and is compared against a single-process,
full-logits reference over the same global batches -- the loss trajectory AND
the final parameters.

Two variants, selected by env var:

* default (``CHUNKED_LOSS_EQ_TP`` unset): dp_shard=2, so the chunked backward
  composes with FSDP2 -- the lm_head's gradient is reduce-scattered once per
  chunk, which must sum to the same sharded gradient the plain path produces.
  Parameters are compared through ``full_tensor()``.
* ``CHUNKED_LOSS_EQ_TP=2``: tp=2, dp_shard=1. Each rank's hidden states are a
  T/tp sequence shard (the sequence-parallelism premise); chunking that local
  shard composes with the sum reduction, and the replicated lm_head's
  token-partial gradient is summed by the trainer's
  ``_allreduce_replicated_tp_grads``. This variant compares the loss
  trajectory only (the TP weight-layout comparison is tp_equivalence.py's
  job, not this one's).

Everything runs in fp32 on CPU/gloo.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from hpmesh.accelerator.collectives import clip_grad_norm_
from hpmesh.components.loss import IGNORE_INDEX, cross_entropy_loss
from hpmesh.datasets.random_data import Batch, RandomTokenSource, batch_iterator
from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config_for
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
GLOBAL_BATCH = 8
SEQ = 32
VOCAB = 128
SEED = 42
NUM_CHUNKS = 3

TP = int(os.environ.get("CHUNKED_LOSS_EQ_TP", "1"))
DP_SHARD = 2 // TP

# fp32; the sanctioned divergences are summation orders (per-chunk CE sums,
# per-chunk reduce-scatters), which sit at ~1e-6 relative.
LOSS_TOL = 1e-4
PARAM_TOL = 1e-4


def _cfg() -> HybridMeshConfig:
    return HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="llama",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            tensor_parallel_size=TP,
            data_parallel_shard_size=DP_SHARD,
        ),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            max_seq_len=SEQ,
            steps=STEPS,
            seed=SEED,
            deterministic=True,
            chunked_loss_num_chunks=NUM_CHUNKS,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def _reference_trajectory(cfg: HybridMeshConfig) -> tuple[list[float], dict]:
    """The same training, un-chunked and unparallelized: one process, full batch.

    Mirrors the trainer's non-PP step arithmetic exactly. As in
    tp_fsdp_equivalence.py, dp sharding turns the global batch into per-group
    flattened sequences, so the reference runs the DP groups as SEPARATE
    forwards and sums -- a single flat forward over the union would give the
    second group's rows different positions and a different causal boundary.
    """
    torch.manual_seed(cfg.seed)
    model = HFTransformerModel(build_model_config_for(cfg))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    batches = batch_iterator(
        RandomTokenSource(
            seed=cfg.seed,
            vocab_size=cfg.vocab_size,
            batch_size=cfg.global_batch_size,
            seq_len=cfg.max_seq_len,
        )
    )
    rows_per_dp = cfg.global_batch_size // DP_SHARD
    losses = []
    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        batch = next(batches)
        num_valid = int(
            (
                model.preprocess_inputs(batch, parallel_dims=None)[1] != IGNORE_INDEX
            ).sum()
        )
        loss_sum = None
        for d in range(DP_SHARD):
            rows = slice(d * rows_per_dp, (d + 1) * rows_per_dp)
            sub = Batch(input_ids=batch.input_ids[rows], labels=batch.labels[rows])
            inputs, targets, _ = model.preprocess_inputs(sub, parallel_dims=None)
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            loss_sum = loss if loss_sum is None else loss_sum + loss
        # Normalized BEFORE backward, as the trainer does.
        (loss_sum / num_valid).backward()
        clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm, foreach=True)
        optimizer.step()
        losses.append(float(loss_sum / num_valid))
    params = {name: p.detach().clone() for name, p in model.named_parameters()}
    return losses, params


def _full(t: torch.Tensor) -> torch.Tensor:
    """Global tensor at this rank: redistribute the FSDP shard."""
    return t.full_tensor() if isinstance(t, DTensor) else t.detach()


def main() -> None:
    cfg = _cfg()
    failures: list[str] = []

    # Trainer init owns the process group (torchrun env).
    trainer = Trainer(cfg)
    rank = trainer.rank
    world = trainer.world_size
    assert world == TP * DP_SHARD == 2, f"this check assumes 2 ranks, got {world}"

    reference, ref_params = _reference_trajectory(cfg)

    # -- the trajectory, through the real Trainer, chunked ---------------------
    data_iterator = trainer._data_iterator()
    losses = []
    for _ in range(STEPS):
        trainer.step += 1
        metrics = trainer.train_step(data_iterator)
        assert metrics is not None  # log_freq=1: every step reports
        losses.append(metrics["loss"])

    max_loss_diff = 0.0
    for step, (got, want) in enumerate(zip(losses, reference, strict=True), 1):
        diff = abs(got - want)
        max_loss_diff = max(max_loss_diff, diff)
        if diff > LOSS_TOL:
            failures.append(
                f"rank {rank}: step {step} loss {got:.6f} vs reference "
                f"{want:.6f} (diff {diff:.3e})"
            )

    # -- final parameters (FSDP variant only; see the module docstring) --------
    max_param_diff = 0.0
    if TP == 1:
        model = trainer.model_parts[0]
        saw_fsdp_shard = False
        for name, p in model.named_parameters():
            full = _full(p)
            want = ref_params[name]
            if full.shape != want.shape:
                failures.append(f"rank {rank}: param {name} shape {full.shape}")
                continue
            if isinstance(p, DTensor) and p.to_local().numel() < p.numel():
                saw_fsdp_shard = True
            diff = (full - want).abs().max().item()
            max_param_diff = max(max_param_diff, diff)
            if diff > PARAM_TOL:
                failures.append(f"rank {rank}: param {name} diff {diff:.3e}")
        if not saw_fsdp_shard:
            failures.append(f"rank {rank}: no parameter was FSDP-sharded -- vacuous")

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"tp={TP} dp_shard={DP_SHARD} chunks={NUM_CHUNKS} steps={STEPS} "
            f"global_batch={GLOBAL_BATCH} seq={SEQ}"
        )
        print(f"reference losses = {[f'{x:.6f}' for x in reference]}")
        print(f"chunked losses   = {[f'{x:.6f}' for x in losses]}")
        print(f"loss   max abs diff = {max_loss_diff:.3e}")
        if TP == 1:
            print(f"params max abs diff = {max_param_diff:.3e}")
        print(f"failed ranks     = {int(local_ok.item())}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "chunked loss equivalence check failed"
    trainer.checkpointer.close()
    trainer.metrics.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
