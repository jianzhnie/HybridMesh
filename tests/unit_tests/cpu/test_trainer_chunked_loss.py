"""Trainer-level chunked loss: the loop must train identically, chunk or not.

The loss-function equivalence is pinned in
``components/test_chunked_loss.py``; this suite pins the *wiring*: the config
field reaches the body, the forward is entered with ``skip_lm_head=True``, and
the per-chunk backward composes with the loop's token normalization, clipping
and optimizer step. Two real (single-process) Trainers are built from the same
seed -- one plain, one chunked -- and must produce the same loss trajectory and
the same final parameters.
"""

from __future__ import annotations

from tests.caps import require_env

require_env('dtensor', 'pipelining', 'spmd_types')


import pytest
import torch

from hpmesh.config import (
    HybridMeshConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from hpmesh.trainer.trainer import Trainer

STEPS = 3
# fp32 CPU; the only sanctioned divergence is the per-chunk summation order.
TOL = 1e-5


def _cfg(chunked_loss_num_chunks: int, dump_folder: str) -> HybridMeshConfig:
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
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=8,
            max_seq_len=32,
            steps=STEPS,
            seed=42,
            deterministic=True,
            chunked_loss_num_chunks=chunked_loss_num_chunks,
            dump_folder=dump_folder,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def _run(cfg: HybridMeshConfig) -> tuple[list[float], dict[str, torch.Tensor]]:
    trainer = Trainer(cfg)
    try:
        data_iterator = trainer._data_iterator()
        losses = []
        for _ in range(STEPS):
            trainer.step += 1
            metrics = trainer.train_step(data_iterator)
            assert metrics is not None  # log_freq=1: every step reports
            losses.append(metrics["loss"])
        params = {
            name: p.detach().clone()
            for name, p in trainer.model_parts[0].named_parameters()
        }
        return losses, params
    finally:
        trainer.close()


def test_chunked_trainer_matches_plain_trainer(tmp_path) -> None:
    plain_losses, plain_params = _run(_cfg(1, str(tmp_path / "plain")))
    chunked_losses, chunked_params = _run(_cfg(3, str(tmp_path / "chunked")))

    assert plain_params.keys() == chunked_params.keys()
    for step, (got, want) in enumerate(zip(chunked_losses, plain_losses, strict=True)):
        assert abs(got - want) <= TOL, (
            f"step {step + 1}: chunked loss {got:.6f} vs plain {want:.6f}"
        )
    for name, want in plain_params.items():
        torch.testing.assert_close(chunked_params[name], want, rtol=TOL, atol=TOL)


def test_chunked_loss_config_validates() -> None:
    with pytest.raises(ValueError, match="chunked_loss_num_chunks"):
        TrainingConfig(chunked_loss_num_chunks=0)


def test_the_reported_token_count_describes_the_data_not_the_split(tmp_path) -> None:
    """``n_tokens_seen`` must count the corpus read, not the rank's slice.

    It is the counterpart to the loss denominator: the loss is normalized by
    the tokens this step *trained on*, so the cumulative count logged beside it
    has to describe the same data -- if the two disagreed, the loss would not be
    reproducible from the token count. The counter is maintained from the
    unsharded batch (``_microbatch``), so on one process it must therefore be
    exactly the batch size times the sequence length times the steps taken,
    whatever the loss body did with those tokens.

    Non-vacuity: the expected value is computed from the config rather than read
    off the trainer, so a counter that reported ``0`` -- which is what it would
    have to be to be absent from the payload -- fails rather than passes.
    """
    cfg = _cfg(1, str(tmp_path / "tokens"))
    expected_per_step = cfg.training.global_batch_size * cfg.training.max_seq_len

    trainer = Trainer(cfg)
    try:
        data_iterator = trainer._data_iterator()
        for step in range(STEPS):
            trainer.step += 1
            metrics = trainer.train_step(data_iterator)
            assert metrics is not None
            assert metrics["n_tokens_seen"] == expected_per_step * (step + 1)
    finally:
        trainer.close()
