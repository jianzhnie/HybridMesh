"""The validation loop: config rules, feasibility gates, and the pass itself.

Everything here runs single-process on CPU: the pass's two loud errors (zero
batches, zero valid tokens), the token-normalized reporting loss, the
eval/train mode bracketing, and the build-time rejection of the combinations
that cannot terminate cleanly (``steps=-1`` with DP > 1 or an infinite corpus,
and pipeline parallelism). The cross-rank reductions are exercised only in
their single-rank identity form; the multi-rank semantics share the training
loss's meshes and need a real cluster to verify.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from hpmesh.components.loss import IGNORE_INDEX
from hpmesh.trainer.config import HybridMeshConfig, TrainingConfig, ValidationConfig


def _trainer_cls():
    """The trainer, imported lazily: on torch < 2.10 its import chain
    (DTensor, pipelining, distributed.checkpoint internals) fails at
    collection, and the tests that need it skip rather than error. The config
    and gating tests above do not need it and always run."""
    return pytest.importorskip(
        "hpmesh.trainer.trainer",
        reason="trainer import needs a newer torch than this host has",
    ).Trainer

VOCAB = 8
SEQ_LEN = 4


class _ListLoader:
    """The smallest object with the dataloader surface the pass uses."""

    def __init__(self, batches: list[dict[str, Any]]):
        self.batches = batches
        self.closed = False

    def __iter__(self):
        return iter(self.batches)

    def close(self) -> None:
        self.closed = True


class _EchoModel(torch.nn.Module):
    """Deterministic logits, so the expected loss is computable in the test."""

    def __init__(self) -> None:
        super().__init__()
        self.saw_eval_mode: bool | None = None
        self.saw_num_valid_tokens_kwarg = False

    def preprocess_inputs(self, batch, **kwargs):
        if "num_valid_tokens" in batch:
            self.saw_num_valid_tokens_kwarg = True
        labels = batch["labels"]
        return batch["input_ids"], labels, {}

    def forward(self, inputs, **kwargs):
        self.saw_eval_mode = not self.training
        logits = torch.zeros(inputs.shape[0], inputs.shape[1], VOCAB)
        logits[..., 0] = 1.0
        logits[..., 1] = 0.5
        return logits


def _batch(valid_tokens: int) -> dict[str, Any]:
    labels = torch.ones(1, SEQ_LEN, dtype=torch.long)
    labels[0, valid_tokens:] = IGNORE_INDEX
    return {
        "input_ids": torch.ones(1, SEQ_LEN, dtype=torch.long),
        "labels": labels,
        "num_valid_tokens": valid_tokens,
    }


def _make_trainer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation: ValidationConfig | None,
    loader: _ListLoader,
    model: _EchoModel | None = None,
):
    Trainer = _trainer_cls()
    trainer = Trainer.__new__(Trainer)
    cfg = HybridMeshConfig()
    cfg.training.validation_config = validation
    trainer.cfg = cfg
    trainer.parallel_dims = None
    trainer.device = torch.device("cpu")
    trainer.model = model if model is not None else _EchoModel()
    trainer.model_parts = [trainer.model]
    trainer.ntokens_seen = 7
    logged: dict[str, Any] = {}
    trainer.metrics = SimpleNamespace(
        add_tokens=lambda n: None,
        log_validation=lambda loss, step: logged.update(loss=loss, step=step),
    )
    monkeypatch.setattr(
        "hpmesh.trainer.trainer.build_dataloader", lambda *a, **k: loader
    )
    return trainer, logged


def _expected_loss(batches: list[dict[str, Any]]) -> float:
    loss_sum = sum(
        float(_trainer_cls()._loss_sum(_EchoModel()(b["input_ids"]), b["labels"]))
        for b in batches
    )
    valid = sum(b["num_valid_tokens"] for b in batches)
    return loss_sum / valid


# -- config rules ------------------------------------------------------------


def test_validation_freq_must_be_positive() -> None:
    with pytest.raises(ValueError, match="validation.freq"):
        ValidationConfig(freq=0)


def test_validation_steps_must_be_positive_or_neg1() -> None:
    for steps in (0, -2):
        with pytest.raises(ValueError, match="validation.steps"):
            ValidationConfig(steps=steps)
    ValidationConfig(steps=-1)
    ValidationConfig(steps=3)


def test_validation_defaults_to_off() -> None:
    # TrainingConfig alone: HybridMeshConfig() also builds ParallelConfig,
    # whose schedule check imports torch.distributed.pipelining (absent on
    # older torch), which is not what this assertion is about.
    assert TrainingConfig().validation is None


# -- build-time feasibility ---------------------------------------------------


def test_feasibility_rejects_pipeline_parallelism() -> None:
    Trainer = _trainer_cls()
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        Trainer._check_validation_feasibility(
            ValidationConfig(), pp_enabled=True, dp_world_size=1,
            training_dataset="local_jsonl",
        )


def test_feasibility_rejects_steps_neg1_when_dp_gt_1() -> None:
    Trainer = _trainer_cls()
    with pytest.raises(ValueError, match="validation collectives"):
        Trainer._check_validation_feasibility(
            ValidationConfig(steps=-1), pp_enabled=False, dp_world_size=2,
            training_dataset="local_jsonl",
        )


def test_feasibility_rejects_steps_neg1_on_the_infinite_corpus() -> None:
    Trainer = _trainer_cls()
    with pytest.raises(ValueError, match="infinite synthetic source"):
        Trainer._check_validation_feasibility(
            ValidationConfig(steps=-1), pp_enabled=False, dp_world_size=1,
            training_dataset="random",
        )
    # The override corpus is what the pass reads, so it is the one checked.
    with pytest.raises(ValueError, match="infinite synthetic source"):
        Trainer._check_validation_feasibility(
            ValidationConfig(steps=-1, dataset="random"), pp_enabled=False,
            dp_world_size=1, training_dataset="local_jsonl",
        )


def test_feasibility_accepts_the_terminating_combinations() -> None:
    Trainer = _trainer_cls()
    Trainer._check_validation_feasibility(
        ValidationConfig(steps=-1), pp_enabled=False, dp_world_size=1,
        training_dataset="local_jsonl",
    )
    Trainer._check_validation_feasibility(
        ValidationConfig(steps=10), pp_enabled=False, dp_world_size=8,
        training_dataset="random",
    )


# -- gating -------------------------------------------------------------------


def test_should_validate() -> None:
    Trainer = _trainer_cls()

    def gating_trainer(validation: ValidationConfig | None):
        trainer = Trainer.__new__(Trainer)
        cfg = HybridMeshConfig()
        cfg.training.validation_config = validation
        trainer.cfg = cfg
        return trainer

    trainer = gating_trainer(ValidationConfig(freq=10))
    assert trainer.should_validate(1)
    assert not trainer.should_validate(2)
    assert trainer.should_validate(10)
    assert trainer.should_validate(20)

    trainer_off = gating_trainer(None)
    assert not trainer_off.should_validate(1)
    assert not trainer_off.should_validate(10)


# -- the pass -------------------------------------------------------------------


def test_validate_reports_token_normalized_loss_and_restores_train_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [_batch(3), _batch(2)]
    loader = _ListLoader(batches)
    model = _EchoModel()
    model.train()
    trainer, logged = _make_trainer(
        monkeypatch, validation=ValidationConfig(steps=-1), loader=loader, model=model
    )

    trainer.validate(step=10)

    assert logged["step"] == 10
    assert logged["loss"] == pytest.approx(_expected_loss(batches))
    # The pass ran in eval mode and the model is back in train mode after.
    assert model.saw_eval_mode is True
    assert model.training
    # The bookkeeping int never reaches the model forward.
    assert not model.saw_num_valid_tokens_kwarg
    # Validation is a pure observer of the checkpointed training counter.
    assert trainer.ntokens_seen == 7
    assert loader.closed


def test_validate_positive_steps_bounds_the_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [_batch(2), _batch(2), _batch(2)]
    loader = _ListLoader(batches)
    trainer, logged = _make_trainer(
        monkeypatch, validation=ValidationConfig(steps=1), loader=loader
    )

    trainer.validate(step=10)

    assert logged["loss"] == pytest.approx(_expected_loss(batches[:1]))


def test_validate_raises_on_zero_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _ListLoader([])
    model = _EchoModel()
    model.train()
    trainer, _ = _make_trainer(
        monkeypatch, validation=ValidationConfig(steps=-1), loader=loader, model=model
    )

    with pytest.raises(ValueError, match="zero batches"):
        trainer.validate(step=1)

    assert model.training
    assert loader.closed


def test_validate_raises_on_zero_valid_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _ListLoader([_batch(0)])
    trainer, _ = _make_trainer(
        monkeypatch, validation=ValidationConfig(steps=-1), loader=loader
    )

    with pytest.raises(ValueError, match="zero valid tokens"):
        trainer.validate(step=1)

    assert loader.closed
