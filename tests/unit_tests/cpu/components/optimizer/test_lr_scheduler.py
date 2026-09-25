"""The learning-rate schedule: the curve, its edges, and its checkpoint contract.

Everything here runs against a real ``torch.optim.AdamW`` because the schedule's
whole job is to scale that optimizer's lrs, and a mocked one would not exercise
``LambdaLR``'s coupling to ``base_lrs``.
"""

from __future__ import annotations

import pytest
import torch

from hpmesh.components.optimizer import LRSchedulersContainer, build_lr_scheduler
from hpmesh.config import LRSchedulerConfig


def _optimizer(lr: float = 1.0) -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    return torch.optim.AdamW([parameter], lr=lr)


def _lrs(scheduler: LRSchedulersContainer) -> list[float]:
    return scheduler.schedulers[0].get_last_lr()


def _step(scheduler: LRSchedulersContainer) -> float:
    """One training step's worth of schedule: optimizer, then scheduler.

    Stepping a real optimizer (not just the scheduler) is what reproduces the
    trainer's ordering, and ``LambdaLR`` warns when it is stepped before its
    optimizer -- a warning that would otherwise fire on every test here and hide
    a real ordering mistake. The optimizer update itself is irrelevant to the
    lr, so the value it is given does not matter.
    """
    scheduler.schedulers[0].optimizer.step()
    scheduler.step()
    return _lrs(scheduler)[0]


def _advance(scheduler: LRSchedulersContainer, steps: int) -> None:
    for _ in range(steps):
        _step(scheduler)


# -- the curve ---------------------------------------------------------------


def test_default_config_is_a_constant_learning_rate() -> None:
    """Off must mean exactly off, or the default run is not the baseline.

    Every loss measurement taken before this schedule existed was made at a
    constant lr, so a default that scaled it even slightly would invalidate all
    of them without failing anything.
    """
    optimizer = _optimizer(lr=0.25)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(), optimizers=[optimizer], training_steps=10
    )

    for step in range(10):
        assert _lrs(scheduler) == [0.25], f"step {step}"
        _step(scheduler)


def test_warmup_ramps_linearly_from_the_second_lr() -> None:
    """``lambda(0)`` is 1/warmup, so the FIRST step runs below the base lr.

    This matches torchtitan: ``scheduler.step()`` runs after ``optimizer.step()``,
    so step 1 uses ``lambda(0)``. A warmup of 4 therefore gives 1/4, 2/4, 3/4,
    then the base rate -- not a first step at the full lr.
    """
    optimizer = _optimizer(lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(warmup_steps=4), optimizers=[optimizer], training_steps=10
    )

    observed = []
    for _ in range(5):
        observed.append(_lrs(scheduler)[0])
        _step(scheduler)

    assert observed == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.0])


def test_decay_ratio_holds_the_peak_then_decays() -> None:
    """WSD: warmup, a stable phase at the peak, then decay over the last slice."""
    optimizer = _optimizer(lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(warmup_steps=2, decay_ratio=0.5),
        optimizers=[optimizer],
        training_steps=8,
    )

    observed = [_lrs(scheduler)[0]]
    for _ in range(7):
        observed.append(_step(scheduler))

    # warmup covers steps 0-1, decay the last half (steps 4-7), so steps 2-4 are
    # the stable phase at the peak rate and 5-7 walk down the decay.
    assert observed[:5] == pytest.approx([0.5, 1.0, 1.0, 1.0, 1.0])
    assert observed[5] > observed[6] > observed[7], "the tail must be decaying"


def test_min_lr_factor_floors_the_decay() -> None:
    optimizer = _optimizer(lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(decay_ratio=1.0, min_lr_factor=0.5),
        optimizers=[optimizer],
        training_steps=4,
    )

    _advance(scheduler, 4)
    assert _lrs(scheduler)[0] >= 0.5


@pytest.mark.parametrize("decay_type", ["linear", "sqrt", "cosine"])
def test_decay_types_all_reach_the_floor(decay_type: str) -> None:
    """Each shape is monotonically non-increasing and never goes negative."""
    optimizer = _optimizer(lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(decay_ratio=1.0, decay_type=decay_type),
        optimizers=[optimizer],
        training_steps=16,
    )

    observed = []
    for _ in range(16):
        observed.append(_lrs(scheduler)[0])
        _step(scheduler)

    assert all(value >= 0.0 for value in observed)
    assert all(
        later <= earlier + 1e-9
        for earlier, later in zip(observed, observed[1:], strict=False)
    )


def test_no_decay_reaches_zero_on_the_final_step() -> None:
    """The virtual last step is what keeps this just above zero, not at it."""
    optimizer = _optimizer(lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(decay_ratio=1.0), optimizers=[optimizer], training_steps=4
    )

    _advance(scheduler, 3)
    assert _lrs(scheduler)[0] == pytest.approx(0.25)


def test_unknown_decay_type_raises() -> None:
    optimizer = _optimizer()
    config = LRSchedulerConfig(decay_ratio=1.0)
    # Bypass __post_init__ the way a hand-built config would: the Literal type is
    # not enforced at runtime, so the guard has to be in the lambda.
    object.__setattr__(config, "decay_type", "exponential")
    scheduler = build_lr_scheduler(config, optimizers=[optimizer], training_steps=4)

    with pytest.raises(ValueError, match="Unknown decay_type"):
        _advance(scheduler, 2)


# -- config validation -------------------------------------------------------


def test_negative_warmup_is_rejected() -> None:
    with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
        LRSchedulerConfig(warmup_steps=-1)


def test_bad_decay_ratio_is_rejected() -> None:
    for ratio in (-0.1, 1.5):
        with pytest.raises(ValueError, match="decay_ratio must be in"):
            LRSchedulerConfig(decay_ratio=ratio)


def test_bad_min_lr_factor_is_rejected() -> None:
    for factor in (-0.1, 1.0):
        with pytest.raises(ValueError, match="min_lr_factor must be in"):
            LRSchedulerConfig(min_lr_factor=factor)


def test_total_steps_shorter_than_the_run_is_rejected() -> None:
    """A schedule that ends early would take the lr negative, not just to zero."""
    with pytest.raises(ValueError, match="shorter than the run"):
        build_lr_scheduler(
            LRSchedulerConfig(total_steps=4),
            optimizers=[_optimizer()],
            training_steps=10,
        )


def test_warmup_longer_than_the_schedule_is_clamped(caplog) -> None:
    """Clamped with a warning, rather than erroring: a short debug run is common."""
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(warmup_steps=100), optimizers=[_optimizer()], training_steps=4
    )
    assert scheduler.total_steps == 4
    assert any("exceeds total_steps" in r.message for r in caplog.records)


def test_total_steps_decouples_the_curve_from_the_run_length() -> None:
    """The point of the knob: a 4-step run can see the lrs a 20-step run would."""
    optimizer = _optimizer(lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(warmup_steps=100, total_steps=20),
        optimizers=[optimizer],
        training_steps=4,
    )

    assert scheduler.total_steps == 20
    # lambda(0) on a 100-step warmup clamped to 20 => 1/20. The step AFTER this
    # is what shows the schedule advanced, hence the explicit step().
    assert _lrs(scheduler)[0] == pytest.approx(0.05)
    assert _step(scheduler) == pytest.approx(0.10)


# -- metrics and checkpointing -----------------------------------------------


def test_lr_metric_is_keyed_by_the_optimizer_name() -> None:
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(), optimizers=[_optimizer()], training_steps=4
    )
    assert scheduler.get_metrics() == {"lr/AdamW": 1.0}


def test_state_dict_is_just_the_epoch() -> None:
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(), optimizers=[_optimizer()], training_steps=8
    )
    _advance(scheduler, 3)

    assert scheduler.state_dict() == {"last_epoch": 3}


def test_load_state_dict_restores_the_scheduled_lr() -> None:
    """A resumed run must continue the curve, not restart it."""
    optimizer = _optimizer(lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(warmup_steps=8), optimizers=[optimizer], training_steps=16
    )
    _advance(scheduler, 4)
    expected = _lrs(scheduler)[0]
    saved = scheduler.state_dict()

    # A fresh optimizer and scheduler, as a resumed run would build.
    resumed = build_lr_scheduler(
        LRSchedulerConfig(warmup_steps=8),
        optimizers=[_optimizer(lr=1.0)],
        training_steps=16,
    )
    assert _lrs(resumed)[0] != expected, "sanity: they must differ before loading"

    resumed.load_state_dict(saved)

    assert _lrs(resumed)[0] == pytest.approx(expected)


def test_load_state_dict_ignores_an_empty_dict() -> None:
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(), optimizers=[_optimizer()], training_steps=4
    )
    scheduler.load_state_dict({})
    assert _lrs(scheduler)[0] == 1.0


def test_a_third_of_the_way_through_warmup_is_a_third_of_the_lr() -> None:
    """End-to-end shape check: a known point on the curve, computed by hand."""
    optimizer = _optimizer(lr=0.9)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(warmup_steps=3), optimizers=[optimizer], training_steps=6
    )

    _advance(scheduler, 1)
    # lambda(1) == 2/3 on a 3-step warmup.
    assert _lrs(scheduler)[0] == pytest.approx(0.9 * 2 / 3)
