# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The learning-rate schedule: linear warmup, stable phase, then decay.

Vendored from torchtitan ``components/optimizer/lr_scheduler.py``. Two things
were dropped, both consequences of hpmesh having one flat optimizer:

* **The container.** Upstream ``LRSchedulersContainer`` wraps a list of
  schedulers, one per optimizer in an ``OptimizersContainer``, and re-derives
  each one's lr from its own ``base_lrs`` on load. hpmesh builds a single
  ``torch.optim.AdamW`` in the trainer, so there is exactly one scheduler and
  the list indirection carries nothing.
* **``Configurable``.** ``Config`` is a plain dataclass whose ``build()`` takes
  the optimizer, the same shape as the checkpointer's and the metrics
  processor's configs.

What is kept is the schedule itself, arithmetic unchanged: a Warmup-Stable-Decay
(WSD) curve (https://arxiv.org/abs/2404.06395) where the stable phase is
implicit -- with no ``decay_ratio`` the decay starts the moment warmup ends,
which is the plain warmup-plus-decay curve.

**Checkpoint behavior.** ``state_dict`` is one integer. A LambdaLR recomputes
its lr from the optimizer's ``base_lrs`` and the shared lambda, so restoring
``last_epoch`` is the whole of the state; the base lrs come back with the
optimizer's own state. That also makes the save and load sides independent of how
many schedulers exist, which is what upstream relies on for resharding.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ["LRScheduler"]


def _wsd_factor(
    current_step: int,
    *,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    decay_type: str,
    min_lr_factor: float,
) -> float:
    """The multiplicative factor to apply to the base lr at ``current_step``.

    A LambdaLR contract: the factor ranges from 1 to ``min_lr_factor``, scaling
    the lr down from its optimizer-configured value.
    """
    warmup_stable_steps = warmup_steps + stable_steps
    if current_step < warmup_steps:
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        assert warmup_steps != 0, "warmup_steps must not be zero to reach this branch"
        return float(current_step / warmup_steps)
    if current_step < warmup_stable_steps:
        return 1.0

    # 0-indexed step, hence + 1 adjustments
    current_step += 1
    assert decay_steps != 0, "decay_steps must not be zero to reach this branch"
    progress = float(current_step - warmup_stable_steps) / decay_steps

    if decay_type == "linear":
        factor = 1 - progress
    elif decay_type == "sqrt":
        factor = 1 - math.sqrt(progress)
    elif decay_type == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")
    return min_lr_factor + (1 - min_lr_factor) * factor


class LRScheduler(Stateful):
    """One ``LambdaLR`` over one optimizer, exposing the loop's three needs.

    Args:
        optimizer: the optimizer whose ``param_groups`` are stepped.
        lr_lambda: maps ``last_epoch`` to the multiplicative lr factor.
        total_steps: the schedule's length, kept only so ``build`` can validate
            it; nothing at step time reads it.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float],
        *,
        total_steps: int,
    ) -> None:
        self.total_steps = total_steps
        self.scheduler = LambdaLR(optimizer, lr_lambda)

    def step(self) -> None:
        """Advance the schedule by one step.

        Called *after* ``optimizer.step()``, so the lr the optimizer just used is
        the one this computes from the previous ``last_epoch`` -- which is why
        the first training step runs at ``lambda(0)``.
        """
        self.scheduler.step()

    def get_metrics(self) -> dict[str, float]:
        """The current lr, keyed so several optimizers could not collide.

        hpmesh has one param group, but the key keeps upstream's shape: a future
        param-group split would show up as ``lr/AdamW/1`` rather than silently
        overwriting ``lr/AdamW``.
        """
        optimizer_name = type(self.scheduler.optimizer).__name__
        last_lrs = self.scheduler.get_last_lr()
        if len(last_lrs) == 1:
            return {f"lr/{optimizer_name}": float(last_lrs[0])}
        return {
            f"lr/{optimizer_name}/{index}": float(value)
            for index, value in enumerate(last_lrs)
        }

    def state_dict(self) -> dict[str, Any]:
        return {"last_epoch": self.scheduler.last_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore ``last_epoch`` only.

        ``LambdaLR`` is stateless apart from that -- the lr is a pure function of
        ``(last_epoch, base_lr)`` -- so ``_last_lr`` is recomputed rather than
        stored. A stateful scheduler (ReduceLROnPlateau and friends) would need
        more, which is why this is spelled out rather than round-tripping the
        whole dict.
        """
        if not state_dict:
            return
        last_epoch = state_dict["last_epoch"]
        self.scheduler.last_epoch = last_epoch
        self.scheduler._step_count = last_epoch + 1
        self.scheduler._last_lr = self.scheduler.get_lr()


@dataclass(kw_only=True)
class LRSchedulerConfig:
    """The WSD schedule's knobs.

    ``decay_ratio`` is the switch that matters. At its default of 0 there is no
    decay phase, so the factor is 1.0 throughout (after any warmup) and the
    learning rate is exactly the one the optimizer was built with -- which is
    what makes the default run comparable to every measurement taken at a
    constant lr. Setting it to a fraction appends a decay covering that fraction
    of ``total_steps``; whatever is left over after warmup and decay is the
    stable phase, at the peak learning rate.
    """

    warmup_steps: int = field(
        default=0,
        metadata={"help": "Steps to linearly ramp the learning rate from 0."},
    )
    total_steps: int | None = field(
        default=None,
        metadata={
            "help": "Length of the schedule. Defaults to --steps. Set it to "
            "decouple the curve from the run length, so a short debugging run "
            "sees the same lrs the full run would."
        },
    )
    decay_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of total_steps spent decaying the learning rate. "
            "0 (the default) never decays, holding the rate at its peak. A "
            "value below 1 leaves the intervening steps at the peak rate "
            "(WSD)."
        },
    )
    decay_type: Literal["linear", "sqrt", "cosine"] = field(
        default="linear",
        metadata={"help": "Shape of the decay phase. Ignored when decay_ratio=0."},
    )
    min_lr_factor: float = field(
        default=0.0,
        metadata={
            "help": "Floor of the decay, as a fraction of the base learning "
            "rate. 0 decays all the way to zero. Ignored when decay_ratio=0."
        },
    )

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.total_steps is not None and self.total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {self.total_steps}")
        if not 0.0 <= self.decay_ratio <= 1.0:
            raise ValueError(
                f"decay_ratio must be in [0, 1], got {self.decay_ratio}"
            )
        if not 0.0 <= self.min_lr_factor < 1.0:
            raise ValueError(
                f"min_lr_factor must be in [0, 1), got {self.min_lr_factor}"
            )

    def build(self, *, optimizer: Optimizer, training_steps: int) -> LRScheduler:
        """Build the scheduler this configuration describes.

        ``training_steps`` is the run's actual length; ``total_steps`` overrides
        it for the curve only. The two are validated against each other rather
        than clamped: a schedule shorter than the run would put the last steps
        past its end, where the decay factor runs off the bottom of the curve
        and turns the learning rate negative -- which ascends the loss instead of
        failing.
        """
        total_steps = (
            self.total_steps if self.total_steps is not None else training_steps
        )
        if total_steps < training_steps:
            raise ValueError(
                f"lr_scheduler.total_steps ({total_steps}) is shorter than the run "
                f"({training_steps} steps). The decay would run past its end and "
                "produce a negative learning rate. Raise total_steps, or drop it "
                "to use the run length."
            )

        warmup_steps = self.warmup_steps
        if warmup_steps > total_steps:
            logger.warning(
                "lr_scheduler.warmup_steps (%d) exceeds total_steps (%d); "
                "clamping the warmup to the whole schedule.",
                warmup_steps,
                total_steps,
            )
            warmup_steps = total_steps

        decay_steps = round(total_steps * self.decay_ratio)
        if warmup_steps + decay_steps > total_steps:
            logger.warning(
                "lr_scheduler warmup (%d) + decay (%d) exceed total_steps (%d); "
                "shortening the decay to %d.",
                warmup_steps,
                decay_steps,
                total_steps,
                total_steps - warmup_steps,
            )
            decay_steps = total_steps - warmup_steps
        # The "+ 1" is a virtual final step. Without it the last real step would
        # land exactly at the end of the decay, where the factor is 0 (linear) --
        # an lr of zero on the final update. With no decay phase it makes the
        # stable region one step longer than the run, which is the point: every
        # real step falls inside it and the factor is a constant 1.0.
        stable_steps = total_steps + 1 - warmup_steps - decay_steps

        lr_lambda = functools.partial(
            _wsd_factor,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            decay_type=self.decay_type,
            min_lr_factor=self.min_lr_factor,
        )
        return LRScheduler(optimizer, lr_lambda, total_steps=total_steps)
