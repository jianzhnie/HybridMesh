# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The learning-rate schedule: linear warmup, stable phase, then decay.

Vendored from torchtitan ``components/optimizer/lr_scheduler.py``. Two things
were dropped:

* **The container.** Upstream ``LRSchedulersContainer`` wraps a list of
  schedulers, one per optimizer in an ``OptimizersContainer``, and re-derives
  each one's lr from its own ``base_lrs`` on load. hpmesh builds a single
  ``torch.optim.AdamW`` in the trainer, so there is exactly one scheduler and
  the list indirection carries nothing.
* **The config.** The knobs moved to ``hpmesh.trainer.config`` with every other
  config in the package (see that module's docstring). What is left here is the
  curve and the ``LambdaLR`` that steps it.

What is kept is the schedule itself, arithmetic unchanged: a Warmup-Stable-Decay
(WSD) curve (https://arxiv.org/abs/2404.06395). ``decay_ratio`` decides how much
of the run the decay covers and whatever is left after warmup is the stable
phase, so the shape is warmup -> stable -> decay with ``decay_ratio=0`` (the
default) degenerating to warmup and then a constant rate.

**Checkpoint behavior.** ``state_dict`` is one integer. A LambdaLR recomputes
its lr from the optimizer's ``base_lrs`` and the shared lambda, so restoring
``last_epoch`` is the whole of the state; the base lrs come back with the
optimizer's own state. That also makes the save and load sides independent of how
many schedulers exist, which is what upstream relies on for resharding.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

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
