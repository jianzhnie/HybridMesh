"""The learning-rate schedule: linear warmup, stable phase, then decay.

Vendored from torchtitan ``components/optimizer/lr_scheduler.py``: the same
``LRSchedulersContainer``, driven by the same ``LambdaLR``, over the same
Warmup-Stable-Decay (WSD) curve (https://arxiv.org/abs/2404.06395).

The container is not optional. A ``LambdaLR`` reads ``lr`` off the first of its
optimizer's ``param_groups``, and an ``OptimizersContainer`` built the way the
loop needs it has none: its ``param_groups`` is the parameter view
``Optimizer.__init__`` merges, and no group carries an ``lr`` of its own. So the
scheduler has to be handed the *inner* optimizers, one ``LambdaLR`` each -- which
is exactly what this class holds, and why it exists rather than a bare
``LambdaLR``.

Departures from upstream:

* **The config is not defined here.** The knobs live in ``LRSchedulerConfig`` in
  ``hpmesh.trainer.config``, with every other config in the package;
  :func:`build_lr_scheduler` is the seam that turns one into the container.
  Upstream's ``Config.build`` is that seam, spelled with torchtitan's config
  system.
* **``state_dict`` reports ``last_epoch`` and nothing else.** ``LambdaLR`` is
  stateless apart from it -- the lr is a pure function of ``(last_epoch,
  base_lr)`` -- so ``_last_lr`` is recomputed on load rather than stored. The
  base lrs come back with the optimizer's own state. A stateful scheduler
  (``ReduceLROnPlateau`` and friends) would need more, which is why the load path
  spells this out instead of round-tripping the whole dict.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ...utils.logger_utils import get_logger

if TYPE_CHECKING:
    # ``trainer.config`` imports this package, so the runtime import happens
    # inside ``build_lr_scheduler`` -- the one place a config's fields are read.
    from ...trainer.config import LRSchedulerConfig

logger = get_logger(__name__)

__all__ = ["build_lr_scheduler", "LRSchedulersContainer"]


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


class LRSchedulersContainer(Stateful):
    """One ``LambdaLR`` per inner optimizer, stepped together.

    The training loop drives this like a single scheduler: ``step`` advances
    them all, ``get_metrics`` reports the lr of each.

    All of them share one lambda. They can still diverge, because a ``LambdaLR``
    scales its own optimizer's ``base_lrs`` -- so a run whose param groups carry
    different base learning rates gets the right curve per group without a
    second lambda.

    Args:
        optimizers: one scheduler is built per entry, in order.
        lr_lambda: maps ``last_epoch`` to the multiplicative lr factor.
        total_steps: the schedule's length, resolved by :func:`build_lr_scheduler`
            from the config and the run length. Kept only so callers can assert
            on the curve's extent; nothing at step time reads it.
    """

    def __init__(
        self,
        optimizers: list[Optimizer],
        lr_lambda: Callable[[int], float],
        *,
        total_steps: int,
    ) -> None:
        if not optimizers:
            raise ValueError(
                "LRSchedulersContainer needs at least one optimizer to schedule."
            )
        self.total_steps = total_steps
        self.schedulers = [LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]

    def __iter__(self) -> Iterator[LambdaLR]:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        """Advance the schedule by one step.

        Called *after* ``optimizer.step()``, so the lr the optimizer just used is
        the one this computes from the previous ``last_epoch`` -- which is why
        the first training step runs at ``lambda(0)``.
        """
        for scheduler in self.schedulers:
            scheduler.step()

    def get_metrics(self) -> dict[str, float]:
        """The current lr of each optimizer, keyed so several cannot collide.

        The key keeps upstream's shape: a run with several schedulers, or a
        param-group split, shows up as ``lr/AdamW/1`` rather than silently
        overwriting ``lr/AdamW``.
        """
        metrics: dict[str, float] = {}
        for scheduler in self.schedulers:
            optimizer_name = type(scheduler.optimizer).__name__
            last_lrs = scheduler.get_last_lr()
            if len(last_lrs) == 1:
                metrics[f"lr/{optimizer_name}"] = float(last_lrs[0])
                continue
            for index, value in enumerate(last_lrs):
                metrics[f"lr/{optimizer_name}/{index}"] = float(value)
        return metrics

    def state_dict(self) -> dict[str, Any]:
        # One integer, not one per scheduler: every scheduler is stepped
        # together from the same lambda, so they share a step count by
        # construction. Storing it once is what lets a checkpoint survive
        # resharding to a different number of optimizers.
        return {"last_epoch": self.schedulers[0].last_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore ``last_epoch`` on every scheduler.

        ``LambdaLR`` is stateless apart from that -- the lr is a pure function of
        ``(last_epoch, base_lr)`` -- so ``_last_lr`` is recomputed rather than
        stored, per scheduler, from its own optimizer's ``base_lrs``.
        """
        if not state_dict:
            return
        last_epoch = state_dict["last_epoch"]
        for scheduler in self.schedulers:
            scheduler.last_epoch = last_epoch
            scheduler._step_count = last_epoch + 1
            scheduler._last_lr = scheduler.get_lr()


def build_lr_scheduler(
    config: LRSchedulerConfig,
    *,
    optimizers: list[Optimizer],
    training_steps: int,
) -> LRSchedulersContainer:
    """Build the schedule a :class:`~hpmesh.trainer.config.LRSchedulerConfig`
    describes.

    ``training_steps`` is the run's actual length; the config's ``total_steps``
    overrides it for the curve only. The two are validated against each other
    rather than clamped: a schedule shorter than the run would put the last steps
    past its end, where the decay factor runs off the bottom of the curve and
    turns the learning rate negative -- which ascends the loss instead of
    failing.
    """
    total_steps = (
        config.total_steps if config.total_steps is not None else training_steps
    )
    if total_steps < training_steps:
        raise ValueError(
            f"lr_scheduler.total_steps ({total_steps}) is shorter than the run "
            f"({training_steps} steps). The decay would run past its end and "
            "produce a negative learning rate. Raise total_steps, or drop it "
            "to use the run length."
        )

    warmup_steps = config.warmup_steps
    if warmup_steps > total_steps:
        logger.warning(
            "lr_scheduler.warmup_steps (%d) exceeds total_steps (%d); "
            "clamping the warmup to the whole schedule.",
            warmup_steps,
            total_steps,
        )
        warmup_steps = total_steps

    decay_steps = round(total_steps * config.decay_ratio)
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
        decay_type=config.decay_type,
        min_lr_factor=config.min_lr_factor,
    )
    return LRSchedulersContainer(optimizers, lr_lambda, total_steps=total_steps)
