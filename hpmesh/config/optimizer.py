"""Optimizer, learning-rate schedule, param-group and EMA configs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from hpmesh.errors import ConfigError
from hpmesh.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass(kw_only=True)
class LRSchedulerConfig:
    """The WSD schedule's knobs.

    The runtime side -- the ``LambdaLR`` this builds -- is
    ``hpmesh.components.optimizer.LRSchedulersContainer``.

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
            raise ConfigError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.total_steps is not None and self.total_steps < 1:
            raise ConfigError(f"total_steps must be >= 1, got {self.total_steps}")
        if not 0.0 <= self.decay_ratio <= 1.0:
            raise ConfigError(f"decay_ratio must be in [0, 1], got {self.decay_ratio}")
        if not 0.0 <= self.min_lr_factor < 1.0:
            raise ConfigError(
                f"min_lr_factor must be in [0, 1), got {self.min_lr_factor}"
            )


@dataclass
class ParamGroupConfig:
    """One parameter group and the optimizer that owns it.

    A list of these, in order, is what ``OptimizerConfig.param_groups`` means:
    each parameter is claimed by the first entry whose ``pattern`` matches its
    FQN, so a catch-all ``.*`` belongs last.

    ``optimizer_name`` and ``optimizer_kwargs`` define the group's optimizer
    completely -- there is no inheritance from ``OptimizerConfig``'s flat
    scalars, which supply the *default* catch-all group and nothing else. A
    config that omits ``lr`` here therefore fails in the optimizer constructor
    rather than silently adopting a value set somewhere else.

    There is no CLI flag for this: ``HfArgumentParser`` builds one flag per
    dataclass *field*, and a list of nested dataclasses has no flag spelling.
    Groups are set from code, by constructing an ``OptimizerConfig`` with
    ``param_groups=[...]``. See that class's ``__post_init__`` for how the flat
    scalars become a group when this list is left empty.
    """

    pattern: str
    """Regex matched against parameter FQNs, e.g. ``r".*\\.bias$"``, ``r".*"``."""

    optimizer_name: str
    """Optimizer class for this group's parameters: ``"Adam"`` or ``"AdamW"``."""

    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the optimizer constructor. Must include everything
    required (``lr`` above all); nothing is filled in from the enclosing config.
    Entries override the run-wide implementation kwargs, so a group can ask for
    ``fused=False`` where the run default is fused."""


@dataclass
class OptimizerConfig:
    """Optimizer (adamw only for now -- the learning path needs just one).

    The schedule lives here rather than in ``TrainingConfig`` because it
    scales this group's learning rate: a lr with no schedule is the degenerate
    case of one, and splitting them would let the two be set independently.
    """

    learning_rate: float = field(default=3e-4, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay"})
    betas: list[float] = field(
        default_factory=lambda: [0.9, 0.999],
        metadata={
            "help": "AdamW (beta1, beta2). Pass as two values: "
            "--betas 0.9 0.95. The default is torch's; torchtitan's reference "
            "LLM recipe uses (0.9, 0.95), which is a NUMERIC change."
        },
    )
    eps: float = field(
        default=1e-8,
        metadata={"help": "AdamW epsilon (denominator floor)."},
    )
    implementation: Literal["fused", "foreach", "for-loop"] = field(
        default="fused",
        metadata={
            "help": "Optimizer kernel. 'fused' is CUDA-only and falls back to "
            "the for-loop kernel elsewhere; on CPU all three are bit-identical. "
            "torchtitan's default."
        },
    )
    param_groups: list[ParamGroupConfig] = field(
        default_factory=list,
        metadata={
            "help": "Per-parameter-group optimizers. Empty (the default) means "
            "one catch-all group built from the flat scalars above. No CLI flag "
            "-- nested dataclass lists cannot be parsed; set it from code."
        },
    )
    lr_scheduler_config: LRSchedulerConfig = field(
        default_factory=LRSchedulerConfig,
        metadata={"help": "Learning-rate schedule (see components/optimizer)."},
    )

    def __post_init__(self) -> None:
        # ``list`` is what the parser can build from two CLI values, but the
        # optimizer wants a tuple and the field must not be mutable: a reused
        # ``HfArgumentParser`` hands every instance the SAME default list (the
        # factory runs once, not per instance), so an in-place edit would leak
        # across runs. Normalizing here makes that unreachable, and the length
        # check is what turns a typo like ``--betas 0.9`` into an error rather
        # than a one-element betas that torch rejects deep in a step.
        betas = tuple(self.betas)
        if len(betas) != 2:
            raise ConfigError(
                f"betas must have exactly 2 entries (beta1, beta2), got "
                f"{len(betas)}: {betas}. Pass both: --betas 0.9 0.95."
            )
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ConfigError(f"betas must each be in [0, 1), got {betas}")
        self.betas = betas

        # The degenerate grouping: no explicit param_groups means one catch-all
        # group over every trainable parameter, built from the flat scalars.
        # Synthesized here rather than in the container so there is exactly one
        # description of the default, and so a caller that reads
        # ``cfg.param_groups`` sees the groups a run actually uses.
        if not self.param_groups:
            self.param_groups = [
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": self.learning_rate,
                        "weight_decay": self.weight_decay,
                        "betas": self.betas,
                        "eps": self.eps,
                    },
                )
            ]

    @property
    def lr_scheduler(self) -> LRSchedulerConfig:
        """The schedule's config.

        A property backed by ``lr_scheduler_config`` so the flat
        ``cfg.lr_scheduler`` spelling works at the call site, while the parser
        still sees a real field to generate flags from. See
        ``TrainingConfig.checkpoint`` for the same shape.
        """
        return self.lr_scheduler_config


@dataclass(kw_only=True)
class EMAConfig:
    """Online EMA of model weights (see ``components/optimizer/ema.py``).

    Set ``training.ema_config`` to one of these to turn EMA on; the default
    ``None`` means no EMA is built and the run pays nothing for it -- there is
    no CLI flag (a nested dataclass does not survive ``HfArgumentParser``;
    ``ParamGroupConfig`` is programmatic-only for the same reason). The field
    names and semantics are upstream's.
    """

    decay: float | None = None
    """Fixed decay per firing: ``ema = decay * ema + (1 - decay) * param``.
    If None (default), computed dynamically from ``half_life_fraction``."""

    half_life_fraction: float = 0.05
    """Used when ``decay`` is None:
    ``decay = 2 ** (-1 / (half_life_fraction * num_updates))``. Keeps roughly
    the most recent ``half_life_fraction`` share of updates dominant."""

    start_step: int = 0
    """Last trainer step before EMA tracking begins, so the first update fires
    at ``start_step + update_every_n_steps``."""

    step_bias: int = 0
    """Offset added to the firing count when computing ``num_updates``, for
    renumbering a new training phase without resetting EMA aging. Measured in
    EMA firings, not raw steps. A normal resume needs no bias."""

    update_every_n_steps: int = 1
    """Only fire the EMA update every N real optimizer steps."""

    buffer_patterns: list[str] = field(default_factory=list)
    """Regex patterns (``re.search``, against buffer FQNs) selecting which
    buffers also get an EMA tracked -- e.g. an MoE's ``expert_bias_E``. Empty
    (default): no buffers tracked."""

    def __post_init__(self) -> None:
        if self.update_every_n_steps < 1:
            raise ConfigError("ema.update_every_n_steps must be greater than 0.")
        if not math.isfinite(self.half_life_fraction):
            raise ConfigError("ema.half_life_fraction must be finite.")
        if self.half_life_fraction <= 0:
            raise ConfigError("ema.half_life_fraction must be greater than 0.")
        if self.step_bias < 0:
            raise ConfigError(
                "ema.step_bias must not be negative; it is added to the firing "
                "count, and a non-positive count has no decay."
            )
        if self.decay is not None and not (
            math.isfinite(self.decay) and 0 <= self.decay < 1
        ):
            raise ConfigError(
                "ema.decay must be finite and in [0, 1); "
                "decay=1 never updates the EMA."
            )
        # A fixed decay replaces the half-life schedule outright, so a
        # half_life_fraction set alongside it would do nothing.
        default_half_life = (
            type(self).__dataclass_fields__["half_life_fraction"].default
        )
        if self.decay is not None and self.half_life_fraction != default_half_life:
            logger.warning(
                "ema.half_life_fraction=%s is ignored because ema.decay=%s is "
                "set; the decay is then fixed and the half-life schedule is "
                "never used. Leave decay unset to use half_life_fraction.",
                self.half_life_fraction,
                self.decay,
            )
