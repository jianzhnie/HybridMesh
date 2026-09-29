"""The config groups as the CLI parser sees them.

``HfArgumentParser`` turns every ``init=True`` dataclass field into a flag, and
three llmtuner fields cannot survive that round trip -- the parser rejects every
value they are given (``invalid dict value``, ``invalid ParamGroupConfig
value``, ``invalid Callable value``):

* ``ModelConfig.arch_overrides`` -- ``dict[str, Any]``;
* ``OptimizerConfig.param_groups`` -- ``list[ParamGroupConfig]``;
* ``CheckpointConfig.purge_exempt`` -- ``Callable[[int], bool] | None``.

All three are configured programmatically (see each field's own docstring), and
a flag that exists only to reject its values is worse than no flag: it shows up
in ``--help`` looking usable. ``init=False`` is the one hook
``HfArgumentParser`` honours -- it skips those fields -- so the CLI is handed a
generated *view* of each group instead: a subclass of the real one whose
programmatic-only fields are re-declared ``init=False``, carrying the base
default over. Defaults, ``__post_init__`` and ``isinstance`` against the real
group all still hold, so nothing downstream can tell the difference; the view
keeps the group's own name so argparse messages read the same.
"""

from __future__ import annotations

import dataclasses
import typing

from .checkpoint import CheckpointConfig
from .data import DataloaderConfig
from .model import ModelConfig
from .optimizer import LRSchedulerConfig, OptimizerConfig
from .parallel import ParallelConfig
from .training import MetricsConfig, ProfilerConfig, TrainingConfig

# The groups the CLI parses, in the order ``LLMTunerConfig.from_groups`` takes
# them. Parser order is significant: ``parse_args_into_dataclasses`` returns one
# instance per class in this exact order, and ``train.py`` zips the two
# together. Kept here, next to the view, so the parser's group set has one
# definition and no caller can quietly drop or reorder a group.
PARSER_GROUPS: tuple[type, ...] = (
    ModelConfig,
    ParallelConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    TrainingConfig,
    CheckpointConfig,
    DataloaderConfig,
    MetricsConfig,
    ProfilerConfig,
)

# The groups that need a view, and which of their fields the CLI cannot carry.
# A new non-scalar field either joins this table or becomes a flag that rejects
# every value it is given; ``cli_view`` fails loudly if a name here goes stale.
PROGRAMMATIC_ONLY: dict[type, frozenset[str]] = {
    ModelConfig: frozenset({"arch_overrides"}),
    OptimizerConfig: frozenset({"param_groups"}),
    CheckpointConfig: frozenset({"purge_exempt"}),
}


def cli_view(cls: type) -> type:
    """``cls`` with its programmatic-only fields hidden from the parser."""
    hidden = PROGRAMMATIC_ONLY.get(cls)
    if not hidden:
        return cls
    unknown = hidden - set(cls.__dataclass_fields__)
    if unknown:
        raise RuntimeError(
            f"{cls.__name__} has no fields {sorted(unknown)}: "
            "config.cli.PROGRAMMATIC_ONLY is stale"
        )
    hints = typing.get_type_hints(cls)
    namespace: dict[str, typing.Any] = {
        "__annotations__": {name: hints[name] for name in hidden}
    }
    for name in hidden:
        base = cls.__dataclass_fields__[name]
        option: dict[str, typing.Any] = {"init": False}
        if base.default is not dataclasses.MISSING:
            option["default"] = base.default
        else:
            option["default_factory"] = base.default_factory
        namespace[name] = dataclasses.field(**option)
    return dataclasses.dataclass(type(cls.__name__, (cls,), namespace))


def cli_groups(groups: tuple[type, ...]) -> tuple[type, ...]:
    """The parser-facing view of ``groups``, in the same order.

    Order is the parser's contract -- ``parse_args_into_dataclasses`` returns
    one instance per class in this order -- so this preserves it.
    """
    return tuple(cli_view(cls) for cls in groups)
