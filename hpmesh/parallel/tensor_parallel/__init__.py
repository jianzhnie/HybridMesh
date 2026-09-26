"""Tensor parallelism: the declaration/realizer machinery (``tp``) and the
``apply_tp`` entry point (``apply``), re-exported at the package seam like
every other parallelism family."""

from .apply import apply_tp
from .tp import (
    ColumnParallelLinear,
    ColwiseLinearNoGather,
    RowParallelLinear,
    ShardingConfig,
    colwise,
    rowwise,
)

__all__ = [
    "ColumnParallelLinear",
    "ColwiseLinearNoGather",
    "RowParallelLinear",
    "ShardingConfig",
    "apply_tp",
    "colwise",
    "rowwise",
]
