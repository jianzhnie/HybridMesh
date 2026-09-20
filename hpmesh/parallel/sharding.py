# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The sharding spec for one module: which axis shards which state.

``ShardingConfig`` is a plain description -- state names and activation names,
each mapped to an ``SpmdType`` keyed by mesh axis -- and ``resolve_placements``
is what turns one into DTensor placements for a given mesh. Splitting the two is
what makes the spec mesh-agnostic: the same declaration resolves against a
1-D dp mesh and a full dp/cp/tp mesh, and a missing axis is a ``ValueError``
rather than a silently-replicated tensor.
"""

from dataclasses import dataclass, field

import spmd_types as spmd
from spmd_types import SpmdType
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Placement, Replicate, Shard

from .parallel_dims import MeshAxisName, unfold_dp_axis

__all__ = [
    "ShardingConfig",
    "resolve_placements",
    "spmd_axes",
]


def spmd_axes(layout: spmd.SpmdType) -> tuple[MeshAxisName, ...]:
    """Return and validate the named mesh axes used by an SPMD layout."""
    axes = []
    for axis in layout.local_type:
        if not isinstance(axis, str):
            raise TypeError(
                f"TorchTitan SPMD layouts require named mesh axes, got {axis!r}"
            )
        axes.append(MeshAxisName(axis))
    return tuple(axes)


def _per_axis_types(
    layout: spmd.SpmdType,
) -> dict[MeshAxisName, spmd.PerMeshAxisSpmdType]:
    result: dict[MeshAxisName, spmd.PerMeshAxisSpmdType] = {}
    for axis, axis_type in layout.local_type.items():
        if not isinstance(axis, str):
            raise TypeError(
                f"TorchTitan SPMD layouts require named mesh axes, got {axis!r}"
            )
        result[MeshAxisName(axis)] = axis_type
    if layout.partition_spec is not None:
        for dim, entry in enumerate(layout.partition_spec):
            axes = (
                () if entry is None else entry if isinstance(entry, tuple) else (entry,)
            )
            for axis in axes:
                if not isinstance(axis, str):
                    raise TypeError(
                        f"TorchTitan SPMD layouts require named mesh axes, got {axis!r}"
                    )
                result[MeshAxisName(axis)] = spmd.S(dim)
    return result


@dataclass(kw_only=True, slots=True)
class ShardingConfig:
    """Declarative sharding for a module's states and activations.

    All placements use ``SpmdType`` keyed by mesh axis names. A module holds one
    of these as ``_sharding_config`` and the SPMD engine reads it off the
    modules it walks; ``resolve_placements`` is what converts the declarations
    to concrete ``Placement`` tuples at that point.

    Completely dtype-agnostic at this moment -- quantization (Float8/MXFP8) is
    orthogonal.

    Redistribution is expressed as a (source, destination) pair: src declares
    what the tensor's placement is entering the boundary, dst declares the
    desired placement after redistribution. Both sides are explicit because
    local SPMD types are erased at runtime.

    Attributes:
        state_shardings: Parameter/buffer SPMD layouts. Outer dict keys are
            state names.
            e.g. ``{"weight": {TP: Shard(0)}}`` for colwise.
        in_src_shardings: Source placements of inputs, keyed by ``forward()``
            arg name. Used to assert the input's local SPMD type and declare
            the source side of the input redistribution pair.
            e.g. ``{"x": {TP: Shard(1)}}``.
        in_dst_shardings: Desired input placements after redistribution,
            keyed by ``forward()`` arg name.
            e.g. ``{"x": {TP: Replicate()}}`` for all-gather.
            ``None`` means no input redistribution.
        out_src_shardings: Source SPMD type of the forward's output. When
            ``local_spmd`` is set, this declares the local region's output
            type. Accepts a single
            ``SpmdType`` (single-output case) or a tuple (multi-
            output case). ``None``
            means "infer from the output" or that there is no local region.
            e.g. ``{TP: Partial()}`` for the MoE wrapper.
        out_dst_shardings: Desired output placement after redistribution.
            e.g. ``{TP: Shard(1)}`` for reduce-scatter to sequence-parallel.
            ``None`` means no output redistribution.
        local_spmd: If true, wraps forward with ``spmd.no_typecheck()`` using
            input types from ``in_dst_shardings`` and output types from
            ``out_src_shardings``.
    """

    state_shardings: dict[str, SpmdType] = field(default_factory=dict)
    in_src_shardings: dict[str, SpmdType] | None = None
    in_dst_shardings: dict[str, SpmdType] | None = None
    out_src_shardings: SpmdType | tuple[SpmdType, ...] | None = None
    out_dst_shardings: SpmdType | None = None
    local_spmd: bool = False

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


def resolve_placements(
    layout: SpmdType,
    mesh: DeviceMesh,
) -> tuple[Placement, ...]:
    """Resolve an SPMD type against a mesh in axis order.

    Every sharding_config must explicitly declare a placement for every mesh axis
    it will be applied against. Missing declarations raise ``ValueError``;
    extra declarations (axes not in the mesh) are ignored.

    ``Shard(d)`` or ``Partial`` on a size-1 mesh axis is normalized to
    ``Replicate()`` -- all three are operationally identical on a 1-rank axis
    (no data is split, and a sum over a single rank is the identity), but
    DTensor's op rules (placement-equality, view/reshape strict mode, ...)
    treat them as distinct and reject ``Shard``/``Partial`` in places where
    ``Replicate`` would work.
    """
    # TODO(fegin): remove the size-1 ``Shard(d)``/``Partial`` to ``Replicate()``
    # conversion once FlexShard replaces ``fully_shard``.
    assert mesh.mesh_dim_names is not None, "DeviceMesh must have named axes"
    concrete_axis_types = {}
    for axis_name, axis_type in _per_axis_types(layout).items():
        for concrete_axis_name in unfold_dp_axis(axis_name):
            concrete_axis_types[concrete_axis_name] = axis_type

    result = []
    for i, axis_name in enumerate(mesh.mesh_dim_names):
        key = MeshAxisName(axis_name)
        if key not in concrete_axis_types:
            raise ValueError(
                f"ShardingConfig does not declare a placement for mesh axis "
                f"{axis_name!r}. Declared: "
                f"{sorted(k.value for k in spmd_axes(layout))}; "
                f"required: {list(mesh.mesh_dim_names)}."
            )
        p = spmd.spmd_type_to_dtensor_placement(concrete_axis_types[key])
        if isinstance(p, Shard | Partial) and mesh.size(i) == 1:
            p = Replicate()
        result.append(p)
    return tuple(result)
