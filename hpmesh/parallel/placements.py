"""Dense-path SPMD placement primitives.

A placement answers "how is this tensor laid out across the mesh". These are the
dense-transformer ones: parameters replicate over DP/CP and let the caller pick
the TP placement, activations are token-sharded on DP and CP and let the caller
pick the TP placement.

Taken from torchtitan's ``models/common/decoder_sharding.py`` -- only the
helpers hf_sharding needs, not the whole module (the rest of that file is about
torchtitan's own attention/feed-forward modules).

Learning note: ``spmd.V`` means "this axis owns a *value* dimension of the
tensor, described by ``partition_spec``"; ``spmd.R`` / ``spmd.S(dim)`` are the
per-axis local types (replicated / sharded on ``dim``). Every activation layout
must carry a ``partition_spec`` because the DTensor bridge needs to know which
tensor dim each sharding axis maps to.
"""

from __future__ import annotations

import spmd_types as spmd
from spmd_types import SpmdType

from .parallel_dims import MeshAxisName

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP

__all__ = [
    "dense_param_placement",
    "dense_activation_placement",
    "dense_sequence_parallel_placement",
]


def dense_param_placement(*, tp: spmd.PerMeshAxisSpmdType) -> SpmdType:
    """Placement for dense-path params/buffers.

    DP/CP axes are ``R``; the DTensor bridge unfolds DP into storage axes. TP
    placement is caller-specified (``R`` to replicate, ``S(0)``/``S(1)`` to
    shard an output/input feature dim).
    """
    return SpmdType({DP: spmd.R, CP: spmd.R, TP: tp})


def dense_activation_placement(
    *,
    tp: spmd.PerMeshAxisSpmdType,
    cp: spmd.PerMeshAxisSpmdType,
) -> SpmdType:
    """Placement for dense-path activations with a batch dim.

    DP token-shards. CP and TP placements are caller-specified; whichever of the
    two actually shards becomes a ``V`` axis named in the ``partition_spec``, so
    e.g. ``tp=S(-1)`` shards the trailing (hidden) dim and ``cp=S(0)`` shards the
    leading (batch) dim. Tensor dims absent from the spec are replicated.
    """
    cp_shards_tokens = isinstance(cp, spmd.Shard)
    tp_shards_features = isinstance(tp, spmd.Shard)
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V if cp_shards_tokens else cp,
            TP: spmd.V if tp_shards_features else tp,
        },
        partition_spec=spmd.PartitionSpec(
            (DP, CP) if cp_shards_tokens else DP,
            TP if tp_shards_features else None,
        ),
    )


def dense_sequence_parallel_placement() -> SpmdType:
    """Sequence-parallel activation: the token dim is split across DP, CP and TP.

    This is the layout the sequence-parallel region carries between projections:
    each rank holds a contiguous slice of the sequence rather than the whole
    sequence with a sharded feature dim.
    """
    return SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.V},
        partition_spec=spmd.PartitionSpec((DP, CP, TP), None),
    )
