"""Step 4: context parallelism OR expert parallelism (pick one to go deep).

CP core idea: shard the sequence across CP ranks; attention all-gathers K/V so
each rank attends its query shard against the full keys.
EP core idea (MoE): shard experts across ranks; an all-to-all routes each token
to its expert's rank and back.

This module owns the CP attention primitives, vendored from torchtitan
``models/common/cp_attention.py``. Upstream they are two ``FlexInnerAttention``
subclasses; hpmesh runs HF attention through a registered flex kernel and has no
``FlexInnerAttention``, so the class hierarchy is gone and what is left is the
pair of redistributions themselves -- which is the whole of the CP logic.

The two strategies trade different things:

* :class:`KVAllGatherContextParallel` all-gathers K/V and leaves Q sharded by
  token. Every rank holds the full key/value sequence; memory for K/V is *not*
  saved, only the attention compute over the queries is.
* :class:`UlyssesContextParallel` all-to-all's the token shard into a head
  shard, so attention runs on the full sequence with ``heads / cp`` heads per
  rank, then converts back. Both Q/K/V memory and compute shard -- at the cost
  of two all-to-alls per layer, which is the expensive part.

Both need a live multi-rank CP axis; they raise rather than silently falling
back, because the fallback would be *wrong*, not just slower: a rank that skips
the redistribution attends its own shard against itself and produces a
confidently incorrect result.

Shape legend, scoped to this file: ``T`` = tokens, ``H`` = heads.

Not ported, and why: torchtitan's ``cp_shard`` classmethods delegate to
``prepare_context_parallel_input`` for sharding the *model inputs*. That lives
in ``torchtitan/distributed/context_parallel/api.py``, not here, and is a thin
wrapper over torch's own
``torch.distributed.tensor.experimental._context_parallel_shard`` plus BlockMask
sharding -- both available to hpmesh directly from torch. It also reads spmd
layouts out of a per-input sharding dict, which hpmesh does not carry through
its forward path. Wire it from torch when a real CP training step is added.
"""

from __future__ import annotations

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from ..trainer.config import HybridMeshConfig
from .parallel_dims import MeshAxisName
from .spmd_types import current_spmd_mesh

__all__ = [
    "HEAD_DIM",
    "TOKEN_DIM",
    "KVAllGatherContextParallel",
    "UlyssesContextParallel",
    "apply_cp_ep",
    "cp_redistribute",
]

# Logical tensor dims the redistributions name. Token 0 is the sequence axis
# under sharded inputs; head 1 is the head axis of a (T, H, *) attention tensor.
TOKEN_DIM = 0
HEAD_DIM = 1

_CP = MeshAxisName.CP.value


def _cp_mesh() -> DeviceMesh | None:
    """The 1D CP mesh from the registered SPMD state, or ``None``.

    ``spmd.redistribute`` takes a single mesh axis, so a multi-axis dense mesh
    is sliced down to its ``cp`` axis first. Returning ``None`` rather than a
    size-1 mesh keeps the callers' "no CP" branch honest: a size-1 group would
    make the collectives no-ops and hide a missing mesh behind a working run.
    """
    mesh = current_spmd_mesh()
    if mesh is None:
        return None
    names = mesh.mesh_dim_names or ()
    if _CP not in names:
        return None
    if names == (_CP,):
        return mesh
    sub = mesh[_CP]
    return sub if sub.ndim == 1 else None


def cp_redistribute(
    x: torch.Tensor,
    *,
    src: spmd.SpmdType,
    dst: spmd.SpmdType,
    backward_op_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Redistribute one tensor across the CP axis.

    A single ``spmd.redistribute`` over the CP process group, which routes to an
    all-gather, reduce-scatter or all-to-all depending on ``src``/``dst``. The
    ``(T/cp, *) -> (T, *)`` and ``(T/cp, H, *) -> (T, H/cp, *)`` pairs the
    context-parallel classes below need are both all-to-alls.

    Args:
        x: the local tensor to redistribute.
        src: its current local SPMD type on the CP axis.
        dst: the type it should have on the CP axis.
        backward_op_dtype: dtype for the backward collective. The forward is
            left alone; only the gradient reduce is cast. ``None`` leaves both
            in the input's dtype.

    Raises:
        RuntimeError: if no multi-rank CP axis is registered. A missing
            redistribution is a wrong answer, not a slow one, so this does not
            degrade to a no-op.
    """
    mesh = _cp_mesh()
    if mesh is None:
        raise RuntimeError(
            "Context parallel distribution requires an active multi-rank CP "
            "mesh axis."
        )
    backward_options = (
        {"op_dtype": backward_op_dtype} if backward_op_dtype is not None else None
    )
    return spmd.redistribute(
        x,
        mesh,
        src=src,
        dst=dst,
        backward_options=backward_options,
    )


class KVAllGatherContextParallel(nn.Module):
    """CP by all-gathering K and V, with Q left sharded by token.

    Each rank keeps its own query tokens and attends them against the assembled
    key/value sequence, so the gathered K/V is full-length on every rank. Q and
    the attention output stay token-sharded: no redistribution is needed on
    either side of the attention call.

    Args:
        reduce_dtype: dtype of the backward reduce-scatter. The default
            ``float32`` matches upstream. ``bfloat16`` halves the backward
            traffic; it is the caller's call to trade that precision away.
    """

    def __init__(self, *, reduce_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.reduce_dtype = reduce_dtype

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(q, k_all, v_all)``; only K/V are redistributed."""
        k_THK, v_THV = (
            cp_redistribute(
                x, src=spmd.S(TOKEN_DIM), dst=spmd.R, backward_op_dtype=self.reduce_dtype
            )
            for x in (k_THK, v_THV)
        )
        return q_THK, k_THK, v_THV


class UlyssesContextParallel(nn.Module):
    """CP by all-to-all, exchanging the token shard for a head shard.

    ``shard`` turns ``(T/cp, H, *)`` into ``(T, H/cp, *)`` for Q/K/V alike, so
    attention sees the whole sequence with a fraction of the heads and no
    variation in its input shape. ``unshard`` puts the output back on the token
    axis so the rest of the layer sees the same layout it started with.

    The two calls are deliberately not fused into one ``forward``: they sit on
    either side of the attention kernel, and hpmesh's kernel is HF's, not a
    module this file can wrap.
    """

    def shard(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(T/cp, H, *) -> (T, H/cp, *)`` for Q, K and V."""
        return tuple(
            cp_redistribute(x, src=spmd.S(TOKEN_DIM), dst=spmd.S(HEAD_DIM))
            for x in (q_THK, k_THK, v_THV)
        )

    def unshard(self, out_THV: torch.Tensor) -> torch.Tensor:
        """``(T, H/cp, V) -> (T/cp, H, V)``, back to sharded tokens."""
        return cp_redistribute(
            out_THV, src=spmd.S(HEAD_DIM), dst=spmd.S(TOKEN_DIM)
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.shard(q_THK, k_THK, v_THV)


def apply_cp_ep(
    model: nn.Module, mesh: DeviceMesh | None, cfg: HybridMeshConfig
) -> nn.Module:
    """Wire CP/EP onto a model.

    Still unimplemented for both. The CP attention primitives above are in
    place, but a runnable CP step also needs the model inputs sharded along the
    sequence axis (see the module docstring) and an attention hook that calls
    them around the HF kernel. EP has no dispatcher yet either.
    """
    if cfg.cp == 1 and cfg.ep == 1:
        return model
    raise NotImplementedError(
        "CP/EP is step 4 of the learning path: implement KV all-gather (CP) or an "
        "all-to-all token dispatcher (EP) here."
    )
