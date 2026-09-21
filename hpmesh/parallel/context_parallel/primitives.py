"""Context-parallel attention redistribution primitives.

CP core idea: shard the sequence across CP ranks; attention all-gathers K/V so
each rank attends its query shard against the full keys.

These primitives are vendored from torchtitan
``models/common/cp_attention.py``. Upstream they are two ``FlexInnerAttention``
subclasses; hpmesh runs HF attention through a registered flex kernel and has no
``FlexInnerAttention``, so the class hierarchy is gone and what is left is the
pair of redistributions themselves -- which is the whole of the CP logic.

The two strategies trade different things:

* :class:`KVAllGatherContextParallel` all-gathers K/V and leaves Q sharded by
  token. Every rank ends up holding the full key/value sequence, so K/V memory
  is *not* saved -- only the attention compute over the queries is.
* :class:`UlyssesContextParallel` all-to-all's the token shard into a head
  shard, so attention runs on the full sequence with ``heads / cp`` heads per
  rank, then converts back. Both memory and compute shard -- at the cost of two
  all-to-alls per layer, which is the expensive part.

Both need a live multi-rank CP axis; they raise rather than silently falling
back, because the fallback would be *wrong*, not just slower: a rank that skips
the redistribution attends its own shard against itself and produces a
confidently incorrect answer.

Shape legend, scoped to this file: ``T`` = tokens, ``H`` = heads.

Not ported, and why: torchtitan's ``cp_shard`` classmethods delegate to
``prepare_context_parallel_input`` for sharding the *model inputs*. That lives
in ``torchtitan/distributed/context_parallel/api.py``, not here, and is a thin
wrapper over torch's own
``torch.distributed.tensor.experimental._context_parallel_shard`` plus BlockMask
sharding -- both available to hpmesh directly from torch. It also derives shard
dims from a per-input SPMD layout dict, which hpmesh does not carry through its
forward path. The hpmesh equivalent lives in ``input_shard.py`` in this package
and is driven by the trainer; :func:`apply_cp` (``apply.py`` here) wires the
attention side.
"""

from __future__ import annotations

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn

from ...utils.spmd_context import spmd_mesh_group
from ..parallel_dims import MeshAxisName

__all__ = [
    "HEAD_DIM",
    "TOKEN_DIM",
    "KVAllGatherContextParallel",
    "UlyssesContextParallel",
    "cp_group",
    "cp_redistribute",
]

# Logical tensor dims the redistributions name. Token 0 is the sequence axis
# under sharded inputs; head 1 is the head axis of a (T, H, *) attention tensor.
TOKEN_DIM = 0
HEAD_DIM = 1


def cp_group() -> dist.ProcessGroup | None:
    """The multi-rank CP process group, or ``None`` when CP is not active.

    ``None`` rather than a size-1 group keeps the callers' "no CP" branch
    honest: collectives on a size-1 group are silent no-ops, so a missing mesh
    would hide behind a run that merely produces the wrong answer.
    """
    return spmd_mesh_group(MeshAxisName.CP.value)


def cp_redistribute(
    x: torch.Tensor,
    *,
    src: spmd.SpmdType,
    dst: spmd.SpmdType,
    backward_op_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Redistribute one tensor across the CP process group.

    A single ``spmd.redistribute``, which routes to an all-gather,
    reduce-scatter or all-to-all depending on ``src``/``dst``. The
    ``(T/cp, *) -> (T, *)`` and ``(T/cp, H, *) -> (T, H/cp, *)`` pairs the
    classes below need are both all-to-alls.

    Args:
        x: the local tensor to redistribute.
        src: its current local SPMD type on the CP axis.
        dst: the type it should have on the CP axis.
        backward_op_dtype: dtype for the backward collective. The forward is
            left alone; only the gradient reduce is cast. ``None`` leaves both
            in the input's dtype.

    Raises:
        RuntimeError: if no multi-rank CP axis is registered. A skipped
            redistribution is a wrong answer, not a slow one, so this does not
            degrade to a no-op.
    """
    group = cp_group()
    if group is None:
        raise RuntimeError(
            "Context parallel distribution requires an active multi-rank CP mesh axis."
        )
    backward_options = (
        {"op_dtype": backward_op_dtype} if backward_op_dtype is not None else None
    )
    return spmd.redistribute(
        x,
        group,
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
        reduce_dtype: dtype of the backward reduce. The default ``float32``
            matches upstream. ``bfloat16`` halves the backward traffic; it is
            the caller's call to trade that precision away.
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
        """Return ``(q, k_all, v_all)``; only K and V are redistributed."""
        k_THK, v_THV = (
            cp_redistribute(
                x,
                src=spmd.S(TOKEN_DIM),
                dst=spmd.R,
                backward_op_dtype=self.reduce_dtype,
            )
            for x in (k_THK, v_THV)
        )
        return q_THK, k_THK, v_THV


class UlyssesContextParallel(nn.Module):
    """CP by all-to-all, exchanging the token shard for a head shard.

    :meth:`shard` turns ``(T/cp, H, *)`` into ``(T, H/cp, *)`` for Q/K/V alike,
    so attention sees the whole sequence with a fraction of the heads and no
    change to its input rank. :meth:`unshard` puts the output back on the token
    axis so the rest of the layer sees the layout it started with.

    The two are kept separate rather than folded into one ``forward``: they sit
    on either side of the attention kernel, and hpmesh's kernel is HF's, not a
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
        return cp_redistribute(out_THV, src=spmd.S(HEAD_DIM), dst=spmd.S(TOKEN_DIM))

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.shard(q_THK, k_THK, v_THV)
