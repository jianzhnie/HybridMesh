"""Model components that fold the TP collectives into their GEMMs.

Vendored from torchtitan ``models/common/dist_gemm.py``. These are drop-in
replacements for the stock QKV, output and SwiGLU projections: they move the TP
collective inside the GEMM, over the autograd Functions in
``hpmesh/parallel/tensor_parallel/linear.py`` (which holds the collective+GEMM
math itself; this file is only the wiring and the fallbacks).

What changed from upstream:

* The ``Configurable`` carriers are gone. Upstream's empty ``Config`` subclasses
  exist solely so ``Config.build()`` binds to the fused class rather than the
  stock one -- with no config system they have nothing left to do. Each class
  now takes its sizes/weights as keyword args, like the rest of hpmesh.
* ``DistGEMMFeedForward`` subclasses :class:`~hpmesh.models.common.feed_forward.
  FeedForward`, as it does upstream: the fused and unfused paths share the weight
  layout (``w13`` holding the interleaved gate and up) and the activation split,
  so only the two GEMMs differ. It overrides ``forward`` and falls back to the
  inherited implementation when TP is off.
* ``torch_remat`` is gone. Upstream wraps each projection in
  ``remat.region(..., recompute=...)``, which only steers activation
  checkpointing; calling the projection directly is the same arithmetic.
* ``current_spmd_mesh`` is read through hpmesh's ``spmd_types`` helper.

The fallback matters: when TP is not active there is no collective to fuse, so
each module runs its plain path. That keeps a TP=1 run working, but it also means
a misconfiguration looks like success -- hence the warning rather than silence.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn as nn

from hpmesh.models.common.feed_forward import FeedForward
from hpmesh.models.common.qkv import QKVLinear
from hpmesh.parallel.spmd_types import current_spmd_mesh
from hpmesh.parallel.tensor_parallel.linear import (
    AllGatherLinear,
    LinearReduceScatter,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AllGatherFusedQKVLinear",
    "DistGEMMFeedForward",
    "RowParallelLinear",
    "validate_dist_gemm_preconditions",
]

# Shape suffix legend, scoped to this file:
#   T = tokens, D = model dimension, N = output features, F = ffn hidden dim,
#   H = head dimension.

_WARNED_NO_TP = False


def _warn_once_no_tp_overlap() -> None:
    """Say so when the dist-GEMM modules were selected but TP is not on.

    Otherwise the fallback is indistinguishable from the feature working: the run
    succeeds, the loss looks fine, and nothing ran fused. The preconditions cover
    the wrong-backend and SP-disabled cases with hard errors, but TP=1 has to stay
    runnable, so it warns instead.
    """
    global _WARNED_NO_TP
    if not _WARNED_NO_TP:
        _WARNED_NO_TP = True
        logger.warning(
            "tp_gemm_backend='dist_gemm' selected but tensor parallelism is not "
            "active; running the standard feed-forward path without collective "
            "overlap."
        )


def _tp_group_from_context() -> dist.ProcessGroup | None:
    """The TP process group from the current SPMD mesh context, or None.

    Resolved per forward rather than captured at parallelize time, so these
    modules need no parallelize override and hold no group state.

    None means "run the stock projection": either no mesh context or a TP degree
    of 1, in which case there is no collective to fuse.
    """
    mesh = current_spmd_mesh()
    if mesh is None or "tp" not in (mesh.mesh_dim_names or ()):
        return None
    tp_group = mesh.get_group("tp")
    return tp_group if tp_group.size() > 1 else None


def validate_dist_gemm_preconditions(*, enable_sp: bool) -> None:
    """Reject configurations the fused modules cannot serve.

    Called from the sharding setup, the first point that sees both the selected
    modules and the parallelism settings. The condition is not detectable from
    inside a module at runtime: under spmd_types an activation is a plain local
    tensor with no placements to inspect.

    Raises:
        ValueError: if sequence parallelism is off. The fused GEMMs *replace* the
            SP all-gather and reduce-scatter, so with SP disabled there is
            nothing for them to fuse with and the ranks would compute different
            things.
    """
    if not enable_sp:
        raise ValueError(
            "tp_gemm_backend='dist_gemm' requires "
            "parallelism.enable_sequence_parallel; the fused GEMMs replace the SP "
            "all-gather and reduce-scatter, so there is nothing for them to fuse "
            "with SP disabled."
        )


class AllGatherFusedQKVLinear(QKVLinear):
    """Fused QKV projection whose forward all-gathers the TP sequence shard.

    Otherwise identical to :class:`QKVLinear`: same weights, same checkpoint
    keys, same output split. Only the projection step differs, which is why it
    overrides ``_project`` rather than ``forward``.
    """

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super()._project(x)

        return AllGatherLinear.apply(
            x,
            self.wqkv.weight,
            self.wqkv.bias,
            tp_group,
            tp_group.group_name,
        )


class RowParallelLinear(nn.Module):
    """A rowwise linear whose matmul is fused with the TP reduce-scatter.

    Named for the role it fills rather than the collective it performs, so it
    does not read like the :class:`LinearReduceScatter` autograd Function it
    calls. Nothing here is attention-specific -- it serves attention's output
    projection and the FFN's down projection alike.

    Args:
        in_features: input feature count (the weight's second dim).
        out_features: output feature count (the weight's first dim).
        bias: whether to hold a bias. The reduce-scatter path handles a partial
            bias, so this is free to be ``None``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return torch.nn.functional.linear(x, self.weight, self.bias)

        return LinearReduceScatter.apply(
            x,
            self.weight,
            self.bias,
            tp_group,
            tp_group.group_name,
        )


class DistGEMMFeedForward(FeedForward):
    """SwiGLU feed-forward with both TP collectives folded into its GEMMs.

    The fused ``w13`` projection consumes an all-gather of the sequence shard;
    ``w2`` is row-parallel and reduce-scatters back to a sequence shard.

    ``w13`` holds the gate and up projections *interleaved* along its output dim
    -- ``[g0, u0, g1, u1, ...]`` -- so ``w1`` and ``w3`` can be recovered by
    unflattening on the last axis. That is the layout an HF-style checkpoint
    expects, and the pairing ``(g_i, u_i)`` is what the activation consumes.
    Both of those are inherited from :class:`FeedForward`; this class replaces
    only the two projections' execution.

    Args:
        w13: the fused gate-and-up projection, ``dim -> 2 * hidden_dim``.
        w2: the down projection, ``hidden_dim -> dim``.
        activation_fn: the gated activation; defaults to SwiGLU.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(x)

        gate_up_TF = AllGatherLinear.apply(
            x,
            self.w13.weight,
            self.w13.bias,
            tp_group,
            tp_group.group_name,
        )
        out_TD = LinearReduceScatter.apply(
            self.activation_fn(*self._split_gate_up(gate_up_TF)),
            self.w2.weight,
            self.w2.bias,
            tp_group,
            tp_group.group_name,
        )
        return out_TD
