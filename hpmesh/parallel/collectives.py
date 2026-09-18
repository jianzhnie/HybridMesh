"""Mesh-aware reductions, and gradient-norm clipping that respects them.

Vendored from torchtitan ``distributed/utils.py``. What changed:

* The ``extra_pg`` argument is gone. torchtitan threads an extra process group
  through every reduction to reach ranks a mesh does not model (its odd-sized TP
  cases); hpmesh's meshes cover every rank, so the mesh argument is the whole
  addressing story.
* ``_clip_grad_norm_with_ep`` (the EP-aware norm path) is not ported: it asserts
  every parameter is a DTensor on a sparse mesh, which hpmesh's HF models are
  not. The norm here is the dense one -- correct for the FSDP / DP / TP
  configurations hpmesh can actually run today. Composing it with EP is a TODO.
* ``dist_mean``, ``all_gather_entries`` and friends are not ported: they exist
  upstream for bucketed per-module metrics, none of which hpmesh reports. Port
  them when there is a caller, not before.

The single non-obvious line kept from upstream is the ``DTensor`` branch in
``clip_grad_norm_``; it carries a comment explaining why it exists.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

__all__ = ["clip_grad_norm_", "dist_max", "dist_sum", "dist_sum_tensor"]


def _reduce(x: torch.Tensor, *, reduce_op: dist.ReduceOp, mesh) -> torch.Tensor:
    """All-reduce ``x`` over ``mesh``, or return it untouched when there is none.

    ``mesh is None`` is the single-rank case: the reduction is the identity, so
    skipping the collective is the correct answer rather than a shortcut. It is
    what lets one training loop run from one device up to a full mesh.
    """
    if mesh is None:
        return x
    dist.all_reduce(x, op=reduce_op, group=mesh.get_group())
    return x


def dist_sum_tensor(x: torch.Tensor, mesh=None) -> torch.Tensor:
    """Sum ``x`` across ``mesh``, keeping the result on its device.

    The on-device counterpart of :func:`dist_sum`: used for the token count that
    normalizes the loss, where a per-step ``.item()`` would cost a device sync.
    """
    return _reduce(x, reduce_op=dist.ReduceOp.SUM, mesh=mesh)


def dist_sum(x: torch.Tensor, mesh=None) -> float:
    """Sum ``x`` across ``mesh`` and return it as a Python float."""
    return float(dist_sum_tensor(x, mesh).item())


def dist_max(x: torch.Tensor, mesh=None) -> float:
    """Max ``x`` across ``mesh`` and return it as a Python float."""
    return float(_reduce(x, reduce_op=dist.ReduceOp.MAX, mesh=mesh).item())


def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh=None,
) -> torch.Tensor:
    """Clip the gradient norm of an iterable of parameters, over the whole model.

    ``torch.nn.utils.clip_grad_norm_`` computes the norm only along the axes its
    own sharding knows about. Under pipeline parallelism the stages hold disjoint
    parameter sets, so no single rank can see the full norm and each would clip
    against a different number. The PP norm is therefore reduced here first.

    Args:
        parameters: an iterable of tensors (or one tensor) to normalize.
        max_norm: max norm of the gradients. A non-positive value skips the clip
            but still returns the norm -- which is how the training loop reports
            ``grad_norm`` without paying for a clip nobody asked for.
        norm_type: type of the used p-norm. ``'inf'`` for infinity norm.
        error_if_nonfinite: throw if the total norm is nan/inf.
        foreach: use the faster foreach implementation (``None`` lets torch pick).
        pp_mesh: pipeline-parallel mesh; when present the norm is reduced across
            stages before clipping.

    Returns:
        The total norm of the parameter gradients (viewed as one vector).

    NOTE: intentionally not ``torch.no_grad()`` -- ``get_total_norm`` must keep
    its autograd history so the clip's effect propagates. Do not add it.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)  # do not exhaust a generator

    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )

    # Under FSDP/TP the norm comes back as a DTensor with a partial (sum)
    # placement: it must be materialized both to be correct along those axes and
    # to return a tensor whose ``.item()`` is the real global value.
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            # A norm does not survive an all-reduce directly: sum the p-th
            # powers, reduce, then take the p-th root.
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    if max_norm > 0:
        torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
