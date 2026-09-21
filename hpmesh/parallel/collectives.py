"""Mesh-aware reductions, and gradient-norm clipping that respects them.

Vendored from torchtitan ``distributed/utils.py``. What changed:

* The ``extra_pg`` argument is gone. torchtitan threads an extra process group
  through every reduction to reach ranks a mesh does not model (its odd-sized TP
  cases); hpmesh's meshes cover every rank, so the mesh argument is the whole
  addressing story.
* ``_clip_grad_norm_with_ep`` (the EP-aware norm path) is not ported: it asserts
  every parameter is a DTensor on a sparse mesh with an ``"ep"`` axis, which
  hpmesh's EP parameters are not (``apply_ep`` physically partitions experts
  across the ep ranks instead). Because the dense norm would then miss the
  cross-EP sum of expert-gradient norms, the Trainer loud-raises for
  ``ep > 1`` with clipping enabled rather than clip to a wrong norm. Port the
  EP-aware reduction before lifting that rejection.
* ``dist_mean``, ``all_gather_entries`` and friends are not ported: they exist
  upstream for bucketed per-module metrics, none of which hpmesh reports. Port
  them when there is a caller, not before.

The single non-obvious line kept from upstream is the ``DTensor`` branch in
``clip_grad_norm_``; it carries a comment explaining why it exists.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)

__all__ = [
    "clip_grad_norm_",
    "dist_max",
    "dist_sum",
    "dist_sum_tensor",
    "set_pg_timeouts",
]


def set_pg_timeouts(
    timeout: timedelta,
    parallel_dims,
    *,
    device: torch.device | None = None,
) -> None:
    """Lower every process group's timeout, once startup is behind the run.

    Called after the first completed train step. The groups are created with a
    long timeout because startup -- model build, the first collective, compile --
    is what genuinely takes minutes; left there, a later hang looks the same as a
    slow start and the job waits out the startup value. By this point that work
    is done, so the timeout can become the one a stall should be measured
    against.

    ``device`` is the rank's device, used only for the barrier's ``device_ids``
    and the device-side sync (NCCL needs both; gloo rejects ``device_ids``, so
    ``None`` -- the default -- omits them).

    The barrier before the change is the point of the whole function: a slow rank
    may still be inside an operation permitted by the OLD timeout while a fast
    rank moves on and issues a collective under the new, shorter one, and
    times out waiting for it. Synchronizing first means every rank crosses the
    reduction together.
    """
    if device is not None and device.type == "cuda":
        dist.barrier(device_ids=[device.index])
        torch.cuda.synchronize(device)
    else:
        dist.barrier()

    # ``None`` names the default (world) group, which is not part of any mesh.
    groups = [
        mesh.get_group()
        for mesh in parallel_dims.get_all_one_dimensional_meshes().values()
    ]
    logger.info(
        "Adjusting the timeout of %d process group(s) plus the default to %s",
        len(groups),
        timeout,
    )
    for group in groups:
        dist.set_timeout(timeout, group)
    dist.set_timeout(timeout)


def _reduce(x: torch.Tensor, *, reduce_op: dist.ReduceOp, mesh) -> torch.Tensor:
    """All-reduce ``x`` over ``mesh``, or return it untouched when there is none.

    ``mesh is None`` is the single-rank case: the reduction is the identity, so
    skipping the collective is the correct answer rather than a shortcut. It is
    what lets one training loop run from one device up to a full mesh.

    The clone is what makes this a *function* rather than a mutation: upstream
    reaches the same place via ``funcol.all_reduce``, which is out-of-place by
    construction. It matters because callers keep using the tensor they passed:
    ``train_step`` reduces the local token count here and then divides the loss
    by that same tensor, expecting the *local* count. Under ``dist.all_reduce``,
    whose own docstring says the input is mutated in place, it instead holds the
    global count, and the per-rank average silently becomes a global one.
    """
    if mesh is None:
        return x
    result = x.clone()
    dist.all_reduce(result, op=reduce_op, group=mesh.get_group())
    return result


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
