"""Mesh-aware reductions, and gradient-norm clipping that respects them.

Vendored from torchtitan ``distributed/utils.py``. What changed:

* The ``extra_pg`` argument is gone. torchtitan threads an extra process group
  through every reduction to reach ranks a mesh does not model (its odd-sized TP
  cases); hpmesh's meshes cover every rank, so the mesh argument is the whole
  addressing story.
* EP-aware clipping is adapted to hpmesh's physical expert partition: callers
  pass the exact local expert parameters plus the EP mesh. Dense parameters are
  counted once, while the p-th powers of local expert norms are reduced across
  EP ranks. This avoids upstream's requirement that every parameter be a
  DTensor carrying an explicit ``"ep"`` mesh axis.
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

from .device import device_module

logger = logging.getLogger(__name__)

__all__ = [
    "clip_grad_norm_",
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
    if device is not None and device.type != "cpu":
        # NCCL accepts device_ids; HCCL selects the current NPU and rejects the
        # CUDA-specific argument on some torch-npu releases.
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()
        # Derive the sync call from the passed device rather than the
        # import-time global.
        getattr(torch, device.type, device_module).synchronize(device)
    else:
        dist.barrier()

    # torch-npu 2.10 exposes the c10d compatibility API, but HCCL does not
    # implement the operation (it emits one warning per group and changes
    # nothing). The synchronization above is still useful; retain the startup
    # timeout and make this limitation explicit once per rank.
    if device is not None and device.type == "npu":
        logger.warning(
            "HCCL cannot change process-group timeouts at runtime; continuing "
            "with the startup timeout"
        )
        return

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
    set_timeout = getattr(dist, "set_timeout", None)
    if set_timeout is None:
        # PyTorch 2.10 exposes this operation only from distributed_c10d.
        # Keep the compatibility detail here rather than forcing the trainer to
        # know which torch release it is running on.
        set_timeout = getattr(dist.distributed_c10d, "_set_pg_timeout", None)
    if set_timeout is None:
        logger.warning(
            "This PyTorch build cannot change process-group timeouts at runtime; "
            "continuing with the startup timeout"
        )
        return

    for group in groups:
        set_timeout(timeout, group)
    set_timeout(timeout)


def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh=None,
    ep_mesh=None,
    expert_parameters: Iterable[torch.Tensor] | None = None,
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
        ep_mesh: expert-parallel mesh. When present, only the norm contribution
            from ``expert_parameters`` is reduced over this mesh.
        expert_parameters: parameters physically partitioned across EP ranks.
            Required exactly when ``ep_mesh`` is provided.

    Returns:
        The total norm of the parameter gradients (viewed as one vector).

    NOTE: intentionally not ``torch.no_grad()`` -- ``get_total_norm`` must keep
    its autograd history so the clip's effect propagates. Do not add it.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)  # do not exhaust a generator

    if (ep_mesh is None) != (expert_parameters is None):
        raise ValueError(
            "ep_mesh and expert_parameters must either both be provided or both be None"
        )

    if ep_mesh is None:
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type, error_if_nonfinite, foreach
        )
    else:
        expert_ids = {id(p) for p in expert_parameters}
        expert_grads = [
            p.grad for p in parameters if id(p) in expert_ids and p.grad is not None
        ]
        dense_grads = [
            p.grad for p in parameters if id(p) not in expert_ids and p.grad is not None
        ]
        expert_norm = torch.nn.utils.get_total_norm(
            expert_grads, norm_type, error_if_nonfinite, foreach
        )
        dense_norm = torch.nn.utils.get_total_norm(
            dense_grads, norm_type, error_if_nonfinite, foreach
        )
        if isinstance(expert_norm, DTensor):
            expert_norm = expert_norm.full_tensor()
        if isinstance(dense_norm, DTensor):
            dense_norm = dense_norm.full_tensor()
        if expert_norm.device != dense_norm.device:
            # An empty expert_grads list yields a CPU zero from
            # get_total_norm; the reduce below needs it on the comm device.
            expert_norm = expert_norm.to(dense_norm.device)

        if math.isinf(norm_type):
            dist.all_reduce(
                expert_norm, op=dist.ReduceOp.MAX, group=ep_mesh.get_group()
            )
            total_norm = torch.maximum(dense_norm, expert_norm)
        else:
            expert_norm = expert_norm.pow(norm_type)
            dist.all_reduce(
                expert_norm, op=dist.ReduceOp.SUM, group=ep_mesh.get_group()
            )
            total_norm = (dense_norm.pow(norm_type) + expert_norm).pow(
                1.0 / norm_type
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
