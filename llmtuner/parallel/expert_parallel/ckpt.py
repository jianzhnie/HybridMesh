"""EP-aware expert-state serialization for checkpoints.

The EP swap leaves expert weights as rank-heterogeneous tensors: each rank's
``GroupedExperts`` holds only its own ``num_local = E / ep`` experts, under the
same FQN every rank reports. A checkpoint backend that treats same-FQN tensors
as replicated collapses all ranks onto one expert slice, so EP runs refused
checkpointing outright (the retired ``ep_checkpoint`` matrix row).

The representation chosen here is the full-tensor one: on save, each expert
entry is all-gathered along the expert dim across the EP group into the
complete ``(E, F, D)`` / ``(E, D, F)`` tensor (upstream's logical view, where
the expert weight is one DTensor sharded on the EP axis); on load, every rank
slices its own ``[ep_rank * num_local, (ep_rank + 1) * num_local)`` piece back
out -- the same per-rank slicing the swap itself performs. Because the stored
tensor is the complete expert weight under the original FQN, a checkpoint is
independent of the EP degree: resuming with a different ``ep`` re-slices the
same full tensor.

Two value forms are handled:

* plain tensors (EP without FSDP): the live value is the local slice; gather
  concatenates, load narrows;
* ``DTensor`` (EP with FSDP2, experts sharded on the efsdp axis): the value is
  first materialized with ``full_tensor()`` per EP rank, then gathered; on
  load the slice is re-distributed with the live parameter's own mesh and
  placements so ``load_state_dict`` can copy it into the sharded parameter.

The transform applies to any flat, FQN-keyed dict: the model wrapper's (key
``fqn``), the optimizer container's (keys ``state.{fqn}.{state_name}`` -- the
shape match against the live parameter keeps scalar entries like ``step``
untouched), and the EMA's (``state.{fqn}.ema_params``).

Known cost: every save materializes the full expert weights on every rank
(and every load plans against them). That is the price of a degree-
independent, replication-safe layout; a sharded write plan is future work.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from ...accelerator import dist_utils

try:  # torch >= 2.4; the FSDP path (DTensor expert weights) needs it.
    from torch.distributed.tensor import DTensor, distribute_tensor
except ImportError:  # pragma: no cover - old torch, plain-tensor path only
    DTensor = None
    distribute_tensor = None

__all__ = [
    "EP_GROUP_ATTR",
    "expert_shard_map",
    "gather_expert_state",
    "load_expert_state",
    "mark_experts_ep_sharded",
]

EP_GROUP_ATTR = "_ep_checkpoint_group"
"""Attribute the EP swap sets on a ``GroupedExperts`` whose weights are
expert-dim-sharded across that process group. Absent attribute == ep == 1 ==
nothing to transform."""


def mark_experts_ep_sharded(grouped: nn.Module, group: dist.ProcessGroup) -> None:
    """Flag ``grouped``'s direct parameters as expert-dim-sharded over ``group``."""
    setattr(grouped, EP_GROUP_ATTR, group)


def expert_shard_map(
    model_parts: list[nn.Module],
) -> dict[str, dist.ProcessGroup]:
    """Map each EP-sharded expert parameter FQN to its EP process group."""
    shard_map: dict[str, dist.ProcessGroup] = {}
    for model in model_parts:
        for module_path, module in model.named_modules():
            group = getattr(module, EP_GROUP_ATTR, None)
            if group is None:
                continue
            for name, _ in module.named_parameters(recurse=False):
                fqn = f"{module_path}.{name}" if module_path else name
                shard_map[fqn] = group
    return shard_map


def _named_params(model_parts: list[nn.Module]) -> dict[str, torch.Tensor]:
    return {
        fqn: param
        for model in model_parts
        for fqn, param in model.named_parameters()
    }


def _matched_entries(
    sd: dict[str, Any],
    shard_map: dict[str, dist.ProcessGroup],
    params: dict[str, torch.Tensor],
):
    """Yield ``(key, group, live_param)`` for entries holding an expert slice.

    An entry matches when its key is the expert FQN itself (model dicts) or a
    nested state entry under it (``state.{fqn}.{name}``, optimizer/EMA dicts)
    AND its tensor has the live parameter's shape -- which is what keeps
    replicated scalars (``state.{fqn}.step``) and unrelated state out of the
    transform.
    """
    for key, value in sd.items():
        for fqn, group in shard_map.items():
            if key != fqn and not key.startswith(f"state.{fqn}."):
                continue
            param = params.get(fqn)
            if param is None or not isinstance(value, torch.Tensor):
                continue
            if value.shape != param.shape:
                continue
            yield key, group, param
            break


def gather_expert_state(
    sd: dict[str, Any],
    shard_map: dict[str, dist.ProcessGroup],
    model_parts: list[nn.Module],
) -> dict[str, Any]:
    """Return ``sd`` with every expert entry all-gathered to the full tensor.

    The result presents one replicated ``(E, ...)`` tensor per expert weight,
    which every checkpoint backend can save (and plan a load against) with
    its ordinary replicated-tensor semantics.
    """
    if not shard_map:
        return sd
    params = _named_params(model_parts)
    out = dict(sd)
    for key, group, _ in _matched_entries(sd, shard_map, params):
        value = out[key]
        if isinstance(value, DTensor):
            # Per-EP-rank materialization of the efsdp-sharded weight.
            value = value.full_tensor()
        world = dist_utils.get_world_size(group)
        gathered = value.new_empty((world * value.shape[0],) + value.shape[1:])
        dist.all_gather_into_tensor(gathered, value.contiguous(), group=group)
        out[key] = gathered
    return out


def load_expert_state(
    sd: dict[str, Any],
    shard_map: dict[str, dist.ProcessGroup],
    model_parts: list[nn.Module],
) -> dict[str, Any]:
    """Return ``sd`` with every full expert entry sliced back to this rank.

    The inverse of ``gather_expert_state``: after the backend has filled the
    full tensors, each rank keeps its own ``num_local`` experts -- the same
    slice the EP swap copied out of the HF block. A live ``DTensor`` parameter
    receives a re-distributed tensor with its own mesh and placements so the
    downstream ``load_state_dict`` copies shard-for-shard.
    """
    if not shard_map:
        return sd
    params = _named_params(model_parts)
    out = dict(sd)
    for key, group, param in _matched_entries(sd, shard_map, params):
        full = out[key]
        world = dist_utils.get_world_size(group)
        rank = dist_utils.get_rank(group)
        num_local = param.shape[0]
        if full.shape[0] != num_local * world:
            raise ValueError(
                f"checkpoint entry {key!r} holds {full.shape[0]} experts, but "
                f"this rank expects {num_local} x ep={world} = "
                f"{num_local * world}. The checkpoint was saved with a "
                "different expert layout."
            )
        local = full.narrow(0, rank * num_local, num_local)
        if isinstance(param, DTensor):
            local = distribute_tensor(
                local.contiguous(), param.device_mesh, param.placements
            )
        out[key] = local
    return out
