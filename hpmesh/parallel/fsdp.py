"""Step 1: data parallelism via FSDP2 `fully_shard`.

Core idea: each DP rank holds a SHARD of every parameter. Before a layer's
forward, FSDP all-gathers the full param; after backward, it reduce-scatters the
gradient. Memory per rank ~ params / dp.

Learning exercise: read torch.distributed.fsdp.fully_shard, then shard the
decoder layers one by one (so the all-gather of layer i+1 overlaps compute of i).
"""

from __future__ import annotations

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from ..trainer.config import HybridMeshConfig
from ._utils import decoder_layers


def apply_fsdp(model: nn.Module, mesh: DeviceMesh | None, cfg: HybridMeshConfig) -> nn.Module:
    if mesh is None or (cfg.dp == 1 and mesh["dp"].size() == 1):
        return model
    from torch.distributed.fsdp import fully_shard

    dp_mesh = mesh["dp"]
    # Shard the inner transformer blocks first (finer-grained overlap), then the root.
    for layer in decoder_layers(model):
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)
    return model
