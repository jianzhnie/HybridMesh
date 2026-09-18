"""Step 2: tensor parallelism (declarative, via spmd_types).

Core idea: instead of hand-writing Column/RowParallelLinear, DECLARE how each
weight/activation is sharded across the TP axis; a runtime inserts the matching
collectives around forward. TODO(learning): express placements with spmd_types
and apply them to the attention/MLP linears.
"""

from __future__ import annotations

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from ..trainer.config import HybridMeshConfig


def apply_tp(model: nn.Module, mesh: DeviceMesh | None, cfg: HybridMeshConfig) -> nn.Module:
    if cfg.tp == 1:
        return model
    raise NotImplementedError(
        "TP is step 2 of the learning path: implement declarative sharding here."
    )
