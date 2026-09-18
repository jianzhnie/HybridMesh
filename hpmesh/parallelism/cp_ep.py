"""Step 4: context parallelism OR expert parallelism (pick one to go deep).

CP core idea: shard the sequence across CP ranks; attention all-gathers K/V so
each rank attends its query shard against the full keys.
EP core idea (MoE): shard experts across ranks; an all-to-all routes each token
to its expert's rank and back.
"""

from __future__ import annotations

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from ..trainer.config import HybridMeshConfig


def apply_cp_ep(model: nn.Module, mesh: DeviceMesh | None, cfg: HybridMeshConfig) -> nn.Module:
    if cfg.cp == 1 and cfg.ep == 1:
        return model
    raise NotImplementedError(
        "CP/EP is step 4 of the learning path: implement KV all-gather (CP) or an "
        "all-to-all token dispatcher (EP) here."
    )
