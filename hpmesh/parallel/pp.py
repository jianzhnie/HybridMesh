"""Step 3: pipeline parallelism (micro-batches + point-to-point).

Core idea: split the layers across PP stages; each step runs several micro-batches
through a schedule (start with 1F1B) so stages stay busy.

The stage-splitting half is done -- ``pipeline.py`` decides the layer assignment
(``generate_llm_fqn_per_model_part``) and builds this rank's stages
(``split_model_into_stages``). What remains here is the schedule: build it from
the stages and drive it from the training loop, which also has to pass
micro-batches and only send ``input_ids``/``labels`` to the stages that want them.
"""

from __future__ import annotations

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from ..trainer.config import HybridMeshConfig


def apply_pp(model: nn.Module, mesh: DeviceMesh | None, cfg: HybridMeshConfig):
    if cfg.pp == 1:
        return None
    raise NotImplementedError(
        "PP is step 3 of the learning path: build stages and a 1F1B schedule here."
    )
