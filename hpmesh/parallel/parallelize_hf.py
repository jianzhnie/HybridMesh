"""One entry point that applies parallelism to a HuggingFace model.

Order matters, and it is the whole content of this file: TP / CP / EP are
declared first, ``torch.compile`` sits between, and FSDP wraps last so its hooks
sit outermost. Each ``apply_*`` is a no-op when its degree is 1, so the same call
runs from a single device up to a full hybrid mesh.

On provenance: this is the *orchestration* half of torchtitan's
``parallelize_hf_transformers``. The other half was three things hpmesh
deliberately does not do, and dropping them is a decision, not an oversight:

* **Untying ``tok_embeddings`` from ``lm_head``.** torchtitan un-ties them
  because its FSDP cannot shard a parameter shared by two FSDP groups. hpmesh
  instead detects the tie (``fsdp_wrap`` sets ``enable_weight_tying``) and shards
  the embedding, norm and head as one unit -- so untying here would silently
  train an un-tied model that no longer matches its HF checkpoint. Models with
  ``tie_word_embeddings=False`` (llama, qwen) are unaffected either way.
* **Converting modules to a ``Module`` protocol.** hpmesh has none; see
  ``hf_sharding``'s module docstring for why its declarations are currently inert.
* **Swapping in a native MoE.** hpmesh ships no MoE implementation; the probing
  half that is standalone lives in ``..models.moe_probe``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..trainer.config import HybridMeshConfig
from .cp_ep import apply_cp_ep
from .fsdp_wrap import apply_fsdp
from .pp import apply_pp
from .tp import apply_tp

logger = logging.getLogger(__name__)

__all__ = ["parallelize_hf_transformers"]


def parallelize_hf_transformers(
    model: nn.Module,
    *,
    cfg: HybridMeshConfig,
    mesh,
    parallel_dims,
) -> nn.Module:
    """Apply every parallelism dimension the config asks for, in order.

    Returns the (possibly wrapped) model; ``apply_pp`` is still a stub, so a
    ``pp > 1`` config raises there rather than producing a schedule.
    """
    model = apply_tp(model, mesh, cfg)
    model = apply_cp_ep(model, mesh, cfg)
    apply_pp(model, mesh, cfg)  # returns the schedule once PP is implemented

    if cfg.compile:
        model = torch.compile(model)

    return apply_fsdp(model, mesh, cfg, parallel_dims)
