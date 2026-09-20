"""One entry point that applies parallelism to a HuggingFace model.

Order matters, and it is the whole content of this file: TP / CP / EP are
declared first, ``torch.compile`` sits between, and FSDP wraps last so its hooks
sit outermost. Each ``apply_*`` is a no-op when its degree is 1, so the same call
runs from a single device up to a full hybrid mesh.

PP is the exception to "one model in, one model out": with ``pp > 1`` the model
is cut into per-stage chunks first (``pipeline_parallel.apply_pp``), each chunk
goes through TP / compile / FSDP in the same relative order, and the caller gets
back a ``PipelineParallelSetup`` (stages, chunks, schedule) instead of a model.

On provenance: this is the *orchestration* half of torchtitan's
``parallelize_hf_transformers``. The other half was three things; the first
two hpmesh deliberately does not do, and dropping them is a decision, not an
oversight:

* **Untying ``tok_embeddings`` from ``lm_head``.** torchtitan un-ties them
  because its FSDP cannot shard a parameter shared by two FSDP groups. hpmesh
  instead detects the tie (``HFTransformerModel.enable_weight_tying``) and
  shards the embedding, norm and head as one unit -- so untying here would
  silently train an un-tied model that no longer matches its HF checkpoint.
  Models with ``tie_word_embeddings=False`` (llama, qwen) are unaffected either
  way.
* **Converting modules to a ``Module`` protocol.** hpmesh has none, and no
  on-the-fly sharding-config declarations either -- the TP plan lives as plain
  data in the model registry instead (see docs/hybridmesh_design.md, SEAM 1).
* **Swapping in a native MoE.** No longer true: ``apply_cp_ep`` swaps HF MoE
  blocks for the ``models/common`` MoE stack when ``ep > 1`` (see
  ``parallel/ep.py``). The swap moves weights rather than re-initializing
  them, so the model still trains from HF's initialization.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..trainer.config import HybridMeshConfig
from .cp_ep import apply_cp_ep
from .fully_shard.fsdp_wrap import apply_fsdp
from .pipeline_parallel import PipelineParallelSetup, apply_pp, build_pipeline_schedule
from .tensor_parallel.tp import apply_tp

logger = logging.getLogger(__name__)

__all__ = ["PipelineParallelSetup", "parallelize_hf_transformers"]


def parallelize_hf_transformers(
    model: nn.Module,
    *,
    cfg: HybridMeshConfig,
    mesh,
    parallel_dims,
    device: torch.device | None = None,
) -> nn.Module | PipelineParallelSetup:
    """Apply every parallelism dimension the config asks for, in order.

    Returns the (possibly wrapped) model -- or, with ``pp > 1``, a
    ``PipelineParallelSetup``: pipeline parallelism cuts the model into
    per-stage chunks, so there is no single module left to return. The two
    return shapes are how the caller learns which case it is in.
    """
    if parallel_dims is not None and parallel_dims.pp_enabled:
        # PP owns the per-chunk application of the other dimensions: each
        # stage's chunk goes through tp/(compile)/fsdp inside apply_pp, in the
        # same relative order as below. The dense (dp, cp, tp) ``mesh`` is not
        # passed down because it does not cover the world under PP; apply_pp
        # resolves the per-stage views off parallel_dims itself.
        stages, model_parts, has_first_stage, has_last_stage = apply_pp(
            model,
            parallel_dims=parallel_dims,
            cfg=cfg,
            device=device if device is not None else next(model.parameters()).device,
        )
        return PipelineParallelSetup(
            schedule=build_pipeline_schedule(stages, cfg=cfg),
            stages=stages,
            model_parts=model_parts,
            has_first_stage=has_first_stage,
            has_last_stage=has_last_stage,
        )

    # The EP group lives on the sparse mesh, not the dense (dp, cp, tp) mesh
    # the apply_* functions are handed, so it is resolved here from
    # parallel_dims and passed down explicitly.
    ep_group = None
    if cfg.ep > 1:
        if parallel_dims is None:
            raise ValueError(
                f"ep={cfg.ep} needs a process group, but this run is "
                "single-process (parallel_dims is None). EP requires "
                "world_size > 1."
            )
        ep_mesh = parallel_dims.get_optional_mesh("ep")
        if ep_mesh is None:
            raise ValueError(
                f"ep={cfg.ep} but parallel_dims has no multi-rank 'ep' axis."
            )
        ep_group = ep_mesh.get_group()

    model = apply_tp(model, mesh, cfg)
    model = apply_cp_ep(model, mesh, cfg, ep_group=ep_group)

    if cfg.compile:
        model = torch.compile(model)

    return apply_fsdp(model, mesh, cfg, parallel_dims)
