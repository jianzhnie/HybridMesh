"""One entry point that applies parallelism to a HuggingFace model.

Order matters, and it is the whole content of this file: TP / CP / EP are
declared first, activation checkpointing wraps each decoder layer next, then
``torch.compile``, and FSDP wraps last so its hooks sit outermost. Each
``apply_*`` is a no-op when its degree is 1 (or its mode off), so the same
call runs from a single device up to a full hybrid mesh.

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
* **Swapping in a native MoE.** No longer true: ``apply_ep`` swaps HF MoE
  blocks for the ``models/common`` MoE stack when ``ep > 1`` (see
  ``parallel/expert_parallel/swap.py``). The swap moves weights rather than
  re-initializing them, so the model still trains from HF's initialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hpmesh.trainer.config import (
    MemoryBudgetACConfig,
    ParallelConfig,
    SelectiveACConfig,
)

from ..utils.logger_utils import get_logger
from .activation_checkpoint import apply_ac
from .context_parallel import apply_cp
from .expert_parallel import apply_ep
from .fully_shard.apply import apply_fsdp
from .pipeline_parallel import PipelineParallelSetup, apply_pp, build_pipeline_schedule
from .tensor_parallel.tp import apply_tp

logger = get_logger(__name__)

__all__ = ["PipelineParallelSetup", "parallelize_hf_transformers"]


def parallelize_hf_transformers(
    model: nn.Module,
    *,
    cfg: ParallelConfig,
    mesh,
    parallel_dims,
    device: torch.device | None = None,
    compile: bool = False,
    activation_checkpoint: str = "none",
    selective_ac: SelectiveACConfig | None = None,
    memory_budget_ac: MemoryBudgetACConfig | None = None,
    global_batch_size: int | None = None,
    dataset: str = "random",
) -> nn.Module | PipelineParallelSetup:
    """Apply every parallelism dimension the config asks for, in order.

    ``compile``, ``activation_checkpoint``, ``selective_ac``,
    ``global_batch_size`` and ``dataset`` are training-side values, passed
    explicitly rather than read off a run-wide config: this layer's contract is
    ``ParallelConfig`` plus the handful of scalars the guards actually need.
    ``global_batch_size`` is required only on the ``pp > 1`` path (microbatch
    validation); ``dataset`` gates the same path's corpus restriction;
    ``selective_ac`` / ``memory_budget_ac`` are read only when
    ``activation_checkpoint`` names their mode (``'selective'`` /
    ``'memory_budget'``; the latter also requires ``compile=True``).

    Returns the (possibly wrapped) model -- or, with ``pp > 1``, a
    ``PipelineParallelSetup``: pipeline parallelism cuts the model into
    per-stage chunks, so there is no single module left to return. The two
    return shapes are how the caller learns which case it is in.
    """
    if parallel_dims is not None and parallel_dims.pp_enabled:
        if activation_checkpoint != "none":
            raise NotImplementedError(
                "activation checkpointing is not wired through the pp > 1 path: "
                "it belongs between apply_tp and compile inside apply_pp's "
                "per-chunk pipeline, which does not accept it yet."
            )
        if global_batch_size is None:
            raise ValueError(
                "pp > 1 needs global_batch_size for microbatch validation; "
                "the trainer passes cfg.training.global_batch_size."
            )
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
            global_batch_size=global_batch_size,
            dataset=dataset,
            compile=compile,
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
    model = apply_ep(model, cfg, ep_group=ep_group)
    model = apply_cp(model, mesh, cfg)
    # AC after the sharding wrappers (it must enclose the TP/CP-modified
    # layer), before compile and FSDP -- torchtitan's order in
    # ``parallelize_llama``.
    model = apply_ac(
        model,
        activation_checkpoint,
        selective=selective_ac,
        memory_budget=memory_budget_ac,
        compile_enabled=compile,
    )

    if compile:
        # Whole-model compile -- the deliberate opposite of torchtitan's
        # ``apply_compile``, which compiles each TransformerBlock so the
        # repeated structure is traced once and each block's graph is reused.
        # torchtitan's version also does three other things hpmesh has no
        # counterpart for: async TP (``inductor._micro_pipeline_tp``),
        # ``regional_inductor`` for inductor-only regions under a non-inductor
        # backend (its FlexInnerAttention needs one), and
        # ``capture_scalar_outputs`` for token-choice MoE dispatch's
        # data-dependent shapes. hpmesh relies on HF's flex implementation
        # instead of its own, and its MoE runs eager, so none of the three is
        # reachable here -- but a model whose MoE dispatch needs dynamic shapes
        # under compile would fail on this line rather than being handled.
        # See docs/hpmesh_upstream_map.md (D: ``distributed/compile.py``).
        model = torch.compile(model)

    return apply_fsdp(model, cfg, parallel_dims)
