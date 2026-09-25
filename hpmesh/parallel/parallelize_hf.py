"""One entry point that applies parallelism to a HuggingFace model.

Order matters, and it is the whole content of this file: TP / CP / EP are
declared first, activation checkpointing wraps each decoder layer next, then
``torch.compile``, and FSDP wraps last so its hooks sit outermost. The order
is not a comment -- it is the ``STAGES`` table in ``stages.py``, which both assembly
paths are driven by (the PP per-part path runs the ``on_pp`` subsequence).
Each ``apply_*`` is a no-op when its degree is 1 (or its mode off), so the
same call runs from a single device up to a full hybrid mesh.

PP is the exception to "one model in, one model out": with ``pp > 1`` the model
is cut into per-stage chunks first (``pipeline_parallel.apply_pp``), each chunk
then goes through TP / compile / FSDP here in the same relative order as the
unsplit path, and the caller gets back a ``PipelineParallelSetup`` (stages,
chunks, schedule) instead of a model.

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

from hpmesh.config import (
    CompileConfig,
    MemoryBudgetACConfig,
    ParallelConfig,
    SelectiveACConfig,
)

from ..utils.logger_utils import get_logger
from . import matrix
from .activation_checkpoint import apply_ac
from .compile import apply_compile
from .context_parallel import apply_cp
from .expert_parallel import apply_ep
from .fully_shard import apply_fsdp
from .pipeline_parallel import PipelineParallelSetup, apply_pp, build_pipeline_schedule
from .stages import PP_STAGE_ORDER, STAGE_ORDER, _stage_enabled
from .tensor_parallel import apply_tp

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
    compile_config: CompileConfig | None = None,
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
    ``compile_config`` tunes the compile step (per-block, backend, async TP);
    ``None`` is the plain whole-model compile.
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
            matrix.pp_activation_checkpoint()
        if global_batch_size is None:
            raise ValueError(
                "pp > 1 needs global_batch_size for microbatch validation; "
                "the trainer passes cfg.training.global_batch_size."
            )
        # apply_pp is PP-only: stage count, split, and the per-stage views.
        # The per-chunk application of the other dimensions lives here, so
        # this file is the single owner of the assembly order on both paths:
        # each stage's chunk goes through tp/(compile)/fsdp in the same
        # relative order as the unsplit path below. The dense (dp, cp, tp)
        # view excludes the pp axis, so it covers exactly this stage's
        # coordinates -- the mesh the per-part apply_* functions would have
        # been handed had the model never been split.
        stages, model_parts, has_first_stage, has_last_stage = apply_pp(
            model,
            parallel_dims=parallel_dims,
            cfg=cfg,
            device=device if device is not None else next(model.parameters()).device,
            global_batch_size=global_batch_size,
            dataset=dataset,
        )
        dense_mesh = parallel_dims.spmd_dense_mesh()
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        pp_runners = {
            "tp": lambda m: apply_tp(m, dense_mesh, cfg),
            "compile": lambda m: apply_compile(
                m, compile_config=compile_config, tp_mesh=tp_mesh
            ),
            "fsdp": lambda m: apply_fsdp(m, cfg, parallel_dims),
        }
        for i, part in enumerate(model_parts):
            for name in PP_STAGE_ORDER:
                if _stage_enabled(name, compile=compile):
                    part = pp_runners[name](part)
            model_parts[i] = part
            # Rebind the stage's submodule in case a transform replaced the chunk.
            stages[i].submod = part
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

    runners = {
        "tp": lambda m: apply_tp(m, mesh, cfg),
        "ep": lambda m: apply_ep(m, cfg, ep_group=ep_group),
        "cp": lambda m: apply_cp(m, mesh, cfg),
        "ac": lambda m: apply_ac(
            m,
            activation_checkpoint,
            selective=selective_ac,
            memory_budget=memory_budget_ac,
            compile_enabled=compile,
        ),
        # ``parallel/compile.py``: whole-model compile by default (the
        # historical behavior), per-block compile and the three compile-side
        # toggles (async TP, regional_inductor, capture_scalar_outputs)
        # behind ``compile_config``'s switches.
        "compile": lambda m: apply_compile(
            m,
            compile_config=compile_config,
            tp_mesh=(
                None
                if parallel_dims is None
                else parallel_dims.get_optional_mesh("tp")
            ),
        ),
        "fsdp": lambda m: apply_fsdp(m, cfg, parallel_dims),
    }
    for name in STAGE_ORDER:
        if _stage_enabled(name, compile=compile):
            model = runners[name](model)
    return model
