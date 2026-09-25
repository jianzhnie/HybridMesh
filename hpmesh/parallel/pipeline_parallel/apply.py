"""Pipeline parallelism: split the model, then build the schedule over it.

``pipeline.py`` (same package) computes the stage split; this module is the
driver around it, vendored in shape from torchtitan's
``distributed/pipeline_parallel.pipeline_llm``:

* ``apply_pp`` -- decides the stage count from the schedule class and splits
  the model into this rank's stages. Parallelizing the resulting chunks (TP,
  compile, FSDP) is the caller's job: ``parallelize_hf`` owns that assembly
  order for both the split and the unsplit path, so this module never imports
  a sibling parallelism family. Its ``first_stage_module_fqns`` option is
  torchtitan's ``pipeline_with_first_stage_modules``: extra top-level modules
  co-located with stage 0 (see ``_prepend_first_stage_modules``).
* ``build_pipeline_schedule`` -- instantiates the torch pipelining schedule
  over this rank's stages, with the summed next-token CE the trainer
  normalizes, wrapped down to the bare scalar a schedule requires.

What is not ported: the ``get_mesh`` callback (hpmesh passes plain tensors
across stage boundaries, never DTensors -- see ``pipeline.py``) and csv-loaded
runtime schedules, which are rejected rather than honored.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    _PipelineSchedule,
    get_schedule_class,
)

from hpmesh.config import ParallelConfig

from ...components.loss import cross_entropy_loss
from ...utils.logger_utils import get_logger
from .. import matrix
from ..parallel_dims import ParallelDims
from .pipeline import generate_llm_fqn_per_model_part, split_model_into_stages

logger = get_logger(__name__)

__all__ = ["PipelineParallelSetup", "apply_pp", "build_pipeline_schedule"]


@dataclass
class PipelineParallelSetup:
    """Everything the trainer needs to drive a ``pp > 1`` run.

    Returned by ``parallelize_hf_transformers`` in place of the parallelized
    model: under pipeline parallelism there is no single model to hand back --
    this rank holds only its stages' chunks, and a schedule drives them.
    """

    schedule: _PipelineSchedule
    stages: list[PipelineStage]
    model_parts: list[nn.Module]
    has_first_stage: bool
    has_last_stage: bool


def _scalar_loss_fn(
    pred: torch.Tensor, labels: torch.Tensor, **loss_kwargs
) -> torch.Tensor:
    """Summed next-token CE over one microbatch, as a bare scalar.

    Same arithmetic as ``Trainer._loss_sum``: ``labels`` arrives already
    aligned (``pred[t]`` predicts ``labels[t]``, ``IGNORE_INDEX`` at document
    boundaries), so no shift happens here. Schedules call
    ``loss_fn(output, target)`` and backward the result directly, so the
    trainer's pair-returning version cannot be handed over -- this is the same
    computation in the shape the schedule needs.

    ``global_valid_tokens`` arrives through the schedule's ``loss_kwargs`` (it
    has to: the denominator spans every rank and every microbatch, so no single
    call could compute it). Dividing here is what makes the schedule's own
    backward produce summed/G -- the step's normalized gradient -- with one
    division per microbatch instead of a rescale applied afterwards.

    The caller divides it back out for reporting.
    """
    return cross_entropy_loss(pred, labels) / loss_kwargs["global_valid_tokens"]


def _get_pipeline_metadata(
    parallel_dims: ParallelDims,
    parallelism: ParallelConfig,
    num_layers: int,
) -> tuple[int, int, int]:
    """Decide the stage count, from the schedule class and the config.

    Returns ``(num_stages, input_weight, output_weight)``. Vendored from
    torchtitan's ``_get_pipeline_metadata``: single-stage schedules (GPipe,
    1F1B) default to one stage per rank, looped ones (Interleaved1F1B, ...)
    to two; ``pipeline_parallel_layers_per_stage`` overrides the default and
    is validated against the schedule kind.
    """
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers

    if layers_per_stage is None:
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_stages = parallel_dims.pp * stages_per_rank
        return num_stages, input_weight, output_weight

    num_stages = math.ceil(
        (num_layers + input_weight + output_weight) / layers_per_stage
    )
    model_info = (
        f"model has {num_layers} layers with "
        f"pipeline_parallel_layers_per_stage={layers_per_stage}"
    )
    if num_stages % parallel_dims.pp != 0:
        raise ValueError(
            f"Number of pipeline stages ({num_stages}) must be divisible by "
            f"the pipeline parallel degree ({parallel_dims.pp}); {model_info}."
        )
    stages_per_rank = num_stages // parallel_dims.pp
    if is_single_stage_schedule and stages_per_rank != 1:
        raise ValueError(
            f"Schedule {parallelism.pipeline_parallel_schedule!r} runs exactly "
            f"1 stage per rank, but the split yields {stages_per_rank}; "
            f"{model_info}."
        )
    if not is_single_stage_schedule and stages_per_rank < 2:
        raise ValueError(
            f"Schedule {parallelism.pipeline_parallel_schedule!r} needs at "
            f"least 2 stages per rank, but the split yields {stages_per_rank}; "
            f"{model_info}."
        )
    return num_stages, input_weight, output_weight


def _prepend_first_stage_modules(
    module_names_per_stage: list[list[str]],
    model: nn.Module,
    first_stage_module_fqns: Sequence[str],
) -> None:
    """Co-locate extra top-level modules with pipeline stage 0, in place.

    The generated LLM split only knows the decoder parts (``tok_embeddings``,
    ``layers.*``, ``norm``, ``lm_head``, ``rotary_emb``). This prepends each
    present module from ``first_stage_module_fqns`` to stage 0's FQN list, in
    the given order; ``split_model_into_stages`` then keeps the real module on
    stage 0 and blanks it to ``nn.Identity`` everywhere else, so the model's
    ``forward`` must tolerate the blanked version.

    Invariants:

    * FQN stability: the modules stay top-level children under their original
      names, so every stage's state-dict keys for them match the unsplit
      model's -- optimizer and checkpoint keys do not move.
    * No cross-stage ownership: an FQN the split already assigned (e.g. a
      decoder part, or a duplicate in ``first_stage_module_fqns``) would put a
      live copy on two stages and collide their keys in one checkpoint, so it
      is rejected rather than merged.
    * Absent modules (``getattr(model, fqn, None) is None``) are skipped, so a
      caller may list modules that only some model variants carry.
    """
    owned = {name for stage in module_names_per_stage for name in stage}
    seen: set[str] = set()
    present = []
    for fqn in first_stage_module_fqns:
        if getattr(model, fqn, None) is None:
            continue
        if fqn in owned:
            raise ValueError(
                f"first-stage module {fqn!r} is already assigned to a pipeline "
                "stage by the split; co-locating it with stage 0 would give "
                "two stages a live copy of the same parameter."
            )
        if fqn in seen:
            raise ValueError(
                f"first-stage module {fqn!r} is listed more than once."
            )
        seen.add(fqn)
        present.append(fqn)
    module_names_per_stage[0][:0] = present


def _validate_microbatches(
    parallel_dims: ParallelDims, cfg: ParallelConfig, global_batch_size: int
) -> None:
    """Fail at setup, not mid-step, on a batch that cannot be microbatched.

    The trainer chunks each rank's rows into ``num_pp_microbatches`` pieces, so
    the per-rank row count must divide evenly; the schedule would otherwise be
    fed uneven (or wrong-count) microbatches and hang in a p2p.
    """
    dp = parallel_dims.dp_replicate * parallel_dims.dp_shard
    if global_batch_size % dp != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) is not divisible by "
            f"the data-parallel degree ({dp})"
        )
    rows_per_rank = global_batch_size // dp
    num_microbatches = cfg.num_pp_microbatches
    # torchtitan validates this in its config's __post_init__ (trainer.py);
    # hpmesh's config does not, so the check lives here -- a 0 would otherwise
    # surface as a ZeroDivisionError on the modulo below.
    if num_microbatches <= 0:
        raise ValueError(
            f"num_pp_microbatches must be greater than 0, got {num_microbatches}."
        )
    if rows_per_rank % num_microbatches != 0:
        raise ValueError(
            f"per-rank batch rows ({global_batch_size} / {dp} = "
            f"{rows_per_rank}) must be divisible by num_pp_microbatches "
            f"({num_microbatches})"
        )


def apply_pp(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    cfg: ParallelConfig,
    device: torch.device,
    global_batch_size: int,
    dataset: str = "random",
    first_stage_module_fqns: Sequence[str] | None = None,
) -> tuple[list[PipelineStage], list[nn.Module], bool, bool]:
    """Split ``model`` into this rank's pipeline stages.

    The chunks come back unparallelized: running each through TP, (optionally)
    ``torch.compile`` and FSDP -- and rebinding ``stage.submod`` afterwards, so
    a wrapping transform cannot leave the stage running the pre-wrap module --
    is ``parallelize_hf_transformers``'s job, which owns that assembly order
    for the split and unsplit paths alike.

    ``global_batch_size`` and ``dataset`` are training-side values the PP
    guards need; they are explicit parameters rather than reads off a
    run-wide config so this layer never sees ``HybridMeshConfig``.

    ``first_stage_module_fqns`` names extra top-level modules (e.g. a
    multimodal encoder) to co-locate with stage 0; see
    ``_prepend_first_stage_modules`` for the invariants. It only applies to
    the auto-generated split -- an explicit ``module_fqns_per_model_part``
    already places modules by hand, so the two are not merged (a warning is
    logged and the explicit split wins). The default ``None`` leaves the
    split, and every stage's state-dict keys, bitwise unchanged. Note the
    auto split does not model the added load (``input_weight`` only accounts
    for ``tok_embeddings``); rebalance with
    ``pipeline_parallel_first_stage_less_layers``.

    Returns ``(stages, model_parts, has_first_stage, has_last_stage)``; the
    schedule over the stages is built separately (``build_pipeline_schedule``).
    """
    if parallel_dims.cp_enabled or parallel_dims.ep_enabled:
        matrix.pp_cp_ep()
    if dataset != "random":
        matrix.pp_real_corpus()
    if getattr(model, "enable_weight_tying", False):
        matrix.pp_weight_tying()
    parallelism = cfg
    pp_mesh = parallel_dims.get_mesh("pp")
    _validate_microbatches(parallel_dims, cfg, global_batch_size)

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        num_layers = len(model.layers)
        num_stages, input_weight, output_weight = _get_pipeline_metadata(
            parallel_dims, parallelism, num_layers
        )
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_stages, num_layers, input_weight, output_weight
        )
        if first_stage_module_fqns:
            _prepend_first_stage_modules(
                module_names_per_stage, model, first_stage_module_fqns
            )
    else:
        if first_stage_module_fqns:
            logger.warning(
                "first_stage_module_fqns is ignored because "
                "module_fqns_per_model_part already defines the split; "
                "place the extra modules in the explicit split instead."
            )
        # An explicit split still has to land a whole number of stages per
        # rank; the per-schedule-kind check happens in _get_pipeline_metadata
        # for the generated path, so assert the divisibility here.
        num_stages = len(module_names_per_stage)
        if num_stages % parallel_dims.pp != 0:
            raise ValueError(
                f"module_fqns_per_model_part defines {num_stages} stages, "
                f"which is not divisible by the pipeline parallel degree "
                f"({parallel_dims.pp})"
            )

    stages, model_parts = split_model_into_stages(
        model,
        pp_mesh,
        parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )

    has_first_stage = any(stage.is_first for stage in stages)
    has_last_stage = any(stage.is_last for stage in stages)
    return stages, model_parts, has_first_stage, has_last_stage


def build_pipeline_schedule(
    stages: list[PipelineStage],
    *,
    cfg: ParallelConfig,
) -> _PipelineSchedule:
    """Build the schedule that drives this rank's stages.

    Vendored from torchtitan's ``_build_pipeline_schedule``, minus the
    csv-loaded runtime schedule. ``scale_grads=False`` because the loss is a
    sum over tokens -- the trainer normalizes by the global token count after
    the reduction, so the schedule must not average over microbatches.
    """
    parallelism = cfg
    if parallelism.pipeline_parallel_schedule_csv:
        raise NotImplementedError(
            "pipeline_parallel_schedule_csv is not wired: hpmesh builds "
            "schedules by name only."
        )
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    num_microbatches = parallelism.num_pp_microbatches
    num_total_stages = parallelism.pipeline_parallel_size * len(stages)
    if num_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({num_microbatches}) is less than the "
            f"total number of stages ({num_total_stages}), which may result in "
            "a bubble in the pipeline."
        )

    if issubclass(schedule_class, PipelineScheduleMulti):
        schedule = schedule_class(
            stages,
            n_microbatches=num_microbatches,
            loss_fn=_scalar_loss_fn,
            scale_grads=False,
        )
    else:
        schedule = schedule_class(
            stages[0],
            n_microbatches=num_microbatches,
            loss_fn=_scalar_loss_fn,
            scale_grads=False,
        )
    # Torch 2.10's public ``step`` accepts a whole batch and splits tensor
    # kwargs itself; hpmesh has already built microbatches because positions
    # and labels are sequence-sharded before PP. Keep the denominator as a
    # per-step schedule attribute and use the internal pre-split driver below.
    schedule._hpmesh_global_valid_tokens = torch.ones((), dtype=torch.float32)
    schedule._loss_fn = lambda pred, labels: _scalar_loss_fn(
        pred,
        labels,
        global_valid_tokens=schedule._hpmesh_global_valid_tokens,
    )
    logger.info(
        f"Using pipeline schedule {parallelism.pipeline_parallel_schedule} "
        f"with {num_microbatches} microbatches and {num_total_stages} stages."
    )
    return schedule
