"""Split a model into pipeline stages.

Two independent pieces, usable on their own:

* ``generate_llm_fqn_per_model_part`` -- pure arithmetic that decides which
  layers go on which stage, so no model is needed to compute the split.
* ``split_model_into_stages`` -- performs the split: deep-copies the model once
  per local stage, deletes everything that stage does not own, and wraps the
  result in a ``PipelineStage``.

Vendored from torchtitan's ``experiments/transformers_modeling_backend/pipeline.py``.
Two changes, both removals of torchtitan's module layer:

* ``ModuleList`` / ``ModuleDict`` -> the plain ``nn`` equivalents. torchtitan's
  versions exist so the containers take part in its module protocol; hpmesh has
  none, and ``nn.ModuleList`` is what every HF model already uses. The kept
  layers are re-keyed to their original indices (via ``add_module``), because a
  fresh ``ModuleList`` would renumber them and two stages' state-dict keys would
  collide in one checkpoint.
* ``get_mesh`` defaults to ``None``. torchtitan passes a callback so each stage
  can rebuild a DTensor from the plain tensor it receives from the previous stage
  (DTensors carry a ProcessGroup, which cannot cross a stage boundary). hpmesh
  runs no SPMD/DTensor path, so stages exchange plain tensors and no callback is
  needed.

The stage-splitting arithmetic is otherwise unchanged.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
    get_schedule_class,
)

from ...utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ["generate_llm_fqn_per_model_part", "split_model_into_stages"]


def generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    """Assign each module to a pipeline stage.

    The model is treated as ``input_weight`` pseudo-layers of embedding, then
    ``num_layers`` transformer layers, then ``output_weight`` pseudo-layers of
    final norm and lm_head. Those weights are how a caller says "the embedding
    costs about as much as N layers", so the split is balanced by cost rather
    than by module count.

    Args:
        num_stages: Number of pipeline stages.
        num_layers: Total transformer layers in the model.
        input_weight: Cost of the input modules, in layers.
        output_weight: Cost of the output modules, in layers.

    Returns:
        One list of module names per stage.

    Example:
        ``generate_llm_fqn_per_model_part(2, 3, input_weight=2, output_weight=2)``
        treats the embedding and the norm+head as two layers each when deciding
        the split.
    """
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")

    if num_stages == 1:
        # Single stage gets everything.
        layer_names = [f"layers.{i}" for i in range(num_layers)]
        return [["tok_embeddings"] + layer_names + ["norm", "lm_head", "rotary_emb"]]

    num_effective_layers = num_layers + input_weight + output_weight

    if num_stages > num_effective_layers:
        raise ValueError(
            f"Number of stages ({num_stages}) cannot be greater than effective "
            f"layers ({num_effective_layers})"
        )

    # Layers per stage, distributing the remainder over the first stages.
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    if layers_per_stage == 0:
        raise ValueError(
            f"Configuration would result in empty stages. "
            f"With {num_stages} stages and {num_effective_layers} effective layers "
            f"(num_layers={num_layers} + input_weight={input_weight} + "
            f"output_weight={output_weight}), each stage would get "
            f"{layers_per_stage} layers on average. "
            f"Reduce num_stages or increase num_layers/weights."
        )

    # The weighted modules must fit inside a stage, or that stage would hold
    # nothing but the embedding (or nothing but the head).
    if input_weight > layers_per_stage:
        raise ValueError(
            f"input_weight ({input_weight}) exceeds minimum layers per stage "
            f"({layers_per_stage})."
        )
    if output_weight > layers_per_stage:
        raise ValueError(
            f"output_weight ({output_weight}) exceeds minimum layers per stage "
            f"({layers_per_stage})."
        )

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules = []

        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        if stage_idx == 0:
            # First stage owns the input modules.
            stage_modules.append("tok_embeddings")
            remaining_layers_for_stage = effective_layers_for_stage - input_weight
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        elif stage_idx == num_stages - 1:
            # Last stage owns the output modules.
            remaining_layers_for_stage = effective_layers_for_stage - output_weight
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1
            stage_modules.extend(["norm", "lm_head"])

        else:
            for _ in range(effective_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        stage_modules.append("rotary_emb")
        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def split_model_into_stages(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
    get_mesh: Callable | None = None,
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """Build this rank's pipeline stages from a per-stage module assignment.

    Each stage gets its own deep copy of the model with every module it does not
    own removed -- layers deleted from the ``ModuleList``, everything else
    replaced by ``nn.Identity``. A deep copy per stage (rather than a partial
    view) is what lets a rank own two stages of different shapes.

    Model-side requirements, inherited from torchtitan:
    - ``forward`` must tolerate deleted layers.
    - weight initialization must tolerate deleted layers.
    - nested ``ModuleDict`` / ``ModuleList`` structures are not supported.

    Args:
        whole_model: The complete model to split.
        pp_mesh: The pipeline-parallel device mesh.
        pp_schedule: Schedule name; used to decide looped vs V layout.
        device: Device the stages run on.
        module_names_per_stage: Module names for each stage, dot-separated:
            ``"tok_embeddings"``, ``"layers.0"``, ``"norm"``, ``"lm_head"``.
        get_mesh: Optional callback letting a stage resolve a ``DeviceMesh`` for
            an incoming tensor. hpmesh leaves this ``None``: it exists so stages
            can rebuild DTensors, and hpmesh passes plain tensors.

    Returns:
        ``(stages, models)`` -- the ``PipelineStage`` objects for this rank and
        their corresponding model chunks.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()

    def _build_stage_from_modules(
        stage_idx: int, module_names: list[str], num_stages: int
    ) -> tuple[PipelineStage, nn.Module]:
        model = copy.deepcopy(whole_model)
        modules_to_keep = set(module_names)

        for module_name, module_value in model.named_children():
            # Layer-like containers (e.g. "layers.0", "layers.1").
            if isinstance(module_value, nn.ModuleDict | nn.ModuleList):
                layers_to_keep = {
                    name.split(".", 1)[1]
                    for name in modules_to_keep
                    if name.startswith(f"{module_name}.")
                }
                if layers_to_keep:
                    if isinstance(module_value, nn.ModuleDict):
                        for layer_name in list(module_value.keys()):
                            if layer_name not in layers_to_keep:
                                del module_value[layer_name]
                    elif isinstance(module_value, nn.ModuleList):
                        indices_to_keep = {
                            int(idx) for idx in layers_to_keep if idx.isdigit()
                        }
                        # add_module with the ORIGINAL index as the key: a
                        # plain ``nn.ModuleList(kept)`` would renumber the kept
                        # layers from 0, and two stages' state-dict keys would
                        # then collide ("layers.0.*" meaning different layers
                        # on different ranks) in one shared checkpoint.
                        kept_layers = nn.ModuleList()
                        for i, layer in enumerate(module_value):
                            if i in indices_to_keep:
                                kept_layers.add_module(str(i), layer)
                        setattr(model, module_name, kept_layers)
                else:
                    # This stage uses none of the container's layers.
                    if isinstance(module_value, nn.ModuleDict):
                        setattr(model, module_name, nn.ModuleDict())
                    elif isinstance(module_value, nn.ModuleList):
                        setattr(model, module_name, nn.ModuleList())
            # Simple attributes (e.g. "norm", "lm_head") not owned by this stage
            # are replaced by an Identity so the forward still runs.
            elif module_name not in modules_to_keep:
                setattr(model, module_name, nn.Identity())

        # Extra top-level modules the model's ``named_children`` override does
        # not present (e.g. a multimodal encoder registered beside the decoder
        # on ``HFTransformerModel``, whose child iteration shows only the five
        # decoder parts). The loop above cannot see them, so keep-or-blank is
        # applied here against the real ``_modules`` children: a non-owner
        # gets an Identity, exactly like the decoder parts above. Without
        # this, the deep copy would leave a live copy on every stage and two
        # stages' state-dict keys would collide in one checkpoint. A module
        # listed in NO stage's FQN list is blanked everywhere -- the model's
        # ``forward`` must tolerate that (the same contract the decoder parts
        # already impose).
        presented = dict(model.named_children())
        presented_ids = {id(module) for module in presented.values()}
        for module_name, module_value in nn.Module.named_children(model):
            if module_name in presented or module_name in modules_to_keep:
                continue
            # A container that holds the presented parts (the wrapper's inner
            # HF model) is what the parts live in, not an extra module --
            # never blank it.
            if presented_ids & {id(m) for m in module_value.modules()}:
                continue
            setattr(model, module_name, nn.Identity())

        stage_kwargs = {"group": pp_mesh.get_group()}
        # ``get_mesh`` was added after PyTorch 2.10. hpmesh passes None because
        # its stages exchange plain tensors, so omitting it on older releases
        # has exactly the same semantics.
        if "get_mesh" in inspect.signature(PipelineStage).parameters:
            stage_kwargs["get_mesh"] = get_mesh
        stage = PipelineStage(model, stage_idx, num_stages, device, **stage_kwargs)
        return stage, model

    num_stages = len(module_names_per_stage)
    stages = []
    models = []

    schedule_class = get_schedule_class(pp_schedule)
    style = (
        "v" if schedule_class in (ScheduleZBVZeroBubble, ScheduleDualPipeV) else "loop"
    )

    def _get_stage_indices() -> tuple[int, ...]:
        """Stage ids this rank runs, for a looped or a V schedule."""
        assert num_stages % pp_size == 0, (
            f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
        )
        stages_per_rank = num_stages // pp_size
        if style == "loop":
            return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
        # "v": the rank takes one stage from each half, paired front-to-back.
        assert stages_per_rank == 2, (
            f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        )
        stage_v_pairs = list(
            zip(
                range(pp_size),
                range(num_stages - 1, pp_size - 1, -1),
                strict=True,
            )
        )
        return stage_v_pairs[pp_rank]

    for stage_idx in _get_stage_indices():
        module_names = module_names_per_stage[stage_idx]
        stage, model_chunk = _build_stage_from_modules(
            stage_idx,
            module_names,
            num_stages,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} "
            f"with modules {module_names}"
        )
        stages.append(stage)
        models.append(model_chunk)

    return stages, models
