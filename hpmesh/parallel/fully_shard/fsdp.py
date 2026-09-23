import logging
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

try:
    from torch.distributed.fsdp import DataParallelMeshDims
except ImportError:
    # PyTorch 2.10 (used by the current vLLM Ascend image) predates this public
    # type. hpmesh passes a dedicated one/two-dimensional FSDP mesh instead of
    # ``dp_mesh_dims``, so the symbol is only needed for annotations here.
    DataParallelMeshDims = Any
from torch.distributed.tensor import Shard
from torch.nn import ModuleDict

from ..parallel_dims import ParallelDims

logger = logging.getLogger(__name__)


def iter_transformer_layers(layers: nn.Module) -> Iterator[tuple[Any, nn.Module]]:
    """Yield ``(index, block)`` for the transformer block container.

    torchtitan's ``Decoder.layers`` is a ``ModuleDict`` keyed by index, so it
    iterates with ``.items()``; HF's ``PreTrainedModel`` stores a ``ModuleList``,
    which iterates by position. Both yield the same pairs.
    """
    if isinstance(layers, ModuleDict):
        return iter(layers.items())
    return iter(enumerate(layers))


def resolve_fsdp_mesh(parallel_dims: ParallelDims) -> DeviceMesh:
    """Build the dense FSDP-only submesh.

    hpmesh's HF models hold plain tensors, so ``fully_shard`` cannot take an
    explicit ``DataParallelMeshDims``: torch requires every parameter to be a
    DTensor on the full SPMD mesh in that mode ("When dp_mesh_dims is
    provided, all parameters must be DTensors ... via distribute_module").
    Without mesh dims, torch reads the mesh by shape alone: 1-D means plain
    FSDP, 2-D means HSDP (dim 0 replicates, dim 1 shards), and anything
    higher-dimensional raises. Handing FSDP the raw multi-axis storage mesh
    therefore mis-assigns the axes whenever that mesh also carries ``tp`` or
    more than two active axes.

    Instead, rebuild a dedicated submesh over exactly the axes torchtitan
    declares in its ``DataParallelMeshDims``:

    * shard: ``dp_shard`` (force-kept-alive in the dense storage mesh even at
      size 1, so pure DDP gets a well-defined HSDP shard axis) plus ``cp``
      when CP is enabled, flattened into a single axis when both are active
      (torchtitan's flattened shard semantics);
    * replicate: ``dp_replicate`` when replication is enabled.

    The result is at most 2-D with the replicate axis first, so torch's
    default reading coincides with the intended one. A size-1 result means no
    DP/CP axis is active and FSDP is a no-op; the caller checks for that.
    """
    shard_axes = ["dp_shard"]
    if parallel_dims.cp_enabled:
        shard_axes.append("cp")
    replicate_axis = "dp_replicate" if parallel_dims.dp_replicate_enabled else None

    axes = ([replicate_axis] if replicate_axis else []) + shard_axes
    submesh = parallel_dims.get_optional_mesh(axes)
    assert submesh is not None  # dp_shard is always kept alive

    if replicate_axis is None and len(shard_axes) == 1:
        # 1-D: plain FSDP over the shard axis.
        return submesh
    if replicate_axis is None:
        # 2-D (dp_shard, cp): flatten into torchtitan's single shard axis.
        return submesh._flatten("dp_shard_cp")
    if len(shard_axes) == 1:
        # 2-D (dp_replicate, dp_shard): HSDP as torch reads it by default.
        return submesh
    # 3-D (dp_replicate, dp_shard, cp): DeviceMesh has no partial flatten, so
    # rebuild from the rank tensor. Row-major reshape keeps dp_replicate on
    # dim 0 and folds (dp_shard, cp) into a single shard dim 1.
    flat_ranks = submesh.mesh.reshape(submesh.mesh.size(0), -1)
    return DeviceMesh(
        submesh.device_type,
        flat_ranks,
        mesh_dim_names=(replicate_axis, "dp_shard_cp"),
    )


def resolve_sparse_fsdp_mesh(parallel_dims: ParallelDims) -> DeviceMesh | None:
    """Sparse counterpart of ``resolve_fsdp_mesh`` for routed experts.

    Returns ``None`` when EP is disabled. Otherwise rebuilds the FSDP-only
    submesh over ``efsdp`` (shard) and, when enabled, ``dp_replicate``
    (replicate) -- the axes torchtitan declares as
    ``DataParallelMeshDims(shard="efsdp", replicate="dp_replicate")``. The
    raw sparse storage mesh also carries the ``ep`` axis, which torch's
    default 2-D reading would mistake for the shard axis, so it is excluded
    here the same way ``tp`` is excluded from the dense mesh.
    """
    if not parallel_dims.ep_enabled:
        return None
    axes = (["dp_replicate"] if parallel_dims.dp_replicate_enabled else []) + ["efsdp"]
    submesh = parallel_dims.get_optional_mesh(axes)
    assert submesh is not None  # efsdp is kept alive whenever ep > 1
    return submesh


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """
    Disable FSDP's automatic gradient division for all FSDP modules.

    Set gradient_divide_factor=1.0 to disable FSDP's automatic gradient division.
    We handle gradient scaling ourselves in the training loop with global token count.

    Note: This also works for ReplicateModule since it inherits from FSDPModule.

    Args:
        model: The model containing FSDP-wrapped or Replicate-wrapped modules
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


def _fsdp_shard_degree(dp_mesh: DeviceMesh) -> int:
    """The degree by which FSDP shards dim 0 over ``dp_mesh``.

    FSDP cuts a parameter's dim 0 only over its shard axes: ``dp_shard``,
    plus ``cp`` when CP is on (``resolve_fsdp_mesh`` folds the two into
    ``dp_shard_cp``). ``dp_replicate`` replicates, so it must not inflate the
    degree compared against ``num_experts`` -- under HSDP (dp_replicate > 1)
    the raw mesh size would over-count and mis-pick ``Shard(1)`` for the
    expert weights.
    """
    degree = dp_mesh.size()
    if "dp_replicate" in (dp_mesh.mesh_dim_names or ()):
        degree //= dp_mesh["dp_replicate"].size()
    return degree


def enable_fsdp_symm_mem(model: nn.Module, scope: str | None = "all") -> None:
    """Enable symmetric-memory communication for the FSDP modules ``scope`` selects.

    ``None`` disables it. ``"all"`` covers every FSDP module; ``"dense"``
    skips any module flagged ``moe_enabled`` -- an MoE transformer block is one
    FSDP module, so its attention parameters are skipped along with its
    experts. Symmetric memory is not always beneficial for the expert (sparse)
    FSDP modules, hence the narrower scope.
    """
    if scope is None:
        return
    if scope not in ("all", "dense"):
        raise ValueError(
            f"enable_fsdp_symm_mem scope must be one of 'all', 'dense', None; "
            f"got {scope!r}"
        )
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        if scope == "dense" and getattr(module, "moe_enabled", False):
            continue
        module.set_force_sum_reduction_for_comms(True)
        module.set_symm_mem_for_comm()


def get_fsdp_reshard_after_forward_policy(
    reshard_after_forward_policy: str, pp_enabled: bool
) -> bool:
    """Resolve fsdp_reshard_after_forward policy string to a boolean.

    Args:
        reshard_after_forward_policy: One of "always", "never", or "default".
        pp_enabled: Whether pipeline parallelism is enabled.

    Returns:
        Boolean indicating whether to reshard after forward.
    """
    match reshard_after_forward_policy:
        case "always":
            return True
        case "never":
            return False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            return not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )


def apply_fsdp_to_vision_encoder(
    vision_encoder: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward_policy: str = "default",
    pp_enabled: bool = False,
    cpu_offload: bool = False,
    *,
    dp_mesh_dims: DataParallelMeshDims | None = None,
) -> None:
    """FSDP a VLM vision encoder as a single unit.

    One all-gather for all vision params is more efficient than per-layer sharding
    (the vision encoder is small relative to the decoder). Call before
    ``apply_fsdp_to_decoder`` so the encoder is already sharded.

    ``cpu_offload`` must match what the caller passes to ``apply_fsdp_to_decoder``.
    Under ``training.enable_cpu_offload`` the trainer materializes the whole model
    on CPU, so a vision encoder sharded without ``CPUOffloadPolicy`` keeps CPU
    parameters while FSDP produces CUDA gradients for them, and backward dies with
    "attempting to assign a gradient with device type 'cuda' to a tensor with
    device type 'cpu'".
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled=pp_enabled
    )
    fsdp_config: dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": mp_policy,
        "reshard_after_forward": reshard_after_forward,
    }
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
    fully_shard(vision_encoder, **fsdp_config)


def apply_fsdp_to_decoder(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_size: int = 1,
    edp_mesh: DeviceMesh | None = None,
    dp_mesh_dims: "DataParallelMeshDims | None" = None,
    edp_mesh_dims: "DataParallelMeshDims | None" = None,
    enable_symm_mem: bool = False,
):
    """
    Apply data parallelism (via FSDP2) to a decoder-style transformer model.

    Shared by all dense and MoE decoders (llama3, qwen3, deepseek_v3,
    gpt_oss, qwen3_vl, ...). The MoE handling is a strict superset of the dense
    case: a dense model leaves ``ep_size=1`` / ``edp_mesh=None`` and has no
    ``moe_enabled`` blocks, so every transformer block is sharded as a single
    FSDP unit and the expert-parallel prefetching below is skipped.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reductions.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to
            CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for
            resharding after the forward pass. Defaults to "default". Other
            options: "never", "always".
            - "default" applies default resharding behavior, implementing
              "smart defaults" for known optimal scenarios.
            - "always" enables ``reshard_after_forward`` for all forward passes.
            - "never" disables ``reshard_after_forward`` for all forward passes.
        ep_size (int, optional): Expert-parallel degree. Defaults to 1 (no EP),
            in which case the MoE-specific sharding and prefetching are no-ops.
        edp_mesh (DeviceMesh | None, optional): The FSDP mesh for routed experts
            when EP > 1. Required (non-None) iff ``ep_size > 1``.
        dp_mesh_dims: Under spmd_types, ``fully_shard`` must flatten
            ``dp_shard`` and ``cp`` into a single FSDP shard dim, so it
            needs to know which axes of the multi-dimensional SPMD mesh are
            data-parallel. We pass this explicitly via ``dp_mesh_dims``
            rather than letting FSDP infer it from mesh axis names: the
            naming contract between ``fully_shard`` and torchtitan is not
            strong enough to infer safely, and an explicit declaration
            avoids silent miscategorization when new mesh axes appear.
        edp_mesh_dims: Sibling of ``dp_mesh_dims`` for the sparse SPMD mesh
            used by routed experts.
        enable_symm_mem (bool): Whether to enable symmetric-memory FSDP
            communication.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if model.enable_weight_tying:
        # When weights are tied, tok_embeddings and output share the same parameter.
        # Group them together in one FSDP unit to avoid duplicate all-gathers.
        modules = [
            m
            for m in (model.tok_embeddings, model.norm, model.lm_head)
            if m is not None
        ]
        fully_shard(
            modules,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    else:
        if model.tok_embeddings is not None:
            fully_shard(
                model.tok_embeddings,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        # As an optimization, do not reshard_after_forward the last layers
        # by default since FSDP would prefetch them immediately.
        if model.norm is not None and model.lm_head is not None:
            fully_shard(
                [model.norm, model.lm_head],
                **fsdp_config,
                reshard_after_forward=reshard_after_forward_policy == "always",
            )

    for _layer_id, transformer_block in iter_transformer_layers(model.layers):
        # NOTE: In an MoE layer, we use shard_placement_fn to apply different
        # FSDP mesh and shard placement to different parameters:
        # - When EP > 1: routed experts use edp_mesh, other params use dp_mesh
        # - When EP = 1: all params use the same FSDP mesh, but experts may
        #   use Shard(1) when FSDP degree > num_experts to avoid padding
        # Dense blocks (no ``moe_enabled``) fall through to a plain fully_shard.
        if getattr(transformer_block, "moe_enabled", False):
            assert hasattr(transformer_block, "moe")
            # Expert weights live on the grouped-GEMM child (inner_experts).
            # pyrefly: ignore [missing-attribute]
            experts = transformer_block.moe.routed_experts.inner_experts
            expert_params = set(experts.parameters())
            # The total expert count, read off the router. hpmesh's grouped
            # weights are built per rank by the EP swap, so
            # ``experts.num_experts`` is only this rank's slice (total / ep);
            # the router is the child that always holds the total. Upstream
            # reads ``experts.num_experts`` instead and gets the same number,
            # because its experts are an SPMD DTensor sharded on ``ep`` with the
            # logical count intact. So this is a spelling difference forced by
            # the different expert representation, not a different threshold --
            # both sides compare ``efsdp * ep`` against the same total.
            num_experts = transformer_block.moe.router.num_experts

            if ep_size > 1:
                assert edp_mesh is not None
                efsdp_ep_size = edp_mesh["efsdp"].size() * ep_size
            else:
                efsdp_ep_size = _fsdp_shard_degree(dp_mesh)

            if efsdp_ep_size > num_experts:
                expert_shard_placement = Shard(1)
            else:
                expert_shard_placement = Shard(0)

            # When ep_size == 1 and no Shard(1) override needed, skip
            # shard_placement_fn entirely for simplicity
            if ep_size == 1 and expert_shard_placement == Shard(0):
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
            elif ep_size == 1:
                # ep_size == 1 but need Shard(1) for experts to avoid padding
                def _experts_shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                ) -> Shard | None:
                    if param in _expert_params:
                        return Shard(1)
                    return None

                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=_experts_shard_placement_fn,
                )
            else:
                # ep_size > 1: per-param mesh
                from torch.distributed.fsdp._fully_shard._fsdp_common import (
                    FSDPMeshInfo,
                    ShardPlacementResult,
                )
                from torch.distributed.fsdp._fully_shard._fsdp_init import (
                    _get_mesh_info,
                )

                assert edp_mesh is not None

                # Delegate to FSDP2's mesh-info builder. When mesh_dims is set
                # it extracts and FLATTENS the DP submesh from the full SPMD
                # mesh.
                edp_mesh_info = _get_mesh_info(edp_mesh, edp_mesh_dims)
                dp_mesh_info = _get_mesh_info(dp_mesh, dp_mesh_dims)
                # _get_mesh_info is typed to the DataParallelMeshInfo base; with
                # a shard dim it always yields FSDPMeshInfo/HSDPMeshInfo.
                assert isinstance(edp_mesh_info, FSDPMeshInfo)
                assert isinstance(dp_mesh_info, FSDPMeshInfo)

                def _shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                    _expert_placement: Shard = expert_shard_placement,
                    _edp_mesh_info: FSDPMeshInfo = edp_mesh_info,
                    _dp_mesh_info: FSDPMeshInfo = dp_mesh_info,
                ) -> ShardPlacementResult:
                    if param in _expert_params:
                        return ShardPlacementResult(
                            placement=_expert_placement, mesh_info=_edp_mesh_info
                        )
                    else:
                        return ShardPlacementResult(
                            placement=Shard(0), mesh_info=_dp_mesh_info
                        )

                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=_shard_placement_fn,
                )
        else:
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    fully_shard(model, **fsdp_config)

    if enable_symm_mem:
        enable_fsdp_symm_mem(model, "all")

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)

    # HSDP when the data-parallel mesh carries a replicate axis, else pure FSDP.
    if "dp_replicate" in (dp_mesh.mesh_dim_names or ()):
        logger.info("Applied HSDP to the model")
    else:
        logger.info("Applied FSDP to the model")
    if cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    # NOTE: set up explicit prefetching when EP is enabled, as D2H syncs
    # in EP could interfere with implicit prefetching in FSDP
    if ep_size == 1:
        return

    # set up explicit prefetching when EP is enabled for forward
    transformer_blocks = [block for _, block in iter_transformer_layers(model.layers)]
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if model.tok_embeddings is not None and transformer_blocks:
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks, strict=False
    ):
        if next_transformer_block is not None:
            # pyrefly: ignore [not-callable]
            transformer_block.set_modules_to_forward_prefetch([next_transformer_block])
        elif model.norm is not None and model.lm_head is not None:
            # pyrefly: ignore [not-callable]
            transformer_block.set_modules_to_forward_prefetch(
                [model.norm, model.lm_head]
            )

    # set up explicit prefetching when EP is enabled for backward
    # pyrefly: ignore [no-matching-overload]
    reversed_transformer_blocks = list(reversed(transformer_blocks))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if model.norm is not None and model.lm_head is not None and transformer_blocks:
        model.lm_head.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks, strict=False
    ):
        if prev_transformer_block is not None:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_backward_prefetch([prev_transformer_block])
        elif model.tok_embeddings is not None:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])
