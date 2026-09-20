"""hpmesh's FSDP entry point.

``fsdp.py`` (next to this file) holds the sharding logic, vendored from
torchtitan's ``distributed/fsdp.py``. This module is the thin driver around it:
the no-op guards that let one trainer run unchanged from a single device up to a
full mesh, the policy values hpmesh fixes, and one backend workaround.

The shape adapter that used to live here is gone, because the shape mismatch it
existed to paper over is now handled on both sides where it belongs:

* torchtitan's FSDP reads ``model.layers`` as an index-keyed mapping (its
  ``Decoder`` uses a ``ModuleDict``); HF stores a ``ModuleList``. ``fsdp.py``
  now iterates through ``iter_transformer_layers``, which accepts either.
* torchtitan's FSDP names ``tok_embeddings`` / ``norm`` / ``lm_head`` on the
  top-level module and reads ``enable_weight_tying`` off it; HF nests those
  under ``model.model``. ``HFTransformerModel`` now exposes all five directly.

With nothing left to adapt, applying FSDP to an HF model is just calling FSDP
on it.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.device_mesh import DeviceMesh

from ...trainer.config import HybridMeshConfig
from ..parallel_dims import ParallelDims
from .fsdp import (
    apply_fsdp_to_decoder,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)

__all__ = ["apply_fsdp"]


def _force_sum_grad_reduction(model: torch.nn.Module) -> None:
    """Use plain SUM for FSDP's gradient reduce comms.

    FSDP defaults to ``ReduceOp.PREMUL_SUM``, which pre-scales the local
    gradient so the collective needs no separate scaling kernel -- but only NCCL
    implements it, so every other backend raises. Forcing plain SUM gives up
    that optimization, which is why the caller only turns it on off-NCCL.
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_force_sum_reduction_for_comms(True)


def apply_fsdp(
    model: torch.nn.Module,
    mesh: DeviceMesh | None,
    cfg: HybridMeshConfig,
    parallel_dims: ParallelDims | None = None,
) -> torch.nn.Module:
    """Fully-shard ``model`` (FSDP2). No-op when data parallelism is disabled.

    Delegates the wrapping to torchtitan's ``apply_fsdp_to_decoder``; this
    function only decides whether to shard at all, fixes the policy values, and
    applies the backend workaround.
    """
    if parallel_dims is None or not parallel_dims.dp_shard_enabled:
        return model

    dp_mesh = parallel_dims.get_optional_mesh("dp_shard")
    if dp_mesh is None or dp_mesh.size() == 1:
        return model

    storage_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    edp_mesh, _edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)

    # torch rejects ``dp_mesh_dims`` unless every parameter is already a DTensor
    # on the full SPMD mesh ("When dp_mesh_dims is provided, all parameters must
    # be DTensors ... via distribute_module"). Meeting that precondition means
    # converting each declared state into a DTensor before FSDP is applied;
    # hpmesh's HF models hold plain tensors. So we hand FSDP a plain 1-D DP mesh
    # and let it do its own sharding -- the mode torch supports out of the box.
    # Wiring the DTensor path is a prerequisite for composing FSDP with
    # tp/cp/ep on one mesh, and is not done here.
    dp_mesh_dims = None
    edp_mesh_dims = None

    # Vectors are lenient: the model may be on CPU (this learning path runs on
    # CPU/gloo) or bf16 (the CUDA default in torchtitan's trainer).
    param_dtype = torch.float32
    reduce_dtype = torch.float32

    apply_fsdp_to_decoder(
        model,
        storage_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=False,
        reshard_after_forward_policy=cfg.parallel.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        enable_symm_mem=cfg.parallel.enable_fsdp_symm_mem,
    )
    # ``apply_fsdp_to_decoder`` already calls ``disable_fsdp_gradient_division``
    # and, when asked, ``enable_fsdp_symm_mem``; do not repeat them here.

    # gloo implements no PREMUL_SUM; NCCL does, and there it is the faster path.
    if dist.is_initialized() and dist.get_backend() != "nccl":
        _force_sum_grad_reduction(model)

    return model
