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

from hpmesh.trainer.config import ParallelConfig

from ..parallel_dims import ParallelDims
from .fsdp import (
    _iter_fsdp_modules,
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
    for module in _iter_fsdp_modules(model):
        module.set_force_sum_reduction_for_comms(True)


def apply_fsdp(
    model: torch.nn.Module,
    cfg: ParallelConfig,
    parallel_dims: ParallelDims | None = None,
) -> torch.nn.Module:
    """Fully-shard ``model`` (FSDP2). No-op when no DP/CP axis is active.

    Applied unconditionally whenever any of ``dp_replicate`` / ``dp_shard`` /
    ``cp`` is active, mirroring torchtitan (``llama3/parallelize.py``):

    * CP must enable FSDP even when ``dp_shard == 1``: every CP rank computes
      the loss over its own sequence shard, so its gradients are partial and
      only become global once reduced across the CP group. The FSDP submesh
      puts ``cp`` on the shard axis for exactly that reason -- parameters are
      sharded over the CP group too, the same semantic torchtitan uses.
    * ``dp_replicate`` alone (pure DDP) must still reduce gradients across
      the replica group: with a shard axis of size 1 the all-gather is a
      no-op and only the gradient all-reduce remains. Skipping FSDP here
      would leave replicas reading different data with gradients never
      reduced.

    The mesh handed to FSDP is the dedicated submesh from
    ``resolve_fsdp_mesh`` -- never the raw storage mesh, whose extra axes
    (``tp``, or more than two active axes) torch's default shape-based
    reading would mis-assign or reject.

    Delegates the wrapping to torchtitan's ``apply_fsdp_to_decoder``; this
    function only decides whether to shard at all, fixes the policy values,
    and applies the backend workaround.
    """
    if parallel_dims is None:
        return model

    storage_mesh = resolve_fsdp_mesh(parallel_dims)
    if storage_mesh.size() == 1:
        return model

    edp_mesh = resolve_sparse_fsdp_mesh(parallel_dims)

    # Preserve the dtype selected at model construction. Hard-coding float32
    # here silently makes FSDP all-gather BF16 parameters as FP32, doubling the
    # transient parameter and operator footprint in mixed-precision runs.
    try:
        param_dtype = next(model.parameters()).dtype
    except StopIteration:
        param_dtype = torch.get_default_dtype()
    reduce_dtype = torch.float32

    apply_fsdp_to_decoder(
        model,
        storage_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=False,
        reshard_after_forward_policy=cfg.fsdp_reshard_after_forward,
        ep_size=parallel_dims.ep,
        edp_mesh=edp_mesh,
        symm_mem_scope=(
            cfg.fsdp_symm_mem_scope if cfg.enable_fsdp_symm_mem else None
        ),
    )
    # ``apply_fsdp_to_decoder`` already calls ``disable_fsdp_gradient_division``
    # and, when asked, ``enable_fsdp_symm_mem``; do not repeat them here.

    # gloo implements no PREMUL_SUM; NCCL does, and there it is the faster path.
    if dist.is_initialized() and dist.get_backend() != "nccl":
        _force_sum_grad_reduction(model)

    return model
