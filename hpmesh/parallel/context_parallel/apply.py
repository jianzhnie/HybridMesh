"""Wire CP onto a model: attach the CP flex kernel to every decoder layer."""

from __future__ import annotations

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from hpmesh.trainer.config import ParallelConfig

from ...utils.logger_utils import get_logger
from .cp_kernel import CPFlexKernel

logger = get_logger(__name__)

__all__ = ["apply_cp"]

# HF names the attention submodule differently across model families; probe in
# the style of hf_wrapper's ``_first_present`` rather than hardcoding one
# family's spelling.
_ATTN_MODULE_NAMES = ("self_attn", "attn", "attention")


def apply_cp(
    model: nn.Module,
    mesh: DeviceMesh | None,
    cfg: ParallelConfig,
) -> nn.Module:
    """Attach a CP flex kernel to every decoder layer's attention module.

    No-op when ``cfg.cp == 1`` (the model is handed back untouched), so the
    caller needs no degree check of its own.

    The kernel declares the CP region: q/k/v reach it sequence-sharded and it
    redistributes them per the configured strategy (K/V all-gather, or ulysses
    all-to-all onto the head axis). The model inputs still have to be sharded
    along the sequence axis -- that is the trainer's half of the wiring
    (``shard_batch_for_cp``).
    """
    if cfg.cp == 1:
        return model

    # Lazy: ``models/hf_wrapper`` imports this package's ``__init__`` at module
    # scope (for the CP input sharding), and the ``__init__`` eagerly
    # re-exports ``apply_cp`` from this module -- a module-level import of
    # hf_wrapper here would close that cycle whenever hf_wrapper is imported
    # first. Keeping it at the call site is what lets both import orders work.
    from ...models.hf_wrapper import _ATTN_IMPLEMENTATION

    impl = getattr(getattr(model, "model", None), "config", None)
    impl = getattr(impl, "_attn_implementation", None)
    if impl != _ATTN_IMPLEMENTATION:
        raise RuntimeError(
            f"CP requires the {_ATTN_IMPLEMENTATION!r} attention backend, but "
            f"this model runs {impl!r}. CP expresses the sharded attention mask "
            "as a BlockMask, which only the flex path consumes -- the same "
            "reason the sdpa fallback cannot run packed sequences. On a "
            "CPU-only machine the wrapper selects 'sdpa', so CP needs CUDA."
        )
    if mesh is None or "cp" not in (mesh.mesh_dim_names or ()):
        raise ValueError(
            f"cp={cfg.cp} requires a device mesh with a 'cp' axis, got "
            f"{None if mesh is None else mesh.mesh_dim_names}."
        )
    cp_mesh = mesh["cp"]
    if cp_mesh.size() != cfg.cp:
        raise ValueError(
            f"mesh 'cp' axis has size {cp_mesh.size()}, but cfg requests cp={cfg.cp}."
        )

    strategy = cfg.context_parallel_strategy
    load_balancer = cfg.context_parallel_load_balancer
    if strategy == "ulysses":
        model_config = getattr(getattr(model, "model", None), "config", None)
        if load_balancer is not None:
            raise ValueError(
                "Ulysses CP requires context_parallel_load_balancer=None: the "
                "all-to-all is an even split of an evenly-sharded tensor, and "
                "every rank ends up attending the full sequence in whatever "
                "order the shards arrived in. A load balancer rearranges the "
                "sequence into head and tail chunks, so that order is no "
                "longer the original one and -- unlike the seq_len x seq_len "
                "causal mask, a function of tokens -- flex only takes a mask "
                "built over the attended order. The kernel therefore attends "
                "the rearranged corpus, a permuted model that trains and "
                "produces plausible numbers with nothing raised. "
                "strategy='kv_allgather' is what supports this: it gathers the "
                "rearranged shards back into the very order the mask was "
                "sharded in, so the rearrangement lands in the mask and cancels."
            )
        if getattr(model_config, "attn_mask_type", "causal") == "block_causal":
            raise ValueError(
                "Ulysses CP does not support packed sequences "
                "(attn_mask_type='block_causal'): the wrapper hands every CP "
                "kernel a Q-sharded BlockMask, but ulysses attends the full "
                "sequence and cannot recover the unsharded document mask. "
                "Use strategy='kv_allgather' for packed runs."
            )
        # TP shards heads first, so what ulysses must divide evenly is each
        # rank's local head count -- equivalently, the global count must divide
        # tp * cp (upstream's head_shard_degree in config/validation.py).
        head_shard_degree = cfg.tp * cp_mesh.size()
        for field_name in ("num_attention_heads", "num_key_value_heads"):
            heads = getattr(model_config, field_name, None) or getattr(
                model_config, "num_attention_heads", None
            )
            if heads is None or heads % head_shard_degree != 0:
                raise ValueError(
                    f"Ulysses CP shards heads across the group: {field_name} "
                    f"({heads}) must be divisible by tp*cp "
                    f"({cfg.tp}*{cp_mesh.size()}={head_shard_degree}), so the "
                    f"local heads per rank divide evenly across cp."
                )

    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"apply_cp expects a HFTransformerModel (with .layers); got "
            f"{type(model).__name__}."
        )
    for idx, layer in enumerate(layers):
        for name in _ATTN_MODULE_NAMES:
            if hasattr(layer, name):
                attn_mod = getattr(layer, name)
                break
        else:
            raise AttributeError(
                f"{type(layer).__name__} (layer {idx}) has no attention module "
                f"under any of {_ATTN_MODULE_NAMES}. Add the model's spelling "
                "to the probe in apply_cp."
            )
        attn_mod._titan_flex_kernel = CPFlexKernel(cp_mesh=cp_mesh, strategy=strategy)

    model.set_cp_mesh(cp_mesh, load_balancer=load_balancer)
    logger.info("Applied CP (%s) with degree %d", strategy, cfg.cp)
    return model
