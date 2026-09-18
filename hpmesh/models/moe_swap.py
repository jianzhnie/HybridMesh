"""Swap an HF MoE block for hpmesh's native one, copying its weights across.

This is the bridge between the two worlds the rest of the framework keeps apart:
HuggingFace owns the model, hpmesh owns the expert compute. It reads the HF
block's layout, builds the matching native ``MoE``, copies the weights, and
replaces the block in place. After this runs, ``layer.mlp`` (or whichever
attribute held the block) is a ``hpmesh`` module holding the *same* weights.

Why copy rather than re-initialize: the native and HF MoE must be numerically
comparable, and the only way to prove a swap changed nothing is to feed both the
same weights and the same tokens. Re-initializing would give a model that is
merely different, and there would be nothing to check against.

What varies across HF versions and families, and is handled here:

* **Fused vs per-expert weights.** transformers 5.x stores
  ``experts.gate_up_proj`` as one ``(E, 2I, D)`` tensor and ``experts.down_proj``
  as ``(E, D, I)``. transformers 4.x and Mixtral store an ``nn.ModuleList`` of
  per-expert modules with ``gate_proj``/``up_proj``/``down_proj``. Both are read.
* **Where the expert count lives.** Upstream torchtitan reads
  ``gate_up_proj.shape[1] // 2`` for the hidden size; that attribute does not
  exist in 4.x, so ``moe_probe`` is used instead -- it already resolves the count
  and hidden size across families, and knows about Mixtral's
  ``block_sparse_moe``.

Under EP>1 each rank keeps only its own contiguous slice of experts; the router
stays global so every rank can score every token against every expert, which is
what the dispatcher's count exchange assumes.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn as nn

from .common.grouped_experts import GroupedExperts
from .common.moe import MoE, RoutedExperts, TokenChoiceTopKRouter
from .common.token_dispatcher import AllToAllTokenDispatcher, LocalTokenDispatcher
from .moe_probe import MoEArchitecture, probe_moe_model

logger = logging.getLogger(__name__)

__all__ = ["build_native_moe", "swap_moe_layers", "copy_moe_weights"]


def _hf_experts_container(hf_block: nn.Module) -> nn.Module:
    """Return the module/ModuleList holding the experts."""
    experts = getattr(hf_block, "experts", None)
    if experts is None:
        raise ValueError(
            f"MoE block {type(hf_block).__name__} has no 'experts' attribute"
        )
    return experts


def copy_moe_weights(
    native: MoE,
    hf_block: nn.Module,
    *,
    expert_offset: int,
    num_local_experts: int,
) -> None:
    """Copy one HF MoE block's weights into ``native``.

    Args:
        native: the hpmesh MoE to fill. Its modules must already exist, sized
            for ``num_local_experts``.
        hf_block: the HF block being replaced.
        expert_offset: index of this rank's first expert in the global ordering.
            Zero unless EP is on.
        num_local_experts: how many experts this rank owns.

    Raises:
        ValueError: if the HF layout is not one the copier recognizes.
    """
    experts = _hf_experts_container(hf_block)
    lo, hi = expert_offset, expert_offset + num_local_experts
    native_experts = native.routed_experts.inner_experts
    w1, w3, w2 = (
        native_experts.w1_EFD,
        native_experts.w3_EFD,
        native_experts.w2_EDF,
    )

    # Fused layout: transformers 5.x. gate_up_proj (E, 2I, D) holds gate and up
    # stacked along its middle dim; slicing it in half recovers the two.
    gate_up_proj = getattr(experts, "gate_up_proj", None)
    if gate_up_proj is not None:
        intermediate = w1.shape[1]
        local = gate_up_proj[lo:hi]
        with torch.no_grad():
            w1.copy_(local[:, :intermediate, :])
            w3.copy_(local[:, intermediate:, :])
            w2.copy_(experts.down_proj[lo:hi])
        _copy_router(native, hf_block)
        return

    # Per-expert layout: transformers 4.x and Mixtral. ``experts`` is an
    # nn.ModuleList of modules, one per expert.
    if not isinstance(experts, nn.ModuleList):
        raise ValueError(
            f"Unrecognized HF expert layout: {type(experts).__name__} has neither "
            "'gate_up_proj' (transformers 5.x) nor a per-expert ModuleList "
            "(transformers 4.x / Mixtral)."
        )
    if len(experts) != expert_offset + num_local_experts and expert_offset == 0:
        # Without EP the slice must cover everything; a mismatch means the probe
        # and the module disagree, which would silently drop experts.
        raise ValueError(
            f"probe reported {num_local_experts} experts but the HF block has "
            f"{len(experts)}; refusing to copy a partial expert set."
        )

    with torch.no_grad():
        for local_idx, global_idx in enumerate(range(lo, hi)):
            expert = experts[global_idx]
            w1[local_idx].copy_(expert.gate_proj.weight)
            w3[local_idx].copy_(expert.up_proj.weight)
            w2[local_idx].copy_(expert.down_proj.weight)
    _copy_router(native, hf_block)


def _copy_router(native: MoE, hf_block: nn.Module) -> None:
    """Copy the router (and shared-expert) weights, if present."""
    gate = getattr(hf_block, "gate", None)
    if gate is None:
        gate = getattr(hf_block, "router", None)
    if gate is not None and hasattr(gate, "weight"):
        with torch.no_grad():
            native.router.gate.weight.copy_(gate.weight)

    # Shared experts are a plain dense FFN; the native MoE takes one as a
    # module, and its weights live under different attribute names per family.
    shared = native.shared_experts
    hf_shared = getattr(hf_block, "shared_expert", None)
    if hf_shared is None:
        hf_shared = getattr(hf_block, "shared_experts", None)
    if shared is not None and hf_shared is not None:
        raise NotImplementedError(
            "shared-expert weight transfer is not implemented yet; build the "
            "shared FFN without one, or add its attribute mapping here."
        )


def build_native_moe(
    arch: MoEArchitecture,
    hf_block: nn.Module,
    *,
    ep_group: dist.ProcessGroup | None = None,
    num_local_experts: int | None = None,
    expert_offset: int = 0,
    use_grouped_mm: bool = False,
) -> MoE:
    """Build a native ``MoE`` matching ``arch`` and fill it from ``hf_block``.

    Args:
        arch: the probed description of the HF block.
        hf_block: the HF block whose weights are copied.
        ep_group: the expert-parallel group; ``None`` keeps every expert local.
        num_local_experts: experts this rank owns. Defaults to all of them.
        expert_offset: this rank's first expert in the global ordering.
        use_grouped_mm: run the expert GEMMs through ``torch._grouped_mm``
            instead of a per-expert loop. See ``GroupedExperts._grouped_mm``.

    Returns:
        The native MoE, on ``hf_block``'s device and in its dtype.
    """
    if num_local_experts is None:
        num_local_experts = arch.num_experts

    grouped = GroupedExperts(
        dim=arch.dim,
        hidden_dim=arch.moe_intermediate_size,
        num_experts=num_local_experts,
        use_grouped_mm=use_grouped_mm,
    )
    router = TokenChoiceTopKRouter(
        num_experts=arch.num_experts,
        dim=arch.dim,
        top_k=arch.top_k,
        score_func=arch.score_func,
        route_norm=arch.route_norm,
        route_scale=arch.route_scale,
    )
    dispatcher: LocalTokenDispatcher
    if ep_group is None or dist.get_world_size(ep_group) == 1:
        dispatcher = LocalTokenDispatcher(arch.num_experts, arch.top_k)
    else:
        dispatcher = AllToAllTokenDispatcher(arch.num_experts, arch.top_k)
        dispatcher.wire_meshes(ep_group=ep_group)

    native = MoE(
        num_experts=arch.num_experts,
        routed_experts=RoutedExperts(grouped, dispatcher),
        router=router,
        load_balance_coeff=arch.load_balance_coeff,
        shared_experts=None,
    )

    # Match the HF block's device/dtype before copying, so the copy is a plain
    # in-place write rather than a cross-device transfer per tensor.
    param = next(hf_block.parameters())
    native = native.to(device=param.device, dtype=param.dtype)

    copy_moe_weights(
        native,
        hf_block,
        expert_offset=expert_offset,
        num_local_experts=num_local_experts,
    )
    return native


def swap_moe_layers(
    model: nn.Module,
    *,
    ep_group: dist.ProcessGroup | None = None,
    use_grouped_mm: bool = False,
) -> int:
    """Replace every MoE block in ``model`` with a native one. Returns the count.

    The blocks' weights are copied, not re-initialized, so the model's function
    is unchanged by the swap -- which is what makes the swap testable.

    Under EP the experts are split evenly across ``ep_group``; the router is not,
    since every rank must score every token against every expert.

    Args:
        model: the HF causal LM to modify in place.
        ep_group: expert-parallel group, or ``None`` for no EP.
        use_grouped_mm: see ``GroupedExperts._grouped_mm``.
    """
    config = getattr(model, "config", None)
    architectures = probe_moe_model(model, config)
    if not architectures:
        return 0

    ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
    ep_rank = dist.get_rank(ep_group) if ep_group is not None else 0

    swapped = 0
    for layer_idx, arch in architectures.items():
        layer = _layer_at(model, layer_idx)
        hf_attr, hf_block = _find_moe_block(layer)

        if arch.num_experts % ep_size != 0:
            raise ValueError(
                f"layer {layer_idx}: {arch.num_experts} experts is not divisible by "
                f"ep_size={ep_size}"
            )
        num_local = arch.num_experts // ep_size
        native = build_native_moe(
            arch,
            hf_block,
            ep_group=ep_group,
            num_local_experts=num_local,
            expert_offset=ep_rank * num_local,
            use_grouped_mm=use_grouped_mm,
        )
        setattr(layer, hf_attr, native)
        # ``moe`` is the name fsdp.py reaches for when it sees ``moe_enabled``.
        object.__setattr__(layer, "moe", native)
        object.__setattr__(layer, "moe_enabled", True)
        swapped += 1
        logger.info(
            "layer %d: swapped %s for a native MoE (%d/%d experts on this rank)",
            layer_idx,
            type(hf_block).__name__,
            num_local,
            arch.num_experts,
        )
    return swapped


def _layer_at(model: nn.Module, layer_idx: int) -> nn.Module:
    """Return the decoder layer with index ``layer_idx``."""
    for path in ("model.layers", "model.model.layers", "layers"):
        container: nn.Module | None = model
        for attr in path.split("."):
            container = getattr(container, attr, None)
            if container is None:
                break
        if container is not None:
            return container[layer_idx]
    raise ValueError(
        f"could not locate the decoder layer list on {type(model).__name__}"
    )


# Mirrors the probe's attribute list; kept next to the swapper so a new family
# added to one is hard to miss in the other.
_MOE_BLOCK_ATTRS = ("mlp", "block_sparse_moe", "feed_forward", "moe")


def _find_moe_block(layer: nn.Module) -> tuple[str, nn.Module]:
    """Return ``(attribute_name, block)`` for the layer's MoE block."""
    for attr in _MOE_BLOCK_ATTRS:
        block = getattr(layer, attr, None)
        if block is not None and hasattr(block, "experts"):
            return attr, block
    raise ValueError(
        f"layer {type(layer).__name__} has an MoE architecture but no block under "
        f"any of {_MOE_BLOCK_ATTRS}"
    )
