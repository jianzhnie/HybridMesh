"""Swap a HF MoE block for hpmesh's EP-capable MoE, weights included.

HF MoE architectures (Qwen3Moe is the probe target) give every sparse decoder
layer a block holding a router linear (``gate``) and a ``ModuleList`` of
per-expert MLPs (``gate_proj``/``up_proj``/``down_proj``). Expert parallelism
needs the experts stacked into grouped-GEMM weights with a token dispatcher in
front -- the stack in ``models/common/`` (``GroupedExperts`` /
``TokenChoiceTopKRouter`` / ``MoE``). This module is the bridge: it detects
the HF block, moves the weights into the hpmesh layout, and replaces
``layer.mlp``.

Weight layout: HF stores each projection as ``(out, in)`` and applies it as
``x @ W.T``; ``GroupedExperts`` stores the same orientation per expert
(``w1_EFD``/``w3_EFD`` are ``(F, D)``, ``w2_EDF`` is ``(D, F)``) and applies
it with the same ``F.linear``. The copy is therefore elementwise -- no
transpose, no regrouping -- and ``tests/test_ep_swap.py`` pins that against
the HF block it replaces.

Routing parity with the HF block: Qwen3Moe scores with a softmax over fp32
logits and renormalizes the top-k weights when ``norm_topk_prob`` is set;
``RouterGateLinear`` already computes in fp32, so ``score_func="softmax"``
plus ``route_norm=norm_topk_prob`` reproduces it. The one deliberate addition
is the load-balance machinery HF never gets to run through this wrapper (the
wrapper never asks for router logits): the router carries a
``MicrobatchWiseLoadBalanceLoss`` with the config's ``router_aux_loss_coef``,
whose gradient is injected on backward (see ``models/common/aux_loss.py``).
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn as nn

from ..models.common.grouped_experts import GroupedExperts
from ..models.common.moe import (
    MicrobatchWiseLoadBalanceLoss,
    MoE,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from ..models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)

logger = logging.getLogger(__name__)

__all__ = ["swap_hf_moe_blocks"]


def _is_hf_moe_block(module: nn.Module) -> bool:
    """Structural probe for the HF sparse-MoE block shape.

    Deliberately duck-typed rather than an ``isinstance`` against one family's
    class: Qwen3Moe, Llama4 and friends share this shape (router ``gate`` plus
    per-expert ``gate_proj``/``up_proj``/``down_proj`` MLPs) under different
    class names.
    """
    experts = getattr(module, "experts", None)
    return (
        isinstance(getattr(module, "gate", None), nn.Linear)
        and isinstance(getattr(module, "top_k", None), int)
        and isinstance(experts, nn.ModuleList)
        and len(experts) > 0
        and all(
            hasattr(expert, proj)
            for expert in experts
            for proj in ("gate_proj", "up_proj", "down_proj")
        )
    )


def _convert_block(
    block: nn.Module,
    *,
    ep_group: dist.ProcessGroup | None,
    aux_loss_coeff: float | None,
) -> MoE:
    """Build the hpmesh MoE for one HF block and move its weights over."""
    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    ep_rank = 0 if ep_group is None else dist.get_rank(ep_group)

    num_experts = len(block.experts)
    top_k = block.top_k
    dim = block.gate.in_features
    hidden = block.experts[0].gate_proj.out_features
    if num_experts % ep_size != 0:
        raise ValueError(
            f"EP degree {ep_size} does not divide num_experts={num_experts}; "
            "each EP rank must hold the same number of experts."
        )
    num_local = num_experts // ep_size
    lo = ep_rank * num_local

    grouped = GroupedExperts(dim, hidden, num_local)
    router = TokenChoiceTopKRouter(
        num_experts,
        dim,
        top_k,
        score_func="softmax",
        route_norm=bool(getattr(block, "norm_topk_prob", False)),
        aux_loss=(
            MicrobatchWiseLoadBalanceLoss(coeff=aux_loss_coeff)
            if aux_loss_coeff
            else None
        ),
    )
    if ep_group is None:
        dispatcher = LocalTokenDispatcher(num_experts, top_k)
    else:
        dispatcher = AllToAllTokenDispatcher(num_experts, top_k)
        dispatcher.wire_meshes(ep_group=ep_group)

    shared = getattr(block, "shared_expert", None)
    if shared is not None and hasattr(block, "shared_expert_gate"):
        raise NotImplementedError(
            f"{type(block).__name__} gates its shared expert "
            "(shared_expert_gate); MoE's shared_experts is additive only. "
            "Qwen3Moe has no shared expert, so this is unreachable there."
        )

    moe = MoE(
        num_experts=num_experts,
        routed_experts=RoutedExperts(grouped, dispatcher),
        router=router,
        # The auxiliary-loss-free bias stays off: nothing updates it in
        # hpmesh, and a frozen zero bias is dead state in the checkpoint.
        load_balance_coeff=None,
        shared_experts=shared,
    )

    # Match the block's dtype/device before the copies so they are exact, and
    # keep its mode: a fresh module defaults to training=True, which would
    # flip eval-built models (aux-loss injection, token counting) back on.
    moe.to(dtype=block.gate.weight.dtype, device=block.gate.weight.device)
    moe.train(block.training)
    with torch.no_grad():
        router.gate.weight.copy_(block.gate.weight)
        for local_idx, hf_expert in enumerate(block.experts[lo : lo + num_local]):
            grouped.w1_EFD[local_idx].copy_(hf_expert.gate_proj.weight)
            grouped.w3_EFD[local_idx].copy_(hf_expert.up_proj.weight)
            grouped.w2_EDF[local_idx].copy_(hf_expert.down_proj.weight)
    return moe


def swap_hf_moe_blocks(model: nn.Module, *, ep_group=None) -> int:
    """Replace every HF MoE block in ``model`` with hpmesh's MoE, in place.

    Args:
        model: a ``HFTransformerModel`` (anything exposing ``.layers``).
        ep_group: the EP process group. ``None`` (or a size-1 group is not
            special-cased -- pass ``None``) keeps all experts local and routes
            them with the reordering-only dispatcher; a multi-rank group
            shards the experts across it and routes with all-to-alls.

    Returns:
        The number of blocks swapped. Mixed sparse/dense models (e.g.
        Qwen3Moe's ``decoder_sparse_step``) swap only the sparse layers.

    Raises:
        TypeError: if no layer carries a recognizable HF MoE block. EP on a
            dense model is a config mistake, and silently swapping nothing
            would run it replicated.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"swap_hf_moe_blocks expects a model with .layers; got "
            f"{type(model).__name__}."
        )

    hf_config = getattr(getattr(model, "model", None), "config", None)
    aux_loss_coeff = getattr(hf_config, "router_aux_loss_coef", None)

    swapped = 0
    for layer in layers:
        block = getattr(layer, "mlp", None)
        # Dense layers of a mixed model fail the probe and are left alone.
        if block is None or not _is_hf_moe_block(block):
            continue
        layer.mlp = _convert_block(
            block, ep_group=ep_group, aux_loss_coeff=aux_loss_coeff
        )
        swapped += 1

    if swapped == 0:
        raise TypeError(
            f"no HF MoE block found on {type(model).__name__} "
            f"({type(getattr(model, 'model', model)).__name__}): no layer's "
            "``mlp`` has the router-gate + experts-ModuleList shape this swap "
            "recognizes. Add the model family's spelling to the probe in "
            "parallel/ep.py."
        )
    logger.info("Swapped %d HF MoE blocks for hpmesh MoE blocks", swapped)
    return swapped
