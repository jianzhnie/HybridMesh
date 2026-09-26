"""The EP swap's block conversion: HF MoE block -> hpmesh MoE.

Split out of ``swap.py``: ``_convert_block`` builds the hpmesh MoE for one
probed HF block and moves its weights over; ``_restore_fp32_state_buffers``
undoes the buffer dtype cast ``Module.to`` applies. The layout probes live in
``probe.py``, the orchestration in ``swap.py``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from ...accelerator import dist_utils
from ...models.common.grouped_experts import GroupedExperts
from ...models.common.moe import (
    MicrobatchWiseLoadBalanceLoss,
    MoE,
    QuantileBalancedTopKRouter,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from ...models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
    TorchAOTokenDispatcher,
)
from .. import matrix
from .probe import (
    _fused_experts_of,
    _read_expert_groups,
    _read_route_norm,
    _read_route_scale,
    _resolve_score_func,
    _resolve_top_k,
    _router_of,
)


def _restore_fp32_state_buffers(module: nn.Module) -> None:
    """Undo the dtype conversion ``Module.to(dtype=...)`` applies to float buffers.

    ``Module.to`` converts parameters *and* every floating-point buffer, with no
    way to ask for one and not the other. The MoE registers two buffers as fp32
    on purpose -- ``expert_bias_E`` (an additive load-balancing correction that
    would erode under a bf16 round per step) and ``tokens_per_expert_E`` (a
    token *count*, which bf16 cannot hold exactly past 256: 1001 becomes 1000).
    Neither has a gradient, so casting them buys nothing and silently costs
    precision. Upstream keeps both fp32 unconditionally.

    Scoped to float buffers: an integer buffer's dtype is already exact and
    casting it back would be wrong.
    """
    for buffer in module.buffers():
        if torch.is_floating_point(buffer):
            buffer.data = buffer.data.to(torch.float32)


def _convert_block(
    block: nn.Module,
    *,
    ep_group: dist.ProcessGroup | None,
    aux_loss_coeff: float | None,
    load_balance_coeff: float | None,
    quantile_balancing: bool,
    token_dispatcher: str,
    torchao_pad_multiple: int,
    tp_enabled: bool = False,
) -> MoE:
    """Build the hpmesh MoE for one HF block and move its weights over."""
    ep_size = 1 if ep_group is None else dist_utils.get_world_size(ep_group)
    ep_rank = 0 if ep_group is None else dist_utils.get_rank(ep_group)

    router_gate = _router_of(block)
    assert router_gate is not None  # the probe established this
    if getattr(router_gate, "bias", None) is not None:
        matrix.router_bias(router_gate)
    experts = _fused_experts_of(block)
    assert experts is not None  # the probe established this

    num_experts = experts.num_experts
    top_k = _resolve_top_k(block)
    assert top_k is not None  # the probe established this
    dim = router_gate.weight.shape[1]
    hidden = experts.gate_EFD.shape[1]
    if num_experts % ep_size != 0:
        raise ValueError(
            f"EP degree {ep_size} does not divide num_experts={num_experts}; "
            "each EP rank must hold the same number of experts."
        )
    num_local = num_experts // ep_size
    lo = ep_rank * num_local

    num_expert_groups, num_limited_groups = _read_expert_groups(block, router_gate)
    grouped = GroupedExperts(dim, hidden, num_local)
    score_func = _resolve_score_func(block, router_gate)
    if quantile_balancing:
        # The quantile scheme is defined over sigmoid scores (the histogram
        # range derives from their [0, 1] bound) and routes freely over all
        # experts, so a softmax family or a group-limited one cannot adopt it.
        if score_func != "sigmoid":
            matrix.quantile_requires_sigmoid(score_func, block)
        if num_expert_groups is not None and num_expert_groups > 1:
            matrix.quantile_no_group_limit(block)
        router = QuantileBalancedTopKRouter(
            num_experts,
            dim,
            top_k,
            route_norm=_read_route_norm(block, router_gate),
            route_scale=_read_route_scale(block, router_gate),
            aux_loss=(
                MicrobatchWiseLoadBalanceLoss(coeff=aux_loss_coeff)
                if aux_loss_coeff
                else None
            ),
        )
        # The quantile update owns expert_bias_E; the sign-based update is
        # off (MoE registers the buffer for a quantile router regardless).
        load_balance_coeff = None
    else:
        router = TokenChoiceTopKRouter(
            num_experts,
            dim,
            top_k,
            score_func=score_func,
            route_norm=_read_route_norm(block, router_gate),
            route_scale=_read_route_scale(block, router_gate),
            num_expert_groups=num_expert_groups,
            num_limited_groups=num_limited_groups,
            aux_loss=(
                MicrobatchWiseLoadBalanceLoss(coeff=aux_loss_coeff)
                if aux_loss_coeff
                else None
            ),
        )
    if token_dispatcher == "torchao":
        # Optional-import adapter: the constructor raises ImportError with an
        # install hint when torchao is absent. EP=1 is supported by the
        # dispatcher itself (local padded permute only).
        dispatcher = TorchAOTokenDispatcher(num_experts, top_k, torchao_pad_multiple)
    elif ep_group is None:
        dispatcher = LocalTokenDispatcher(num_experts, top_k)
    else:
        dispatcher = AllToAllTokenDispatcher(num_experts, top_k)
    if ep_group is not None:
        dispatcher.wire_meshes(ep_group=ep_group)

    shared = getattr(block, "shared_expert", None) or getattr(
        block, "shared_experts", None
    )
    if shared is not None and hasattr(block, "shared_expert_gate"):
        matrix.shared_expert_gate(block)
    if shared is not None and tp_enabled:
        matrix.shared_expert_tp_ep(block)

    moe = MoE(
        num_experts=num_experts,
        routed_experts=RoutedExperts(grouped, dispatcher),
        router=router,
        # Auxiliary-loss-free load balancing, mirroring the HF block's own
        # e_score_correction_bias. Off for models that carry no such buffer --
        # a frozen zero bias is dead state in the checkpoint.
        load_balance_coeff=load_balance_coeff,
        shared_experts=shared,
    )

    # Match the block's dtype/device before the copies so they are exact, and
    # keep its mode: a fresh module defaults to training=True, which would
    # flip eval-built models (aux-loss injection, token counting) back on.
    moe.to(dtype=router_gate.weight.dtype, device=router_gate.weight.device)
    _restore_fp32_state_buffers(moe)
    moe.train(block.training)
    with torch.no_grad():
        router.gate.weight.copy_(router_gate.weight)
        # HF keeps all E experts in two stacked parameters; each rank keeps its
        # own slice of them. ``gate_EFD``/``up_EFD`` are the two halves of
        # ``gate_up_proj`` split along its output dim -- see ``_fused_experts_of``.
        lo = ep_rank * num_local
        grouped.w1_EFD.copy_(experts.gate_EFD[lo : lo + num_local])
        grouped.w3_EFD.copy_(experts.up_EFD[lo : lo + num_local])
        grouped.w2_EDF.copy_(experts.down_EDF[lo : lo + num_local])
        # The load-balancing bias is optimization state, not a learned weight:
        # it is copied so a resumed or swapped run keeps the balance the HF
        # model had reached.
        bias = getattr(router_gate, "e_score_correction_bias", None)
        if bias is not None and moe.expert_bias_E is not None:
            moe.expert_bias_E.copy_(bias.float())
    return moe
