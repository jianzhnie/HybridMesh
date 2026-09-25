"""Swap a HF MoE block for hpmesh's EP-capable MoE, weights included.

Every sparse decoder layer of a transformers 5.x MoE model holds the same block
under ``layer.mlp``: a router, plus two stacked expert parameters. Expert
parallelism needs those weights in grouped-GEMM form with a token dispatcher in
front -- the stack in ``models/common/`` (``GroupedExperts`` /
``TokenChoiceTopKRouter`` / ``MoE``). This module is the bridge: it detects the
HF block, moves the weights into the hpmesh layout, and replaces ``layer.mlp``.
It also flags the layer for the FSDP MoE branch (``layer.moe_enabled = True``
plus a non-registered ``layer.moe`` alias, as upstream's swap does), without
which the expert weights would be sharded as dense parameters over the dense
DP mesh. The swap is in place and dtype-preserving, so it must run before FSDP
wraps the model and after the weights are loaded.

Both halves of the block vary by family, and the probe is duck-typed against
structures rather than class names, which are not stable across transformers
releases:

* **The router.** ``block.gate`` or ``block.router``, either a plain
  ``nn.Linear`` (DeepSeek-V2) or a bespoke module holding the same ``(E, D)``
  weight plus, for DeepSeek-V3/GLM4, an ``e_score_correction_bias`` buffer.
  Identified by its *weight tensor*, not its type (``_router_of``).
* **The expert weights.** Always ``gate_up_proj (E, 2F, D)`` and
  ``down_proj (E, D, F)``, split by ``chunk(2, dim=-1)`` after the gate+up
  GEMM. Identified by shape (``_fused_experts_of``).

Where a family keeps its routing attributes also moved in 5.x: ``top_k``,
``n_group``, ``topk_group``, ``norm_topk_prob``, ``scoring_func`` and
``routed_scaling_factor`` now sit on the *block* for every family, having been
split between block and router in 4.x. Every ``_read_*`` helper therefore takes
both owners and prefers whichever declares the field.

Weight layout: HF stores each projection as ``(out, in)`` and applies it as
``x @ W.T``; ``GroupedExperts`` stores the same orientation per expert
(``w1_EFD``/``w3_EFD`` are ``(F, D)``, ``w2_EDF`` is ``(D, F)``) and applies it
with the same ``F.linear``. Moving a weight is therefore elementwise -- no
transpose, no regrouping -- except for the one split of ``gate_up_proj`` into
its gate and up halves. ``tests/unit_tests/cpu/distributed/test_ep_swap.py``
pins both halves of that claim against the HF block being replaced.

Routing parity with the HF block: Qwen3Moe and Mixtral score with a softmax over
fp32 logits; DeepSeek-V3/GLM4 score with a sigmoid and apply
``routed_scaling_factor``; DeepSeek-V2 scores with a softmax but never
renormalizes, whatever its config declares (see ``_ignores_norm_topk_prob``).
``RouterGateLinear`` computes in fp32, so the same score function reproduces it,
and ``TokenChoiceTopKRouter`` takes node-limited routing as
``num_expert_groups``/``num_limited_groups``. The one deliberate addition is the
load-balance machinery HF never gets to run through this wrapper (the wrapper
never asks for router logits): the router carries a
``MicrobatchWiseLoadBalanceLoss`` with the config's ``router_aux_loss_coef``,
whose gradient is injected on backward (see ``models/common/aux_loss.py``).

Two families are deliberately *not* swapped, each because the HF block asks for
something the hpmesh stack does not express -- refusing beats approximating,
since the difference shows up only as different experts being chosen, which no
loss curve reveals:

* ``GPT-OSS``: per-expert bias vectors, a transposed ``(E, D, 2F)`` layout, and
  a hardcoded clamped sigmoid-GLU activation rather than a module
  (see ``_fused_experts_of``).
* ``DeepSeek-V2`` with ``topk_method="group_limited_greedy"``: scores a group by
  its single best expert where V3/GLM4 sum the top-2 (see
  ``_read_expert_groups``). Its default ``"greedy"`` is supported and exact.

``Qwen2Moe``'s ``shared_expert_gate`` multiplies where ``MoE.shared_experts``
only adds, and is refused at the point the shared expert is found.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn as nn

from ...accelerator import dist_utils
from ...models.common.grouped_experts import GroupedExperts
from ...models.common.moe import (
    MOE_LAYER_ATTRS,
    MicrobatchWiseLoadBalanceLoss,
    MoE,
    QuantileBalancedTopKRouter,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from ...models.common.token_dispatcher import (
    EP_DISPATCHER_BACKENDS,
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
    TorchAOTokenDispatcher,
)
from ...utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ["swap_hf_moe_blocks"]


def _router_of(block: nn.Module) -> nn.Module | None:
    """The block's router module, whatever the family calls it.

    ``gate`` is the common spelling and what every supported family but GPT-OSS
    uses. DeepSeek-V2's is a plain ``nn.Linear``; Qwen3Moe's, Mixtral's,
    DeepSeek-V3's and GLM4's are bespoke classes that differ only in the buffers
    they carry.
    """
    return getattr(block, "gate", None) or getattr(block, "router", None)


class _FusedExperts(NamedTuple):
    """One family's expert weights, as the three separate tensors hpmesh wants.

    A transformers 5.x expert keeps all E experts in two parameters rather than
    a list of per-expert MLPs: ``gate_up_proj (E, 2F, D)`` -- gate and up
    interleaved along the output dim, split by ``chunk(2, dim=-1)`` -- and
    ``down_proj (E, D, F)``. ``gate_EFD``/``up_EFD`` are views into the first,
    not copies.
    """

    gate_EFD: torch.Tensor
    up_EFD: torch.Tensor
    down_EDF: torch.Tensor
    num_experts: int


def _fused_experts_of(block: nn.Module) -> _FusedExperts | None:
    """Probe a MoE block for the fused expert tensors, or ``None``.

    One probe covers every supported family because transformers 5.x moved all
    of Qwen3Moe, OLMoE, Mixtral, DeepSeek-V2/V3 and GLM4 onto the same two
    parameters under the same forward shape::

        gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)
        out = linear(act(gate) * up, down_proj[e])

    so the split is by role -- the first half is the gate, the second the up
    projection -- for every one of them. That is asserted family by family
    against HF's own output in ``tests/unit_tests/cpu/distributed/test_ep_swap.py``.

    The probe reads *shapes*, not class names: the experts' class name is not
    stable across transformers versions (``Qwen3MoeExperts``,
    ``DeepseekV3NaiveMoe``, ``MixtralExperts``), but ``(E, 2F, D)`` is.

    Raises:
        NotImplementedError: for GPT-OSS, whose transposed layout
            (``gate_up_proj`` is ``(E, D, 2F)``) and per-expert bias vectors
            have no counterpart in hpmesh's ``GroupedExperts``. Copying it under
            the shared convention would put the wrong weight in the wrong slot.
    """
    experts = getattr(block, "experts", None)
    if experts is None:
        return None
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor):
        return None
    if gate_up.dim() != 3 or down.dim() != 3:
        return None
    if gate_up.shape[0] != down.shape[0]:
        return None
    if hasattr(experts, "gate_up_proj_bias") or hasattr(experts, "down_proj_bias"):
        raise NotImplementedError(
            f"{type(experts).__name__} carries per-expert bias vectors, which "
            "hpmesh's GroupedExperts has no slot for. Only GPT-OSS has them, "
            "and it differs further: its gate_up_proj is transposed to "
            "(E, D, 2F) and its activation is a hardcoded clamped sigmoid-GLU "
            "rather than a module. Support needs a bias-bearing expert module "
            "with its own activation seam, not a wider copy here."
        )
    num_experts, double_hidden, dim = gate_up.shape
    # down_proj is (E, D, F) with the *same* D: the token dim must agree on both
    # sides. Its trailing dim is F, which equals ``double_hidden / 2`` only for
    # architectures whose expert hidden size matches the dense MLP's -- Mixtral
    # and OLMoE, but not Qwen3Moe or DeepSeek -- so it is deliberately not
    # checked against ``double_hidden``.
    if double_hidden % 2 != 0 or down.shape[1] != dim:
        raise ValueError(
            f"unrecognized expert weight shapes: gate_up_proj "
            f"{tuple(gate_up.shape)}, down_proj {tuple(down.shape)}. Expected "
            "(E, 2F, D) and (E, D, F), the layout shared by every supported "
            "family in transformers 5.x."
        )
    hidden = double_hidden // 2
    return _FusedExperts(
        gate_EFD=gate_up[:, :hidden],
        up_EFD=gate_up[:, hidden:],
        down_EDF=down,
        num_experts=num_experts,
    )


def _has_router_weight(router: nn.Module | None) -> bool:
    """Whether a router carries its own ``(E, D)`` gate weight.

    DeepSeek-V3 and GLM4 routers are bespoke classes rather than ``nn.Linear``,
    so the shape is what identifies them -- the class name is not stable across
    transformers versions.
    """
    weight = getattr(router, "weight", None)
    return isinstance(weight, nn.Parameter) and weight.dim() == 2


def _is_hf_moe_block(module: nn.Module) -> bool:
    """Structural probe for the HF sparse-MoE block shape.

    Deliberately duck-typed rather than an ``isinstance`` against one family's
    class: Qwen3Moe, OLMoE, Mixtral, DeepSeek-V2/V3 and GLM4 share this shape
    (a router with an ``(E, D)`` weight, plus fused expert tensors) under
    different class names and different router spellings.

    The router is identified by its *weight tensor*, not by being ``nn.Linear``:
    DeepSeek-V3's and GLM4's routers are plain ``nn.Module``s holding the same
    ``(E, D)`` parameter, while DeepSeek-V2's really is an ``nn.Linear``.

    ``top_k`` is read from the block or the router because the families disagree
    on where it lives -- DeepSeek and Mixtral declare it on the block, Qwen3Moe
    and OLMoE on the router.
    """
    try:
        experts = _fused_experts_of(module)
    except NotImplementedError:
        # A block this swap refuses to convert is still a MoE, and the caller
        # must see the refusal rather than a generic "not a MoE block" skip.
        raise
    if experts is None:
        return False
    return _has_router_weight(_router_of(module)) and _read_top_k(module) is not None


def _read_top_k(block: nn.Module) -> int | None:
    """Top-K per token, from the block or its router."""
    for owner in (block, _router_of(block)):
        if owner is None:
            continue
        for attr in ("top_k", "num_experts_per_tok"):
            value = getattr(owner, attr, None)
            if isinstance(value, int):
                return value
    return None


def _ignores_norm_topk_prob(block: nn.Module, router: nn.Module) -> bool:
    """Whether the block declares ``norm_topk_prob`` without ever applying it.

    Only DeepSeek-V2 does. Its routing ignores the field entirely::

        router_logits = router_logits.softmax(dim=-1, dtype=torch.float32)
        ...
        topk_weight = topk_weight * self.routed_scaling_factor

    that is, the selected scores are scaled but never divided by their own sum.
    A model with ``norm_topk_prob=True`` in its config therefore still behaves
    as ``route_norm=False``.

    The tell is ``topk_method``, not the class name (which transformers renames
    between versions) and not ``norm_topk_prob`` itself -- transformers 5.x
    *removed* that field from DeepSeek-V2 while keeping the behaviour, so a
    guard that read it would find nothing, fall through to the ``softmax ->
    renormalize`` default, and silently route differently from HF.
    """
    if getattr(block, "topk_method", None) is not None:
        return True
    return getattr(router, "topk_method", None) is not None


def _read_route_norm(block: nn.Module, router: nn.Module) -> bool:
    """Whether the selected K scores are renormalized to sum to 1.

    ``norm_topk_prob`` may live on either the block or the router. When neither
    declares it, sigmoid routing scores are used as-is and anything else is taken
    to be normalized -- matching how torchtitan's probe resolves it.
    """
    if _ignores_norm_topk_prob(block, router):
        return False
    for owner in (block, router):
        value = getattr(owner, "norm_topk_prob", None)
        if value is not None:
            return bool(value)
    return _read_score_func(block, router) != "sigmoid"


def _read_score_func(block: nn.Module, router: nn.Module) -> str:
    """The router's scoring function.

    ``e_score_correction_bias`` is the reliable DeepSeek-V3/GLM4 marker: a router
    that carries that buffer scores with sigmoid. Otherwise the block's declared
    ``scoring_func`` wins, defaulting to softmax as HF does.
    """
    buffers = getattr(router, "_buffers", {})
    if "e_score_correction_bias" in buffers:
        return "sigmoid"
    declared = getattr(block, "scoring_func", None) or getattr(
        router, "scoring_func", None
    )
    if declared:
        return str(declared).lower()
    return "softmax"


def _read_int_attr(block: nn.Module, router: nn.Module, name: str) -> int | None:
    """Read an optional integer attribute off the block or the router."""
    for owner in (block, router):
        value = getattr(owner, name, None)
        if value is not None:
            return int(value)
    return None


def _read_route_scale(block: nn.Module, router: nn.Module) -> float:
    """The multiplier applied to the selected K scores after normalization.

    HF puts ``routed_scaling_factor`` on whichever object owns the rest of the
    routing: the MoE block for Qwen3Moe, the router for DeepSeek-V3/GLM4. Both
    are checked so the same probe serves either layout.
    """
    for owner in (block, router):
        value = getattr(owner, "routed_scaling_factor", None)
        if value is not None:
            return float(value)
    return 1.0


def _moe_block_of(layer: nn.Module) -> tuple[str, nn.Module | None]:
    """The layer's MoE block and the attribute it is held under.

    Every supported family keeps it on ``mlp``. The attribute name is returned
    alongside the block because the swap has to *replace* the attribute, so the
    replacement must land in the same slot.

    ``MOE_LAYER_ATTRS`` is shared with the expert-bias hook in
    ``models/common/moe.py`` -- that hook rediscovers these blocks after the
    swap, so the two must agree on where a MoE can live.
    """
    for name in MOE_LAYER_ATTRS:
        block = getattr(layer, name, None)
        if block is not None:
            return name, block
    return MOE_LAYER_ATTRS[0], None


def _read_expert_groups(
    block: nn.Module, router: nn.Module
) -> tuple[int | None, int | None]:
    """The node-limited-routing config, as ``(num_expert_groups, num_limited_groups)``.

    Returns ``(None, None)`` when the block does no group-limited routing, which
    is the case for Qwen3Moe, OLMoE and Mixtral.

    The grouping attributes are read only when the model is actually routing by
    group. DeepSeek-V2 carries ``num_group``/``topk_group`` but consults them
    only under ``topk_method="group_limited_greedy"``; its default ``"greedy"``
    routes freely over all experts. Reading the attributes without checking the
    method feeds those tokens into a group restriction HF never applies.

    Raises:
        NotImplementedError: for DeepSeek-V2's ``group_limited_greedy``, whose
            rule differs from the one implemented here: it scores a group by its
            single best expert (``max``) where DeepSeek-V3/GLM4 sum the group's
            top-2. Both are "group-limited", so the distinction is invisible in
            the config and would only show up as different experts being chosen.
    """
    # ``n_group`` is the DeepSeek-V3/GLM4 spelling, ``num_group`` DeepSeek-V2's.
    # Both are probed: reading only one silently disables grouping for the
    # family that spells it the other way, which is a routing change HF does not
    # make and no error surfaces.
    num_groups = _read_int_attr(block, router, "n_group")
    if num_groups is None:
        num_groups = _read_int_attr(block, router, "num_group")
    if num_groups is None:
        return None, None

    # ``topk_method`` sits on the block in transformers 5.x (it was on
    # DeepSeek-V2's router in 4.x), so both are read.
    topk_method = getattr(block, "topk_method", None) or getattr(
        router, "topk_method", None
    )
    if topk_method == "greedy":
        return None, None
    if _ignores_norm_topk_prob(block, router):
        raise NotImplementedError(
            "DeepSeek-V2's group_limited_greedy scores a group by its single "
            "best expert (max); the implemented rule sums the group's top-2 "
            "(DeepSeek-V3/GLM4). Routing this checkpoint with that rule picks "
            "different experts, so refusing is the point -- add a group-scoring "
            "option to TokenChoiceTopKRouter to support it."
        )
    return num_groups, _read_int_attr(block, router, "topk_group")


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
        raise NotImplementedError(
            f"{type(router_gate).__name__} carries a router bias, which "
            "RouterGateLinear has no slot for. Every supported family "
            "(Qwen3Moe, OLMoE, Mixtral, DeepSeek-V2/V3, GLM4) is bias-free, "
            "so this fires only on a family the probe does not know."
        )
    experts = _fused_experts_of(block)
    assert experts is not None  # the probe established this

    num_experts = experts.num_experts
    top_k = _read_top_k(block)
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
    score_func = _read_score_func(block, router_gate)
    if quantile_balancing:
        # The quantile scheme is defined over sigmoid scores (the histogram
        # range derives from their [0, 1] bound) and routes freely over all
        # experts, so a softmax family or a group-limited one cannot adopt it.
        if score_func != "sigmoid":
            raise NotImplementedError(
                f"quantile-balanced routing requires sigmoid router scores, "
                f"got {score_func!r} for {type(block).__name__}."
            )
        if num_expert_groups is not None and num_expert_groups > 1:
            raise NotImplementedError(
                f"quantile-balanced routing selects a free Top-(K+1) over all "
                f"experts; {type(block).__name__}'s group-limited routing is "
                "incompatible with it. (A single group is no restriction and "
                "is accepted.)"
            )
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
        raise NotImplementedError(
            f"{type(block).__name__} gates its shared expert "
            "(shared_expert_gate); MoE's shared_experts is additive only. "
            "Qwen3Moe has no shared expert, so this is unreachable there."
        )
    if shared is not None and tp_enabled:
        raise NotImplementedError(
            f"tp x ep over {type(block).__name__}: the block has a shared "
            "expert, which the TP plan shards with the dense colwise/rowwise "
            "realizers. Composing those with the swapped MoE's sequence-"
            "sharded dispatch layout is unverified; run shared-expert models "
            "with tp=1 (EP handles the shared expert) or ep=1."
        )

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


def _probe_expert_layouts() -> None:
    """Development-time survey of the expert layouts transformers 5.9 ships.

    Not called at import: it builds seven real models, which is far too much
    work to pay for on every import of a training package. Run it by hand
    against a new transformers minor.

    Kept because the shapes are the whole premise of ``_fused_experts_of``:
    when an upgrade breaks the swap, this says immediately whether the layout
    moved or the probe broke.

    Measured on transformers 5.9.0 with the uniform tiny config above::

        qwen3_moe    gate_up_proj (8, 96, 64)   down_proj (8, 64, 48)    silu
        olmoe        gate_up_proj (8, 256, 64)  down_proj (8, 64, 128)   silu
        mixtral      gate_up_proj (8, 256, 64)  down_proj (8, 64, 128)   silu
        gpt_oss      gate_up_proj (8, 64, 256)  down_proj (8, 128, 64)   clamped glu

    Mixtral and OLMoE have ``2F == D``, which is why ``_fused_experts_of``
    cannot validate the gate/up halves against each other and checks the token
    dim instead.

    GPT-OSS is unlike the others twice over, and both are why it is refused:
    ``gate_up_proj`` is transposed (``(E, D, 2F)``, splitting on the token dim
    rather than the output dim), and its activation is not a module at all but a
    hardcoded clamped sigmoid-GLU (``gate.clamp(max=7) * sigmoid(1.702 * gate)``
    with a clamped ``up``), plus a bias vector per projection. DeepSeek-V2/V3 and
    GLM4 do not appear here: their layer 0 is the dense ``first_k_dense_replace``
    MLP, and their later layers are reached through the swap tests instead.

    Uses ``sys.modules`` rather than importing transformers at the top of this
    module, which would make the package unimportable without it.
    """
    transformers = sys.modules.get("transformers")
    if transformers is None:
        raise RuntimeError(
            "_probe_expert_layouts needs transformers imported first; it is a "
            "development tool, not part of the training path."
        )

    # A uniform tiny config: several families build a 4000-expert model from
    # their own defaults, and the point here is the shape convention, not size.
    common = dict(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        num_local_experts=8,
        n_routed_experts=8,
        n_shared_experts=1,
        decoder_sparse_step=1,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        max_position_embeddings=128,
    )
    for family in ("qwen3_moe", "olmoe", "mixtral", "gpt_oss"):
        try:
            config = transformers.AutoConfig.for_model(family, **common)
            layers = transformers.AutoModelForCausalLM.from_config(
                config, experts_implementation="eager"
            ).model.layers
            # The last layer, not the first: DeepSeek and GLM4 put a dense MLP
            # at index 0 (first_k_dense_replace), so layer 0 carries no experts.
            experts = layers[-1].mlp.experts
            gate_up = getattr(experts, "gate_up_proj", None)
            down = getattr(experts, "down_proj", None)
            # GPT-OSS spells it ``_act_fn`` and has no ``act_fn``.
            act = getattr(experts, "act_fn", None) or getattr(experts, "_act_fn", None)
            logger.info(
                "%s: gate_up_proj %s, down_proj %s, act %s",
                family,
                "?" if gate_up is None else tuple(gate_up.shape),
                "?" if down is None else tuple(down.shape),
                type(act).__name__,
            )
        except Exception as err:  # noqa: BLE001 - a survey, not a gate
            logger.info("%s: unavailable (%s)", family, err)


def swap_hf_moe_blocks(
    model: nn.Module,
    *,
    ep_group=None,
    router_aux_loss_coef: float | None = None,
    quantile_balancing: bool = False,
    token_dispatcher: str = "alltoall",
    torchao_pad_multiple: int = 16,
    tp_enabled: bool = False,
) -> int:
    """Replace every HF MoE block in ``model`` with hpmesh's MoE, in place.

    Args:
        model: a ``HFTransformerModel`` (anything exposing ``.layers``).
        ep_group: the EP process group. ``None`` (or a size-1 group is not
            special-cased -- pass ``None``) keeps all experts local and routes
            them with the reordering-only dispatcher; a multi-rank group
            shards the experts across it and routes with all-to-alls.
        router_aux_loss_coef: coefficient for the per-forward load-balance
            loss. ``None`` takes the HF config's ``router_aux_loss_coef``
            (Qwen3Moe has one; DeepSeek-V3's config has no such field and gets
            no loss). A float here overrides the config for every MoE layer.
        quantile_balancing: replace the sign-based load-balancing bias with
            quantile-balanced routing (``QuantileBalancedTopKRouter``): the
            bias is then re-solved from a required-bias histogram once per
            optimizer step instead of nudged by the sign rule, and
            ``load_balance_coeff`` is forced off. Requires sigmoid router
            scores and no group-limited routing.
        token_dispatcher: EP dispatch backend (``"alltoall"`` default,
            ``"torchao"`` optional-import adapter, ``"deepep"``/``"hybridep"``
            registered gaps refused here). See
            ``ParallelConfig.ep_token_dispatcher``.
        torchao_pad_multiple: padding multiple for the ``"torchao"`` backend.
        tp_enabled: the model is also tensor-parallelized (tp x ep). Used only
            for fail-fast validation of unverified combinations (currently a
            shared expert): the swap itself is layout-identical either way --
            the swapped block consumes and produces the T/tp sequence shard
            directly, which is the layout upstream's ep+sp MoE uses.

    Returns:
        The number of blocks swapped. Mixed sparse/dense models (e.g.
        Qwen3Moe's ``decoder_sparse_step``) swap only the sparse layers.

    Raises:
        TypeError: if no layer carries a recognizable HF MoE block. EP on a
            dense model is a config mistake, and silently swapping nothing
            would run it replicated.
        NotImplementedError: if a MoE block uses a layout this swap does not
            implement -- GPT-OSS's transposed, bias-bearing experts (see
            ``_fused_experts_of``) or DeepSeek-V2's ``group_limited_greedy``
            routing (see ``_read_expert_groups``).
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"swap_hf_moe_blocks expects a model with .layers; got "
            f"{type(model).__name__}."
        )
    # Backend gating, before any probing: ParallelConfig.__post_init__ is the
    # primary gate; this is the defensive copy for callers that reach the swap
    # directly.
    if token_dispatcher not in EP_DISPATCHER_BACKENDS:
        raise ValueError(
            f"unknown ep_token_dispatcher {token_dispatcher!r}; expected one "
            f"of {EP_DISPATCHER_BACKENDS}."
        )
    if token_dispatcher in ("deepep", "hybridep"):
        raise NotImplementedError(
            f"ep_token_dispatcher={token_dispatcher!r} is a registered gap: "
            "CUDA-only kernels plus torchtitan's distributed/deepep/ wrappers "
            "that hpmesh does not vendor. Use 'alltoall' meanwhile."
        )

    hf_config = getattr(getattr(model, "model", None), "config", None)
    aux_loss_coeff = (
        router_aux_loss_coef
        if router_aux_loss_coef is not None
        else getattr(hf_config, "router_aux_loss_coef", None)
    )
    # The load-balance coefficient belongs to a *native* hpmesh run; an HF
    # checkpoint has no equivalent field. It is therefore derived from the
    # probed bias: a block that carries ``e_score_correction_bias`` is one the
    # model was trained with the auxiliary-loss-free scheme on (DeepSeek-V3,
    # GLM4), and gets the hpmesh default. Everything else stays off, so a
    # frozen zero bias never lands in the checkpoint.
    default_coeff = getattr(hf_config, "load_balance_coeff", 1e-3)

    swapped = 0
    for layer in layers:
        attr, block = _moe_block_of(layer)
        # Dense layers of a mixed model (Qwen3Moe's decoder_sparse_step, and the
        # first_k_dense_replace of the DeepSeek/GLM4 families) are left alone.
        if block is None or not _is_hf_moe_block(block):
            continue
        router_gate = _router_of(block)
        load_balance_coeff = (
            float(default_coeff)
            if getattr(router_gate, "e_score_correction_bias", None) is not None
            else None
        )
        moe = _convert_block(
            block,
            ep_group=ep_group,
            aux_loss_coeff=aux_loss_coeff,
            load_balance_coeff=load_balance_coeff,
            quantile_balancing=quantile_balancing,
            token_dispatcher=token_dispatcher,
            torchao_pad_multiple=torchao_pad_multiple,
            tp_enabled=tp_enabled,
        )
        # Replace in the slot the block was actually found in. Writing to a
        # different attribute would leave the original block in place and route
        # around the swap entirely.
        setattr(layer, attr, moe)
        # FSDP's MoE branch (fully_shard/fsdp.py) keys off these two
        # attributes, as does upstream's swap (moe_replacement.py):
        # ``moe_enabled`` marks the block as sparse, and ``moe`` is where the
        # branch reads the expert weights from. Without them the experts are
        # sharded as dense parameters over the dense DP mesh, mixing ranks of
        # different EP coordinates into one FSDP group. ``object.__setattr__``
        # keeps ``moe`` out of the module registry: it is the same module as
        # the block attribute, and registering it would double every expert
        # weight in the state_dict.
        layer.moe_enabled = True
        object.__setattr__(layer, "moe", moe)
        swapped += 1

    if swapped == 0:
        raise TypeError(
            f"no HF MoE block found on {type(model).__name__} "
            f"({type(getattr(model, 'model', model)).__name__}): no layer's "
            "``mlp`` has the router-gate + fused gate_up_proj/down_proj shape "
            "this swap recognizes. Add the model family's layout to the probe "
            "in parallel/expert_parallel/swap.py."
        )
    logger.info("Swapped %d HF MoE blocks for hpmesh MoE blocks", swapped)
    return swapped
