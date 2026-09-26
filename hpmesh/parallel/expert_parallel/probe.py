"""HF MoE block layout probes for the EP swap.

The entry point is ``swap.py`` (``swap_hf_moe_blocks``); the conversion is
``convert.py``. This module is the duck-typed detector every HF family is
identified through, shared by the swap and by ``tensor_parallel/apply.py``
(``_is_hf_moe_block``).

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
import torch.nn as nn

from ...models.common.moe import MOE_LAYER_ATTRS
from ...utils.logger_utils import get_logger
from .. import matrix

logger = get_logger(__name__)


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
        matrix.gpt_oss_layout(experts)
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
    return _has_router_weight(_router_of(module)) and _resolve_top_k(module) is not None


def _resolve_top_k(block: nn.Module) -> int | None:
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
    return _resolve_score_func(block, router) != "sigmoid"


def _resolve_score_func(block: nn.Module, router: nn.Module) -> str:
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
        matrix.group_limited_greedy()
    return num_groups, _read_int_attr(block, router, "topk_group")


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
