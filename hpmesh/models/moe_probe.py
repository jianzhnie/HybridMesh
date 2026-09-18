"""Probe a HuggingFace MoE block and describe its architecture.

The problem this solves: HF ships a different MoE layout per model family.
``Qwen3MoeSparseMoeBlock`` keeps its router in ``gate``, ``Mixtral`` packs experts
into one fused ``gate_up_proj`` tensor, DeepSeek-V3 adds group-limited routing and
a sigmoid scoring function, Qwen3.5 adds a sigmoid-gated shared expert, and Gemma4
puts the router and experts directly on the decoder layer instead of inside
``mlp``. Downstream code that wants to build its own MoE should not have to
re-derive all of that.

``MoEArchitecture`` is that derivation's output: a flat, framework-agnostic
description of one layer's MoE. Nothing here constructs a module, allocates a
tensor, or knows what will consume the result.

Scope: this is the *probing* half of torchtitan's ``moe_replacement.py``. That
file's other half built torchtitan's own MoE and swapped it in, which would drag
in its entire MoE stack; hpmesh describes the architecture and leaves building to
whoever wants it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

__all__ = [
    "MoEArchitecture",
    "SharedExpertArchitecture",
    "detect_moe_layers",
    "probe_moe_layer",
    "probe_moe_model",
]

# Where a layer keeps its MoE. ``block`` means router+experts live inside a
# submodule; ``layer_level`` means they are siblings of the dense MLP on the
# decoder layer itself.
MoELayerKind = Literal["none", "block", "layer_level"]

# HF hangs the MoE block off a different attribute per family: Mixtral uses
# ``block_sparse_moe``, most others use ``mlp``. Probing a list rather than
# hardcoding ``mlp`` is the point of this module -- a miss here would report "no
# MoE" for a model that has one, silently.
_MOE_BLOCK_ATTRS = ("mlp", "block_sparse_moe", "feed_forward", "moe")


@dataclass(frozen=True)
class SharedExpertArchitecture:
    """The always-on expert that runs alongside the routed ones.

    Args:
        hidden_dim: The shared expert's intermediate (inner) dimension.
        dim: The model's hidden dimension.
        has_sigmoid_gate: Whether a learned sigmoid gate scales the shared
            expert's output before it is added (the Qwen3.5 pattern).
    """

    hidden_dim: int
    dim: int
    has_sigmoid_gate: bool = False


@dataclass(frozen=True)
class MoEArchitecture:
    """A normalized description of one MoE layer.

    Args:
        num_experts: Total routed experts.
        dim: The model's hidden dimension.
        moe_intermediate_size: Each expert's intermediate (inner) dimension.
        top_k: How many experts each token routes to.
        score_func: Router scoring function, ``"softmax"`` or ``"sigmoid"``.
        route_norm: Whether the top-k router weights are renormalized to sum to 1.
        route_scale: Multiplier applied to the routing weights.
        num_expert_groups: Group-limited routing: how many groups the experts are
            partitioned into. ``None`` when the model routes over all experts.
        num_limited_groups: Group-limited routing: how many groups a token may
            select experts from. ``None`` when routing is ungrouped.
        load_balance_coeff: Auxiliary load-balancing loss coefficient.
        comm_backend: The token-dispatch backend the model's config asked for.
            Recorded as a hint; this module does not act on it.
        shared_expert: The shared expert, or ``None`` if the model has none.
    """

    num_experts: int
    dim: int
    moe_intermediate_size: int
    top_k: int
    score_func: str
    route_norm: bool
    route_scale: float
    num_expert_groups: int | None
    num_limited_groups: int | None
    load_balance_coeff: float
    comm_backend: str
    shared_expert: SharedExpertArchitecture | None


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _iter_layers(model: nn.Module):
    """Yield the model's decoder layers, whichever container holds them.

    Accepts either a decoder wrapper (which exposes ``layers`` directly) or a bare
    ``ForCausalLM`` (where the decoder is one level in). Layers live in a
    ``ModuleList`` on most models, but some wrappers swap in a ``ModuleDict`` to
    preserve string-indexed state-dict keys -- hence the ``values()`` branch.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise AttributeError(
            f"{type(model).__name__} exposes no 'layers'. Pass the decoder "
            "wrapper or its ForCausalLM, not the bare model config."
        )
    return layers.values() if hasattr(layers, "values") else list(layers)


def detect_moe_layers(model: nn.Module) -> None:
    """Mark which decoder layers carry MoE, and in which layout.

    Sets ``layer.moe_enabled`` and, for the sibling layout,
    ``layer._layer_level_moe``. This is the signal ``hf_sharding`` reads when it
    skips a Titan MoE subtree, so it must be set before sharding configs.

    Detection is structural rather than config-driven: a layer is MoE if one of
    its submodules (or the layer itself) owns both a router and an experts
    module. Router names vary -- ``gate`` or ``router`` -- so both are probed.
    """
    for layer in _iter_layers(model):
        block = _find_moe_block(layer)
        if block is not None and _has_router(block) and hasattr(block, "experts"):
            layer.moe_enabled = True
            continue

        if _has_router(layer) and hasattr(layer, "experts"):
            layer.moe_enabled = True
            layer._layer_level_moe = True
            continue

        layer.moe_enabled = False


def probe_moe_layer(
    layer: nn.Module, config: PretrainedConfig
) -> MoEArchitecture | None:
    """Describe one decoder layer's MoE, or ``None`` if it has none.

    Expects ``detect_moe_layers`` to have run: the ``moe_enabled`` flag it sets
    is what distinguishes an MoE layer here.
    """
    if not getattr(layer, "moe_enabled", False):
        return None

    if getattr(layer, "_layer_level_moe", False):
        params = _probe_layer_level_moe(layer, config)
    else:
        params = _probe_hf_moe_block(_find_moe_block(layer), config)
    return MoEArchitecture(**params)


def probe_moe_model(
    model: nn.Module, config: PretrainedConfig
) -> dict[int, MoEArchitecture]:
    """Describe every MoE layer in ``model``, keyed by layer index.

    Returns an empty dict for a dense model, so callers can treat "no MoE" as the
    same code path as "MoE somewhere".
    """
    detected: dict[int, MoEArchitecture] = {}
    for index, layer in enumerate(_iter_layers(model)):
        arch = probe_moe_layer(layer, config)
        if arch is not None:
            detected[index] = arch

    if detected:
        logger.info(f"Probed MoE architecture for {len(detected)} layer(s)")
    return detected


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def _has_router(module: nn.Module) -> bool:
    """True if ``module`` owns a router under any of its known names."""
    return hasattr(module, "gate") or hasattr(module, "router")


def _find_moe_block(layer: nn.Module) -> nn.Module | None:
    """Return the submodule holding this layer's MoE block, or ``None``.

    Probes ``_MOE_BLOCK_ATTRS`` rather than a fixed ``mlp`` so families that name
    it differently (Mixtral) are still found.
    """
    for attr in _MOE_BLOCK_ATTRS:
        block = getattr(layer, attr, None)
        if block is not None and _has_router(block) and hasattr(block, "experts"):
            return block
    return None


def _probe_hf_moe_block(moe_block: nn.Module, config: PretrainedConfig) -> dict:
    """Extract MoE configuration from an HF MoE block.

    Args:
        moe_block: The HF MoE block (e.g. ``Qwen3MoeSparseMoeBlock``).
        config: The HF model config carrying MoE-related attributes.

    Returns:
        Kwargs for :class:`MoEArchitecture`.
    """
    gate = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)
    experts = moe_block.experts

    num_experts = _resolve_num_experts(experts, gate, moe_block, config)
    dim = config.hidden_size

    # Intermediate size: HF MoE models use fused gate_up_proj with the
    # standard (E, 2*I, H) layout, so dim 1 is 2*I.
    if hasattr(experts, "gate_up_proj"):
        moe_intermediate_size = experts.gate_up_proj.shape[1] // 2
    elif hasattr(config, "moe_intermediate_size") and config.moe_intermediate_size:
        moe_intermediate_size = config.moe_intermediate_size
    else:
        moe_intermediate_size = getattr(config, "intermediate_size", dim * 4)

    return {
        "num_experts": num_experts,
        "dim": dim,
        "moe_intermediate_size": moe_intermediate_size,
        "top_k": _resolve_top_k(moe_block, gate, config),
        "score_func": _resolve_score_func(gate, config),
        "route_norm": _resolve_route_norm(gate, config),
        "route_scale": getattr(config, "routed_scaling_factor", 1.0),
        "num_expert_groups": getattr(config, "n_group", None),
        "num_limited_groups": getattr(config, "topk_group", None),
        "load_balance_coeff": getattr(config, "load_balance_coeff", 1e-3),
        "comm_backend": getattr(config, "comm_backend", "standard"),
        "shared_expert": _probe_shared_experts(moe_block, config),
    }


def _probe_layer_level_moe(layer: nn.Module, config: PretrainedConfig) -> dict:
    """Probe layer-level MoE, where router/experts are siblings of the dense MLP.

    The dense MLP acts as the shared expert: the original HF forward sums its
    output with the routed experts' output.
    """
    gate = getattr(layer, "gate", None) or getattr(layer, "router", None)
    experts = layer.experts

    num_experts = _resolve_num_experts(experts, gate, layer, config)
    dim = config.hidden_size

    # Expert intermediate size from the fused gate_up_proj. Which axis holds the
    # hidden dim varies by checkpoint, so disambiguate by shape rather than
    # assuming a layout.
    if hasattr(experts, "gate_up_proj"):
        shape = experts.gate_up_proj.shape
        if shape[2] == dim and shape[1] != dim:
            moe_intermediate_size = shape[1] // 2
        elif shape[1] == dim and shape[2] != dim:
            moe_intermediate_size = shape[2] // 2
        else:
            moe_intermediate_size = getattr(
                config, "moe_intermediate_size", shape[1] // 2
            )
    else:
        moe_intermediate_size = getattr(config, "moe_intermediate_size", dim * 4)

    # The dense MLP is the shared expert.
    shared_expert = None
    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        gate_proj = getattr(mlp, "gate_proj", None)
        if gate_proj is not None and hasattr(gate_proj, "weight"):
            shared_hidden_dim = gate_proj.weight.shape[0]
        else:
            shared_hidden_dim = getattr(config, "intermediate_size", dim * 4)
        shared_expert = SharedExpertArchitecture(
            hidden_dim=shared_hidden_dim,
            dim=dim,
            has_sigmoid_gate=False,
        )

    return {
        "num_experts": num_experts,
        "dim": dim,
        "moe_intermediate_size": moe_intermediate_size,
        "top_k": _resolve_top_k(layer, gate, config),
        "score_func": _resolve_score_func(gate, config),
        "route_norm": _resolve_route_norm(gate, config),
        "route_scale": getattr(config, "routed_scaling_factor", 1.0),
        "num_expert_groups": getattr(config, "n_group", None),
        "num_limited_groups": getattr(config, "topk_group", None),
        "load_balance_coeff": getattr(config, "load_balance_coeff", 1e-3),
        "comm_backend": getattr(config, "comm_backend", "standard"),
        "shared_expert": shared_expert,
    }


# ---------------------------------------------------------------------------
# Field resolvers
# ---------------------------------------------------------------------------


def _resolve_num_experts(
    experts: nn.Module, gate: nn.Module | None, moe_block: nn.Module, config
) -> int:
    """Infer the total expert count from the HF MoE block or config.

    The count is spelled ``num_experts``, ``n_routed_experts``, or
    ``num_local_experts`` depending on the family, and lives on whichever of the
    three modules the family chose to hang it on -- so all three are probed
    before falling back to the config, and finally to the router's own shape.
    """
    for owner in (experts, gate, moe_block):
        if owner is None:
            continue
        for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
            val = getattr(owner, attr, None)
            if val is not None:
                return int(val)
    for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
        val = getattr(config, attr, None)
        if val is not None:
            return int(val)
    if gate is not None and hasattr(gate, "weight"):
        return gate.weight.shape[0]
    raise ValueError(
        f"Could not determine the expert count for {type(moe_block).__name__}: "
        "no num_experts / n_routed_experts / num_local_experts, and the router "
        "has no weight to read a shape from."
    )


def _resolve_top_k(moe_block: nn.Module, gate: nn.Module | None, config) -> int:
    """Infer top-k routing from the HF MoE block or config."""
    for owner in (moe_block, gate):
        if owner is None:
            continue
        for attr in ("top_k", "num_experts_per_tok", "top_k_experts"):
            val = getattr(owner, attr, None)
            if val is not None:
                return int(val)
    for attr in ("num_experts_per_tok", "top_k_experts"):
        val = getattr(config, attr, None)
        if val is not None:
            return int(val)
    raise ValueError(
        f"Could not determine top-k for {type(moe_block).__name__}: no "
        "top_k / num_experts_per_tok / top_k_experts on the block, the router, "
        "or the config."
    )


def _resolve_score_func(gate: nn.Module | None, config) -> str:
    """Determine the router scoring function (softmax or sigmoid)."""
    # DeepSeek-V3 / GLM use sigmoid with e_score_correction_bias.
    if gate is not None and "e_score_correction_bias" in getattr(gate, "_buffers", {}):
        return "sigmoid"

    scoring_func = getattr(config, "scoring_func", None)
    if scoring_func is not None:
        if scoring_func in ("softmax", "sigmoid"):
            return scoring_func
        raise ValueError(
            f"Unsupported scoring function '{scoring_func}'. Only 'softmax' and "
            "'sigmoid' are understood; add a case to _resolve_score_func."
        )

    return "softmax"


def _resolve_route_norm(gate: nn.Module | None, config) -> bool:
    """Determine whether the top-k router weights are renormalized.

    Some models (Mixtral, Qwen3.5) always normalize without exposing a config
    flag, so an absent ``norm_topk_prob`` means the router hardcodes it -- except
    for sigmoid routing, where the scores are used directly as weights.
    """
    if hasattr(config, "norm_topk_prob"):
        return bool(config.norm_topk_prob)
    if gate is not None and hasattr(gate, "norm_topk_prob"):
        return bool(gate.norm_topk_prob)
    return _resolve_score_func(gate, config) != "sigmoid"


def _probe_shared_experts(
    moe_block: nn.Module, config
) -> SharedExpertArchitecture | None:
    """Detect the shared expert on an HF MoE block, if it has one."""
    shared = None
    for name in ("shared_expert", "shared_experts", "shared_mlp"):
        shared = getattr(moe_block, name, None)
        if shared is not None:
            break

    if shared is None:
        return None

    # Determine shared expert intermediate size.
    for attr in ("intermediate_size", "hidden_size"):
        if hasattr(shared, attr):
            shared_hidden_dim = getattr(shared, attr)
            break
    else:
        # Try to infer from weight shapes.
        gate_proj = getattr(shared, "gate_proj", None)
        if gate_proj is not None and hasattr(gate_proj, "weight"):
            shared_hidden_dim = gate_proj.weight.shape[0]
        else:
            shared_hidden_dim = getattr(config, "shared_expert_intermediate_size", None)
            if shared_hidden_dim is None:
                n_shared = getattr(config, "n_shared_experts", 1)
                shared_hidden_dim = (
                    getattr(config, "moe_intermediate_size", config.hidden_size)
                    * n_shared
                )

    # Check for a sigmoid-gated shared expert (the Qwen3.5 pattern).
    has_sigmoid_gate = getattr(moe_block, "shared_expert_gate", None) is not None

    return SharedExpertArchitecture(
        hidden_dim=shared_hidden_dim,
        dim=config.hidden_size,
        has_sigmoid_gate=has_sigmoid_gate,
    )
