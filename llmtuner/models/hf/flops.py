"""Training FLOPs per token: the MFU denominator.

One node of the ``hf`` package. The arithmetic is pure -- it reads an HF config
and returns a number -- which is why it lives apart from ``factory.py``, whose
job is building things. ``num_flops_per_token`` (the entry point the trainer
calls) stays there because it needs the resolved config; it delegates here.

The convention, matching torchtitan's (``models/utils.py`` plus the HF
backend's ``get_nparams_and_flops``), is documented on :func:`flops_per_token`.
"""

from __future__ import annotations

from typing import NamedTuple

from transformers.configuration_utils import PretrainedConfig

__all__ = [
    "flops_per_token",
    "quadratic_attention_flops_per_token",
]


def quadratic_attention_flops_per_token(
    *,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    seq_len: int,
    sliding_window_size: int | None = None,
) -> int:
    """Training FLOPs per token for quadratic or windowed attention.

    The two attention contractions (``q @ k`` and ``attn @ v``), each counted
    three times for forward and backward as ``6`` -- the convention upstream's
    ``models/utils.py`` uses. A windowed layer attends to at most
    ``sliding_window_size`` keys per query, so the sequence length is capped
    there; causal sparsity and the recomputed backward pass are not discounted,
    exactly as upstream.
    """
    attended_tokens = (
        seq_len if sliding_window_size is None else min(seq_len, sliding_window_size)
    )
    return 6 * num_heads * (qk_head_dim + v_head_dim) * attended_tokens


class _MoE(NamedTuple):
    """The MoE geometry of a config, and which layers route."""

    num_experts: int
    top_k: int
    expert_intermediate: int
    shared_intermediate: int
    layers: frozenset[int]


# ``_moe_geometry`` has three outcomes: no MoE (``None``), a resolved geometry, or
# a declared MoE whose layers/widths the config does not determine -- which makes
# the whole FLOPs count unresolvable rather than approximate.
_UNRESOLVED = object()


def flops_per_token(arch: PretrainedConfig, *, seq_len: int) -> int:
    """Training FLOPs per token for the architecture ``arch`` describes.

    The convention, matching torchtitan's (``models/utils.py`` plus the HF
    backend's ``get_nparams_and_flops``):

    * Every matmul weight applied to every token costs ``2 * in * out`` FLOPs
      per token (one multiply and one add per weight), counted three times --
      once for the forward pass and twice for the backward. That ``3 * 2 = 6``
      is where the familiar ``6N`` comes from. Norms, biases and attention
      sinks are not matmuls and are not counted; upstream's parameter walk does
      include them, which understates its own ``6N`` term by well under a
      percent.
    * Attention adds :func:`quadratic_attention_flops_per_token` per layer, the
      two attention contractions that the parameter term above does not cover.
    * A dense FFN is the three gated projections (gate, up, down). A MoE layer
      is its router plus ``top_k`` of its ``num_experts`` experts plus *all* of
      its shared experts -- the routed experts are the one place the parameter
      count and the compute diverge, which is why this cannot be read off
      ``intermediate_size`` alone.
    * MLA layers project through the low-rank ``q_lora``/``kv_lora`` path
      instead of a plain QKV, using ``qk_head_dim``/``v_head_dim`` rather than
      the ``head_dim`` field (which some MLA configs spell as the rope slice
      alone -- DeepSeek-V3's is 64 there against 192 for QK).
    * Embedding lookups are not matmuls and cost nothing here, but the output
      projection is a matmul and is counted, whether or not its weight is tied
      to the embedding table: tying is a storage decision, not a compute one.

    Returns 0 -- suppressing MFU and tflops rather than reporting a wrong
    magnitude -- when the geometry cannot be resolved exactly: a missing size,
    a MoE whose per-layer split or expert width the config does not state, a
    ``layer_types`` list shorter than the layer count, or a layer type whose
    parameter term is not derivable from the config (``linear_attention`` and
    friends, whose projections a per-family or model-based count would have to
    supply).
    """
    hidden = _int_attr(arch, "hidden_size")
    num_layers = _int_attr(arch, "num_hidden_layers")
    vocab_size = _int_attr(arch, "vocab_size")
    num_heads = _int_attr(arch, "num_attention_heads")
    if None in (hidden, num_layers, vocab_size, num_heads):
        return 0

    head_dims = _attention_head_dims(arch, hidden=hidden, num_heads=num_heads)
    if head_dims is None:
        return 0
    qk_head_dim, v_head_dim = head_dims

    moe = _moe_geometry(arch, num_layers=num_layers)
    if moe is _UNRESOLVED:
        return 0

    attended = _attention_windows(arch, num_layers=num_layers)
    if attended is None:
        return 0

    dense_intermediate = _int_attr(arch, "intermediate_size")
    if moe is None and dense_intermediate is None:
        return 0

    attention_params = _attention_param_term(
        arch,
        hidden=hidden,
        num_heads=num_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
    )
    if attention_params is None:
        return 0

    matmul_flops = 0
    for layer in range(num_layers):
        matmul_flops += attention_params
        if moe is not None and layer in moe.layers:
            matmul_flops += _moe_ffn_param_term(hidden=hidden, moe=moe)
        else:
            matmul_flops += 3 * 2 * hidden * dense_intermediate

    # The lm_head is the one parameter term outside the layer stack.
    matmul_flops += 2 * vocab_size * hidden
    attention_flops = sum(
        quadratic_attention_flops_per_token(
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            seq_len=seq_len,
            sliding_window_size=window,
        )
        for window in attended
    )
    return 3 * matmul_flops + attention_flops


def _int_attr(arch: PretrainedConfig, name: str) -> int | None:
    """An int attribute, or ``None`` when absent or not an int."""
    value = getattr(arch, name, None)
    return value if isinstance(value, int) else None


def _first_int_attr(arch: PretrainedConfig, *names: str) -> int | None:
    """The first of ``names`` the config spells as an int (families differ)."""
    for name in names:
        value = _int_attr(arch, name)
        if value is not None:
            return value
    return None


def _attention_head_dims(
    arch: PretrainedConfig, *, hidden: int, num_heads: int
) -> tuple[int, int] | None:
    """``(qk_head_dim, v_head_dim)``, or ``None`` when the config cannot say."""
    v_head_dim = _first_int_attr(arch, "v_head_dim")
    qk_head_dim = _first_int_attr(arch, "qk_head_dim")
    if qk_head_dim is None:
        nope = _int_attr(arch, "qk_nope_head_dim")
        rope = _int_attr(arch, "qk_rope_head_dim")
        qk_head_dim = None if nope is None or rope is None else nope + rope
    if qk_head_dim is not None or v_head_dim is not None:
        # MLA: half the pair is not enough to size the projections either.
        if qk_head_dim is None or v_head_dim is None:
            return None
        return qk_head_dim, v_head_dim

    head_dim = _int_attr(arch, "head_dim")
    if head_dim is None:
        if hidden % num_heads:
            return None
        head_dim = hidden // num_heads
    return head_dim, head_dim


def _attention_param_term(
    arch: PretrainedConfig,
    *,
    hidden: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
) -> int | None:
    """Per-layer attention projection cost, in units of ``2 * in * out``.

    ``None`` for an MLA config that does not state the ranks its projections
    are built from (``q_lora_rank``/``kv_lora_rank``), which would otherwise be
    a guessed parameter count.
    """
    num_kv_heads = _int_attr(arch, "num_key_value_heads") or num_heads
    q_lora_rank = _int_attr(arch, "q_lora_rank")
    if q_lora_rank is not None:
        kv_lora_rank = _int_attr(arch, "kv_lora_rank")
        qk_rope_head_dim = _int_attr(arch, "qk_rope_head_dim")
        qk_nope_head_dim = _int_attr(arch, "qk_nope_head_dim")
        if None in (kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim):
            return None
        # ``q_b_proj`` widens to the full QK width; ``kv_b_proj`` splits its
        # output into the nope key and the value, so it is sized by their sum.
        q_proj = hidden * q_lora_rank + q_lora_rank * num_heads * qk_head_dim
        kv_proj = hidden * (kv_lora_rank + qk_rope_head_dim) + kv_lora_rank * (
            num_heads * (qk_nope_head_dim + v_head_dim)
        )
    else:
        q_proj = hidden * num_heads * qk_head_dim
        kv_proj = hidden * num_kv_heads * qk_head_dim + hidden * num_kv_heads * (
            v_head_dim
        )
    o_proj = num_heads * v_head_dim * hidden
    return 2 * (q_proj + kv_proj + o_proj)


def _moe_geometry(
    arch: PretrainedConfig, *, num_layers: int
) -> _MoE | None | object:
    """The MoE geometry and its routed layers, ``None`` if there is no MoE.

    Returns :data:`_UNRESOLVED` for a declared MoE the config does not pin down
    -- that is the call sites' signal that the whole count must be suppressed.
    """
    num_experts = _first_int_attr(
        arch, "num_experts", "n_routed_experts", "num_local_experts"
    )
    if num_experts is None:
        return None
    top_k = _first_int_attr(arch, "num_experts_per_tok", "top_k")
    if top_k is None:
        return _UNRESOLVED

    # Dense layers: an explicit prefix (DeepSeek), an explicit index list
    # (Qwen3-MoE), or every-Nth (both, via ``moe_layer_freq``/``decoder_sparse_step``).
    first_dense = _int_attr(arch, "first_k_dense_replace") or 0
    sparse_step = _int_attr(arch, "decoder_sparse_step") or 1
    moe_freq = _int_attr(arch, "moe_layer_freq") or 1
    mlp_only = getattr(arch, "mlp_only_layers", None) or ()
    layers = frozenset(
        layer
        for layer in range(num_layers)
        if layer not in mlp_only
        and layer >= first_dense
        and (layer + 1) % sparse_step == 0
        and layer % moe_freq == 0
    )
    if not layers:
        return _UNRESOLVED

    # Every expert's width: the MoE field, or the single width the config uses
    # for every FFN when no layer is dense (Mixtral, OLMoE, GPT-OSS).
    uniform_width = not first_dense and not mlp_only and sparse_step == 1
    expert_intermediate = _int_attr(arch, "moe_intermediate_size")
    if expert_intermediate is None and uniform_width:
        expert_intermediate = _int_attr(arch, "intermediate_size")
    if expert_intermediate is None:
        return _UNRESOLVED

    shared_intermediate = _int_attr(arch, "shared_expert_intermediate_size")
    if shared_intermediate is None:
        n_shared = _int_attr(arch, "n_shared_experts") or 0
        shared_intermediate = n_shared * expert_intermediate

    if len(layers) != num_layers and _int_attr(arch, "intermediate_size") is None:
        return _UNRESOLVED
    return _MoE(
        num_experts=num_experts,
        top_k=top_k,
        expert_intermediate=expert_intermediate,
        shared_intermediate=shared_intermediate,
        layers=layers,
    )


def _moe_ffn_param_term(*, hidden: int, moe: _MoE) -> int:
    """A MoE layer's matmul cost, in units of ``2 * in * out``.

    Router + ``top_k`` routed experts + *all* shared experts: the routed
    experts are stored ``num_experts`` times over but only ``top_k`` of them
    see each token, which is exactly the ``top_k / num_experts`` active-expert
    ratio upstream's parameter walk applies.
    """
    router = hidden * moe.num_experts
    routed = moe.top_k * 3 * hidden * moe.expert_intermediate
    shared = 3 * hidden * moe.shared_intermediate
    return 2 * (router + routed + shared)


def _attention_windows(
    arch: PretrainedConfig, *, num_layers: int
) -> list[int | None] | None:
    """Per-layer attention window (``None`` = full), or ``None`` if unresolved."""
    layer_types = getattr(arch, "layer_types", None)
    if layer_types is None:
        sliding_window = _int_attr(arch, "sliding_window")
        # ``use_sliding_window=False`` is the explicit opt-out; a window that is
        # declared and not opted out of is what the model's own attention reads.
        if sliding_window is not None and getattr(arch, "use_sliding_window", True):
            return [sliding_window] * num_layers
        return [None] * num_layers

    layer_types = list(layer_types)
    if len(layer_types) < num_layers:
        # A config that declares fewer types than the layers it builds does not
        # say what the remaining layers do; upstream substitutes full attention,
        # which is only safe when that is really the default.
        return None
    windows: list[int | None] = []
    for layer_type in layer_types[:num_layers]:
        if layer_type in ("attention", "full_attention"):
            windows.append(None)
        elif layer_type == "sliding_attention":
            window = _int_attr(arch, "sliding_window")
            if window is None:
                return None
            windows.append(window)
        elif layer_type == "chunked_attention":
            chunk = _int_attr(arch, "attention_chunk_size")
            if chunk is None:
                return None
            windows.append(chunk)
        else:
            # ``linear_attention`` and any future spelling: its projection term
            # is not derivable from these generic fields.
            return None
    return windows
