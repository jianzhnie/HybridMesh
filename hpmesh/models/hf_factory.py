"""Construction-time helpers for the HF model layer.

Everything here runs BEFORE the model exists (or, for ``num_flops_per_token``,
describes it from the config alone): building the HF ``PretrainedConfig``,
resolving the ``ForCausalLM`` class it names, materializing a meta-device
model, and counting FLOPs. The model body itself -- ``HFTransformerModel`` and
the five-part contract -- lives in ``hf_wrapper.py``, which imports from here;
nothing here imports back.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

__all__ = [
    "build_model_config",
    "build_model_config_for",
    "materialize_meta_model",
    "num_flops_per_token",
]


# HF picks its attention function off ``config._attn_implementation``. Registering
# a name of our own lets us route through ``flex_attention_hf`` without tripping
# HF's per-model ``_supports_flex_attn`` gate -- some models support flex but do
# not advertise it.
_ATTN_IMPLEMENTATION = "flex_torchtitan"


def build_model_config(
    model_name_or_path: str,
    *,
    seq_len: int,
    arch_overrides: dict[str, Any] | None = None,
) -> PretrainedConfig:
    """Build the HF config the wrapper will instantiate.

    Two paths, matching the rest of hpmesh: a hub id (``"org/name"``) loads the
    real architecture, anything else is treated as an offline architecture name
    built from ``arch_overrides`` so the framework runs with no network access.
    """
    overrides = dict(arch_overrides or {})
    if overrides:
        overrides.setdefault("max_position_embeddings", seq_len)
        return AutoConfig.for_model(model_name_or_path, **overrides)

    config = AutoConfig.from_pretrained(model_name_or_path)
    config.max_position_embeddings = max(config.max_position_embeddings, seq_len)
    return config



def build_model_config_for(cfg) -> PretrainedConfig:
    """Build the model config for a training run.

    A thin adapter over :func:`build_model_config`: it maps the run's config onto
    the explicit ``(name, seq_len, arch_overrides)`` that function takes, and
    handles the offline case where the name is a bare architecture ("llama")
    rather than a hub id ("org/name"). Offline, the explicit sizes in ``cfg`` are
    authoritative, so they become the overrides; otherwise the Hub's own config
    wins and the overrides are empty.

    ``cfg.arch_overrides`` rides along on top of the explicit sizes, and exists
    because those sizes are only the dense-decoder six: an MoE or an MLA
    attention has fields -- ``n_routed_experts``, ``q_lora_rank`` -- that
    ``ModelConfig`` names nowhere, and a model built without them silently gets
    the architecture's own defaults (DeepSeek V3 asks for 256 experts). Same
    offline-only rule as the sizes: a hub id or a local checkpoint directory
    carries its own config, and overriding its architecture by hand is the one
    thing that would make the built model disagree with the weights it loads.

    It also sets ``attn_mask_type``, which is derived rather than configured. Any
    real corpus is packed -- ``datasets/build.py`` always runs the samples
    through ``ConcatThenSplitPackingConfig`` -- so a row holds several documents
    and attention must not cross a boundary between them. The synthetic random
    corpus is one document per row, where the document mask is a no-op.

    Left unset, ``attn_mask_type`` falls back to ``"causal"`` at the mask site.
    That is correct on a CPU/sdpa run only because the wrapper's packed-sequence
    guard raises first, and silently wrong on the flex path, where a causal-only
    mod attends across document boundaries without complaint. Deriving it here
    means the flag cannot disagree with the corpus the trainer loaded.
    """
    # A local checkpoint directory (e.g. "/abs/path" or "./ckpt") holds its own
    # config.json and must go through ``from_pretrained``; the "/" count test
    # alone would misread any absolute path as a hub id.
    offline = not os.path.exists(cfg.hf_model) and cfg.hf_model.count("/") != 1
    overrides = (
        {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            # Last, so an architecture setting may correct one of the six --
            # an MoE's hidden width and its per-expert width are independent,
            # but a model that redefines ``intermediate_size`` can say so
            # rather than have the override silently lose.
            **cfg.arch_overrides,
        }
        if offline
        else None
    )
    config = build_model_config(
        cfg.hf_model, seq_len=cfg.max_seq_len, arch_overrides=overrides
    )
    # A real corpus normally packs several documents and needs block-causal
    # masking. ``max_num_documents=1`` is the deliberate tensor-attention path:
    # one padded SFT document per row needs only ordinary causality, which lets
    # NPU/SDPA train without allowing attention across sample boundaries.
    config.attn_mask_type = (
        "causal"
        if cfg.dataloader.dataset == "random"
        or cfg.dataloader.max_num_documents == 1
        else "block_causal"
    )
    # The requested HF experts kernel travels on the config; the wrapper
    # validates settable-ness against the resolved model class at build time.
    config.experts_implementation = getattr(cfg, "experts_implementation", "native")
    # Same ride for the lm_head compute-dtype cast: the wrapper applies it at
    # build time, and None (the default) means no cast.
    config.compute_dtype = getattr(cfg, "compute_dtype", None)
    return config



def num_flops_per_token(cfg) -> int:
    """Training FLOPs per token for the model ``cfg`` describes.

    This is the denominator MFU divides into, so it has to describe the model
    that actually runs rather than a textbook transformer. It is derived from
    the same arch config the model is built from (``build_model_config_for``),
    which is what keeps the two from drifting.

    The convention, matching torchtitan's:

    * Every linear layer applied to every token costs ``2 * in * out`` FLOPs per
      token (one multiply and one add per weight), counted three times -- once
      for the forward pass and twice for the backward. That ``3 * 2 = 6`` is
      where the familiar ``6N`` comes from.
    * Attention adds ``6 * num_heads * (qk_head_dim + v_head_dim) * seq_len``
      per layer, which is the two attention contractions, not counted in the
      parameter term above. Causal sparsity and the recomputed backward pass are
      deliberately not discounted, following the same convention -- discounting
      them would produce an MFU above 100%, since the hardware peak is quoted
      against full dense matmuls.
    * Embedding lookups are not matmuls and cost nothing here, but the output
      projection is a matmul and is counted, whether or not its weight is tied
      to the embedding table: tying is a storage decision, not a compute one.

    Returns 0 for a model whose config does not expose the sizes the formula
    needs, which suppresses MFU rather than reporting a number derived from
    guessed geometry.
    """
    arch = unwrap_text_config(build_model_config_for(cfg))
    hidden = getattr(arch, "hidden_size", None)
    intermediate = getattr(arch, "intermediate_size", None)
    num_layers = getattr(arch, "num_hidden_layers", None)
    vocab_size = getattr(arch, "vocab_size", None)
    num_heads = getattr(arch, "num_attention_heads", None)
    if None in (hidden, intermediate, num_layers, vocab_size, num_heads):
        return 0

    num_kv_heads = getattr(arch, "num_key_value_heads", None) or num_heads
    head_dim = getattr(arch, "head_dim", None) or hidden // num_heads

    # Per token, in units of one multiply-add -- doubled at the end.
    per_layer = (
        2 * hidden * num_heads * head_dim  # q_proj
        + 2 * hidden * num_kv_heads * head_dim  # k_proj
        + 2 * hidden * num_kv_heads * head_dim  # v_proj
        + 2 * num_heads * head_dim * hidden  # o_proj
        + 3 * 2 * hidden * intermediate  # gate, up, down
    )
    lm_head = 2 * vocab_size * hidden
    # qk_head_dim and v_head_dim are both head_dim for the decoder-only models
    # this wrapper builds.
    attention = 6 * num_heads * 2 * head_dim * cfg.max_seq_len

    return 3 * (num_layers * per_layer + lm_head) + num_layers * attention



def unwrap_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return the text sub-config of a composite (vision-language) model.

    A VL checkpoint's top config describes the conditional model, not the text
    stack we train; its ``text_config`` does. Also ensure the sub-config carries
    ``architectures``, or the class lookup below has nothing to resolve.
    """
    if not hasattr(config, "text_config"):
        return config
    text_config = config.text_config
    text_config._attn_implementation = _ATTN_IMPLEMENTATION
    # ``build_model_config_for`` derives ``attn_mask_type`` and sets it on the
    # TOP config, but the model class is built from this sub-config and every
    # reader of that flag -- the packed guard, the mask builder, and
    # ``apply_cp``'s ulysses check -- reads ``self.model.config``. Left on the
    # top config it is invisible on a composite model: a packed corpus would
    # train with a causal-only mask (attention straight across document
    # boundaries), and the ulysses packed refusal would never fire. Carrying it
    # down here is what keeps "which corpus did the trainer load" answerable
    # from the object that actually runs.
    if hasattr(config, "attn_mask_type"):
        text_config.attn_mask_type = config.attn_mask_type
    if not getattr(text_config, "architectures", None):
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        )

        cls_name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(text_config.model_type, "")
        if cls_name:
            text_config.architectures = [cls_name]
    return text_config



def resolve_model_class(config: PretrainedConfig) -> type:
    """Find the ``ForCausalLM`` class named by the config.

    Prefer the name the config declares; fall back to the ``model_type`` ->
    ``Auto`` mapping, which covers configs whose architecture name does not match
    the installed class name.
    """
    import importlib

    import transformers
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    )

    for name in (config.architectures or []) + [
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(getattr(config, "model_type", ""), "")
    ]:
        if not name:
            continue
        cls = getattr(transformers, name, None) or getattr(
            importlib.import_module("transformers"), name, None
        )
        if cls is not None:
            return cls

    raise ImportError(
        f"Could not resolve a ForCausalLM class for model_type "
        f"{getattr(config, 'model_type', None)!r} "
        f"(architectures={getattr(config, 'architectures', None)!r}). "
        "Check that the installed transformers version ships this model."
    )



def materialize_meta_model(model: nn.Module, device: torch.device) -> None:
    """Materialize an already-parallelized HF model without a full copy.

    FSDP turns meta parameters into sharded meta ``DTensor`` objects; calling
    ``to_empty`` afterwards allocates only their local shards. Non-persistent
    buffers are not present in the HF checkpoint, so they must be reconstructed
    rather than left as uninitialized ``to_empty`` storage. Decoder RoPE is the
    only such buffer in the currently supported HF text models; unknown buffers
    fail loudly so a new architecture cannot train on garbage state.
    """
    meta_buffers = [
        (module, name)
        for module in model.modules()
        for name, value in module.named_buffers(recurse=False)
        if value.is_meta
    ]
    model.to_empty(device=device)

    refreshed: set[tuple[int, str]] = set()
    rope_modules = {module for module, name in meta_buffers if name == "inv_freq"}
    for module in rope_modules:
        if not hasattr(module, "config"):
            continue
        fresh = type(module)(module.config, device=device)
        for name, value in fresh.named_buffers(recurse=False):
            if name in {"inv_freq", "original_inv_freq"}:
                setattr(module, name, value)
                refreshed.add((id(module), name))

    unresolved = [
        f"{type(module).__name__}.{name}"
        for module, name in meta_buffers
        if (id(module), name) not in refreshed
    ]
    if unresolved:
        raise RuntimeError(
            "meta materialization has unsupported non-checkpoint buffers: "
            + ", ".join(unresolved)
        )

