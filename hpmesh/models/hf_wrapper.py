"""A thin wrapper that plugs a HuggingFace model into a training loop.

This is the whole model layer: build a HF ``ForCausalLM`` from its config, give
the training loop a uniform ``(tokens, positions) -> logits`` entry point, and
expose the decoder's parts under the names the rest of the framework expects
(``tok_embeddings`` / ``layers`` / ``norm`` / ``lm_head`` / ``rotary_emb``).

Deliberately thin. An earlier version of this file (torchtitan's
``experiments/transformers_modeling_backend/model.py``) carried a 230-line
monkey-patch of ``PreTrainedModel._init_weights`` that re-derived every weight
with torchtitan's depth-scaled scheme, plus a ``PretrainedConfig`` subclass that
synced torchtitan and HF attribute names. Both existed to make checkpoints
bit-compatible with torchtitan's native models. hpmesh trains HF models with
HF's own initialization, so neither is needed -- ``from_config`` produces a
fully initialized model and that is the one we train.

What replaces them is per-forward work the HF model does not do on its own:

* attention is routed through flex attention so packed-document masking can be
  expressed (see ``get_attention_masks``),
* RoPE is driven by explicit ``positions`` rather than an ``arange``, because
  packed samples reset their positions at each document boundary.

Forward shape: this wrapper is a *decoder wrapper*, not a CausalLM. It runs the
decoder and applies ``lm_head`` itself, so ``self.model.model`` is the bare text
stack. Callers pass flat ``(T,)`` token and position tensors; the batch dim is
added and removed internally. ``forward(..., skip_lm_head=True)`` returns the
hidden states instead, for the trainer's chunked-loss path.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.attention.flex_attention import and_masks
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.modeling_utils import AttentionInterface

from ..components.loss import next_token_targets
from ..datasets.random_data import Batch
from ..parallel.context_parallel import (
    shard_attention_mask_for_cp,
    shard_batch_for_cp,
    shard_batch_for_tp,
)
from ..parallel.parallel_dims import ParallelDims
from ..utils.batch_invariant import is_in_batch_invariant_mode
from .common.masks import (
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HFTransformerModel",
    "build_model_config",
    "build_model_config_for",
    "materialize_meta_model",
    "num_flops_per_token",
]

# HF picks its attention function off ``config._attn_implementation``. Registering
# a name of our own lets us route through ``_flex_attention_hf`` without tripping
# HF's per-model ``_supports_flex_attn`` gate -- some models support flex but do
# not advertise it.
_ATTN_IMPLEMENTATION = "flex_torchtitan"


def _flex_supported() -> str:
    """The attention implementation this machine can actually run.

    Flex attention lowers through inductor, and inductor has no CPU target, so
    ``torch.compile(flex_attention, ...)`` raises ``NotImplementedError`` off a
    CUDA device. That is a property of the machine, not of the run, so it is
    decided here rather than requested by the config.

    ``"sdpa"`` is a fallback in *backend*, not in *arithmetic*: on the path this
    wrapper takes (causal, no packing) both compute the same thing, for a reason
    worth spelling out. The wrapper deliberately passes no ``attention_mask``
    down to the decoder, and HF's sdpa path ignores ``is_causal`` whenever a mask
    is present, deriving causality from the mask instead. Handing it the
    ``BlockMask`` would therefore silently disable masking. Leaving the mask
    unset lets sdpa default to causal -- the same thing the flex causal mod
    applies. What is genuinely lost off CUDA is packed-document masking:
    ``get_attention_masks`` still builds a correct ``BlockMask``, it just has no
    flex kernel to run it in. Packed batches must run on CUDA.
    """
    return _ATTN_IMPLEMENTATION if torch.cuda.is_available() else "sdpa"


def _flex_attention_hf(module, query, key, value, attention_mask, **kwargs):
    """HF ``AttentionInterface`` entry point for flex attention.

    When a kernel has been attached under the attention module (the parallelism
    layer does this to declare a local SPMD region), route through it so the
    sharding declarations take effect; otherwise run flex directly on the plain
    tensors.
    """
    kernel = getattr(module, "_titan_flex_kernel", None)
    if kernel is None:
        return flex_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )
    out = kernel(query, key, value, module=module, block_mask=attention_mask, **kwargs)
    return out, None


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
    arch = _unwrap_text_config(build_model_config_for(cfg))
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


def _unwrap_text_config(config: PretrainedConfig) -> PretrainedConfig:
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


def _resolve_model_class(config: PretrainedConfig) -> type:
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


def _first_present(module: nn.Module, names: tuple[str, ...], what: str) -> str:
    """Return the first of ``names`` that ``module`` has.

    HF names the same submodule differently across model families
    (``embed_tokens``/``wte``, ``norm``/``final_layernorm``/``ln_f``), so the
    wrapper probes instead of hardcoding one family's spelling. Resolved once at
    construction: a missing part should fail at build time, not mid-forward.
    """
    for name in names:
        if hasattr(module, name):
            return name
    raise AttributeError(
        f"{type(module).__name__} has no {what} under any of {names}. "
        "Add the model's spelling to the probe in HFTransformerModel.__init__."
    )


def _collapse_batch_dims(
    inputs: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a ``(B, T)`` batch into the ``(B*T,)`` shape the forward takes.

    The wrapper is a single-sequence entry point -- it adds and removes its own
    batch dim around the decoder call. The synthetic source yields one document
    per row of length ``max_seq_len``, so the concatenation is exactly the
    single causal document the fallback attention path expects; RoPE is driven
    per row because positions restart at each row boundary.

    A packed (Grain) batch already arrives as a flat token stream, so for it
    this is the identity -- which is what makes the two sources one code path
    from here on. Used by both the training path (``preprocess_inputs``) and the
    pipeline path, which chunks rows before collapsing; keeping it in one place
    is what makes the two chunkings agree.
    """
    return inputs.reshape(-1), labels.reshape(-1)


def _document_shift(labels: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Next-token targets within a row, ``IGNORE_INDEX`` at each row end.

    The synthetic source hands over labels equal to its inputs, so the shift is
    the model's. Rows are independent documents of length ``seq_len``, so the
    shift has to stay *within* a row: the row-final position would predict the
    next document's first token, which the model had no context for. Those
    positions come back ``IGNORE_INDEX`` and are excluded from the loss.

    The packed source arrives already shifted and already masked at its document
    boundaries, so it never calls this.
    """
    return next_token_targets(labels.reshape(-1), seq_len=seq_len)


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


class HFTransformerModel(nn.Module):
    """A HF decoder stack behind a uniform training forward.

    Args:
        config: the HF config to instantiate (see ``build_model_config``).
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        config = _unwrap_text_config(config)
        num_heads = getattr(config, "num_attention_heads", None)
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads is not None and num_heads < 1:
            raise ValueError(f"num_attention_heads must be >= 1, got {num_heads}")
        if num_kv_heads is not None and num_kv_heads < 1:
            raise ValueError(f"num_key_value_heads must be >= 1, got {num_kv_heads}")
        if (
            num_heads is not None
            and num_kv_heads is not None
            and num_heads % num_kv_heads != 0
        ):
            raise ValueError(
                f"num_attention_heads ({num_heads}) must be divisible by "
                f"num_key_value_heads ({num_kv_heads})"
            )
        config._attn_implementation = _flex_supported()
        AttentionInterface._global_mapping[_ATTN_IMPLEMENTATION] = _flex_attention_hf

        model_cls = _resolve_model_class(config)
        self.model = model_cls(config=config)
        self.model.config._attn_implementation = config._attn_implementation

        self.max_seq_len = getattr(config, "max_position_embeddings", None)
        self.cp_mesh = None
        self._cp_load_balancer = None

        # The decoder is the text stack; lm_head is its sibling on the CausalLM.
        # Stored with object.__setattr__ on purpose: a plain ``self._decoder = ...``
        # would go through nn.Module.__setattr__, which registers it in
        # ``_modules`` and makes every parameter appear twice in the state dict
        # (once under ``model.``, once under ``_decoder.``).
        #
        # The attribute names are resolved once here; the accessors below are
        # properties so a pipeline stage can swap a part out in place.
        object.__setattr__(self, "_decoder", self.model.model)
        object.__setattr__(
            self,
            "_embed_name",
            _first_present(self._decoder, ("embed_tokens", "wte"), "token embedding"),
        )
        object.__setattr__(
            self,
            "_norm_name",
            _first_present(
                self._decoder, ("norm", "final_layernorm", "ln_f"), "final norm"
            ),
        )

    # -- parts, under the names the rest of the framework uses ------------------

    @property
    def tok_embeddings(self) -> nn.Module:
        return getattr(self._decoder, self._embed_name)

    @tok_embeddings.setter
    def tok_embeddings(self, value: nn.Module) -> None:
        setattr(self._decoder, self._embed_name, value)

    @property
    def layers(self) -> nn.ModuleList:
        return self._decoder.layers

    @layers.setter
    def layers(self, value) -> None:
        self._decoder.layers = value

    @property
    def norm(self) -> nn.Module:
        return getattr(self._decoder, self._norm_name)

    @norm.setter
    def norm(self, value: nn.Module) -> None:
        setattr(self._decoder, self._norm_name, value)

    @property
    def lm_head(self) -> nn.Module | None:
        return getattr(self.model, "lm_head", None)

    @lm_head.setter
    def lm_head(self, value: nn.Module | None) -> None:
        self.model.lm_head = value

    @property
    def rotary_emb(self) -> nn.Module | None:
        return getattr(self._decoder, "rotary_emb", None)

    @rotary_emb.setter
    def rotary_emb(self, value: nn.Module | None) -> None:
        self._decoder.rotary_emb = value

    @property
    def enable_weight_tying(self) -> bool:
        """Whether ``lm_head`` and the embedding share one ``Parameter``.

        Read by FSDP, which must not let one Parameter be owned by two FSDP
        units; the two modules are wrapped together when this is true.

        Compared by identity rather than by ``config.tie_word_embeddings``: that
        flag records intent, and a model may carry an unshared head anyway (or
        share one the flag does not mention). Identity is also exactly the check
        FSDP2 itself performs, so the answer here matches what FSDP will do.
        """
        embed, head = self.tok_embeddings, self.lm_head
        if head is None:
            return False
        return getattr(embed, "weight", None) is getattr(head, "weight", None)

    # -- HF integration hooks --------------------------------------------------

    def set_cp_mesh(self, mesh, *, load_balancer: str | None = None) -> None:
        """Record the CP mesh so logit dumps can tag their CP coordinate.

        Also records the CP load-balancer type: the trainer shards the batch
        with it, and the forward's BlockMask Q-shard must rearrange Q the same
        way or the mask indexes the wrong queries.
        """
        self.cp_mesh = mesh
        self._cp_load_balancer = load_balancer

    @property
    def tp_plan(self) -> dict[str, str]:
        """HF's TP plan, with patterns rewritten to THIS wrapper's module paths.

        HF states its plan relative to the model it ships -- ``layers.*.q_proj``,
        plus a ``model.``-prefixed variant for families that nest one level
        deeper. The parallel layer, however, walks *this* wrapper, whose
        ``named_modules`` paths all sit under ``model.`` because that is the
        attribute the HF CausalLM is held in. Prefixing every pattern with
        ``model.`` maps one spelling onto the other, so HF's own two variants
        both resolve here and no call site has to know either layout.

        Without this the plan is simply not found (the attribute lives on the
        inner HF model) and ``apply_tp`` silently shards nothing -- a replicated
        run that looks like a working one. That failure is what makes this
        translation load-bearing rather than cosmetic.

        Returns ``{}`` when the model ships no plan. That is left as an empty
        plan rather than an error because "no declared plan" is a real answer:
        ``apply_tp`` then leaves the model replicated instead of guessing.
        """
        plan = getattr(self.model, "_tp_plan", None) or {}
        return {f"model.{pattern}": spec for pattern, spec in plan.items()}

    def named_children(self):
        """Present the decoder's parts as direct children.

        ``nn.Module.named_children`` would yield exactly one child (``self.model``),
        so the parallelism layer, which walks children, would see a single opaque
        blob. Yielding the parts here is what lets it address the decoder's pieces
        (``layers.*`` and friends) directly.

        This does NOT flatten ``state_dict`` keys: the state dict is built from
        ``_modules``, which still holds everything under ``self.model``, so keys
        keep their ``model.`` prefix. Only the child *iteration* is reshaped.
        """
        yield "tok_embeddings", self.tok_embeddings
        yield "layers", self.layers
        yield "norm", self.norm
        if self.lm_head is not None:
            yield "lm_head", self.lm_head
        if self.rotary_emb is not None:
            yield "rotary_emb", self.rotary_emb

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims | None,
        parallelism=None,
        max_context_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Turn a dataloader batch into ``(inputs, labels, extra_kwargs)``.

        The seam torchtitan's trainer calls, and the reason it is on the model
        rather than in the loop: every step below is a statement about *this*
        architecture's input contract, and one of them needs the full-length
        positions that only exist before the sequence is sharded.

        1. **Normalize the batch shape.** The two loaders disagree about what a
           batch is -- the synthetic one yields ``(B, T)`` rows of one document
           each, a Grain one a flat packed stream -- so they are reconciled
           here. ``num_valid_tokens`` is *not* read: the trainer pops it before
           this call, because the loss denominator has to be the pre-shard count
           and reduced across DP before the first backward.
        2. **Collapse the batch dim**, turning ``(B, T)`` into the flat ``(B*T,)``
           the forward takes.
        3. **Build the attention mask**, when the batch carries ``positions``.
           This has to happen before step 5: the document structure of a packed
           batch is a global property, so the mask is built over the FULL
           sequence and only then Q-sharded.
        4. **Shard for context parallelism**, positions included, so RoPE
           follows each token to its rank.
        4b. **Shard for tensor parallelism** (sequence parallelism): the fused
           TP GEMMs all-gather the sequence inside each projection, so the
           forward must be entered holding only this rank's ``T / tp`` slice --
           otherwise the gather concatenates ``tp`` copies of the full sequence
           and every sharded weight gradient comes out ``tp`` times too large.
           Only the token-carrying tensors (``inputs``, ``labels``) are cut:
           after the in-projection gather, attention and RoPE see the assembled
           sequence (the CP shard, or the full sequence with CP off), so
           ``positions`` and the mask keep their CP/full length. This matches
           torchtitan's joint ``(CP, TP)`` sequence sharding
           (``hf_sharding.py``'s ``PartitionSpec(DP, (CP, TP), None)``): the CP
           shard composed with a contiguous TP slice of it is exactly the joint
           (CP outer, TP inner) split.
        5. **Return the leftover dict as ``extra_kwargs``.** Those are splatted
           into ``forward``, so anything left here must be one of its keyword
           parameters -- ``positions`` and ``attention_masks``, and nothing else.

        ``positions`` is optional: the synthetic source has none and the forward
        falls back to its own ``arange``, which is right for a single document
        but must not be relied on for a packed one. ``parallelism`` is accepted
        for signature parity with the reference but unused: hpmesh's CP
        load-balancer string is latched onto this wrapper by ``apply_cp`` (see
        ``set_cp_mesh``).

        The reference also takes ``max_num_documents``, to size a fixed-shape
        varlen mask for CUDA graph capture. It is not accepted here because
        there is no path for it to take: hpmesh's CP path goes through
        ``create_attention_mask`` (a flex ``BlockMask``), which rebuilds from
        the positions shard and takes no capacity bound, and the packing
        collator already caps segments at one row. Accepting it would mean a
        parameter the trainer passes and this method silently discards.

        Tensors arrive on the trainer's device; the trainer moves them before
        calling, so this is pure structure and stays device-free.
        """
        del parallelism
        extra_kwargs: dict[str, Any] = {}
        positions = None

        if isinstance(input_dict, Batch):
            # Rows are independent documents of length T.
            labels = _document_shift(
                input_dict.labels, seq_len=input_dict.labels.shape[-1]
            )
            inputs = input_dict.input_ids
        else:
            # Packed stream: the collator already shifted and masked the labels
            # at every document boundary, so for this path the shift is a read.
            inputs, labels = input_dict["input"], input_dict["labels"]
            positions = input_dict.get("positions")

        inputs, labels = _collapse_batch_dims(inputs, labels)
        if positions is not None:
            positions = positions.reshape(-1)

        # Whether the mask must be prebuilt full-length: a packed corpus
        # (``block_causal``) carries a document structure that is not
        # recoverable from a CP positions shard, so the mask must exist before
        # the shard below. This is decided by ``attn_mask_type``, NOT by
        # scanning ``positions`` for restarts: a restart scan misses the two
        # shapes a real packed corpus produces -- a document that fills the
        # whole row (the packing collator splits overlong documents into
        # single-document rows, whose positions are a plain arange) and a
        # boundary after a length-1 document (positions ``[0, 0, ...]`` have no
        # descending edge). The first would crash the CP forward for lack of a
        # prebuilt mask; the second would disarm the sdpa packed-guard and let
        # attention cross the boundary silently.
        packed = positions is not None and (
            getattr(self.model.config, "attn_mask_type", "causal") == "block_causal"
        )

        # Built before the CP shard, always from the FULL-length positions --
        # which is exactly why this lives here and not in the loop: after the
        # shard below, no rank holds a positions vector that can describe the
        # document structure. When the mask is built there is no need to hand
        # it to the forward: ``_apply_attention`` builds one from ``positions``
        # anyway, so passing it would be a second copy rather than a saving.
        cp_mesh = (
            None if parallel_dims is None else parallel_dims.get_optional_mesh("cp")
        )
        if cp_mesh is None and positions is not None:
            mask = self.get_attention_masks(positions=positions)
            if self.model.config._attn_implementation == _ATTN_IMPLEMENTATION:
                extra_kwargs["attention_masks"] = mask

        if cp_mesh is not None:
            if positions is None:
                # The forward's own ``arange`` default would restart at 0 on
                # every rank; the shard needs positions that describe the whole
                # sequence.
                positions = torch.arange(inputs.numel(), device=inputs.device)
            # A causal-only mask (a single document) can be rebuilt from this
            # rank's positions shard, which is what ``_get_cp_attention_masks``
            # does. Packed cannot: ``positions`` is about to be sharded and the
            # document structure is not recoverable from a shard of it, so the
            # full-length mask is built first and Q-sharded to match. The GQA
            # head count still divides by cp -- sharding Q does not change how
            # many Q heads a rank owns.
            if packed:
                attention_masks = shard_attention_mask_for_cp(
                    self.get_attention_masks(positions=positions),
                    cp_mesh,
                    self._cp_load_balancer,
                )
                if self.model.config._attn_implementation == _ATTN_IMPLEMENTATION:
                    extra_kwargs["attention_masks"] = attention_masks
            inputs, labels, positions = shard_batch_for_cp(
                inputs,
                labels,
                positions,
                cp_mesh,
                load_balancer=self._cp_load_balancer,
            )

        tp_mesh = (
            None if parallel_dims is None else parallel_dims.get_optional_mesh("tp")
        )
        if tp_mesh is not None:
            # Sequence parallelism premise (step 4b above): cut the token-
            # carrying tensors along the TP axis. Positions are NOT cut --
            # after the in-projection all-gather, RoPE and attention see the
            # assembled sequence, so they keep the CP-shard (or, with CP off,
            # full) length. Synthesize them full-length when the batch did not
            # carry any: the forward's own ``arange`` default would be sized to
            # the TP-sharded input and restart at 0 on every rank.
            if positions is None:
                positions = torch.arange(inputs.numel(), device=inputs.device)
            inputs, labels = shard_batch_for_tp(inputs, labels, tp_mesh)

        if positions is not None:
            extra_kwargs["positions"] = positions
        return inputs, labels, extra_kwargs

    def get_attention_masks(self, positions: torch.Tensor):
        """Build the flex BlockMask for this batch.

        ``attn_mask_type`` selects between plain causal and causal-plus-same-
        document. The latter is the packed path: samples share one sequence, so
        attention must not cross a document boundary (positions reset to 0
        there). Both cases return a BlockMask -- with no mask at all flex would
        compute full attention.
        """
        if getattr(self.model.config, "attn_mask_type", "causal") == "block_causal":
            mask_mod = and_masks(
                get_causal_mask_mod(),
                get_document_mask_mod(positions),
            )
        else:
            mask_mod = get_causal_mask_mod()

        num_tokens = positions.shape[0]
        return create_attention_mask(
            mask_mod,
            1,
            None,
            num_tokens,
            num_tokens,
            device=positions.device,
            BLOCK_SIZE=128,
            separate_full_blocks=not is_in_batch_invariant_mode(),
        )

    def _get_cp_attention_masks(self, positions: torch.Tensor):
        """Build the BlockMask for a CP forward: full-length, then Q-sharded.

        Under CP, ``positions`` is this rank's shard of the sequence (possibly
        load-balancer-rearranged), so it cannot describe the full document
        structure: the mask is built over the FULL sequence and then sharded
        along its Q axis, matching how the CP kernel's gathered K/V stay
        full-length. ``get_attention_masks`` builds the full mask from an
        arange -- valid because only the causal mod is taken here.

        Packed batches (``block_causal``) cannot take this path: the document
        mask needs the full positions, which only the caller has. Build the
        full-length mask with ``get_attention_masks(full_positions)``, Q-shard
        it with ``shard_attention_mask_for_cp``, and pass it as
        ``attention_masks``.
        """
        if getattr(self.model.config, "attn_mask_type", "causal") == "block_causal":
            raise ValueError(
                "Context parallel with packed sequences needs a prebuilt mask: "
                "build the full-length BlockMask with get_attention_masks from "
                "the FULL positions, Q-shard it with shard_attention_mask_for_cp, "
                "and pass it to forward as attention_masks. The positions this "
                "forward receives are already CP-sharded and cannot describe the "
                "full document structure."
            )
        cp_size = self.cp_mesh.size()
        full_len = positions.shape[0] * cp_size
        full_positions = torch.arange(full_len, device=positions.device)
        mask = self.get_attention_masks(positions=full_positions)
        return shard_attention_mask_for_cp(mask, self.cp_mesh, self._cp_load_balancer)

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        attention_masks=None,
        skip_lm_head: bool = False,
    ) -> torch.Tensor:
        """Run the decoder over one packed sequence and return logits.

        Args:
            input_ids: ``(T,)`` flat token ids. Under CP, this rank's sequence
                shard -- ``(T/cp,)``. Under pipeline parallelism, a non-first
                stage receives the previous stage's output instead: ``(T, H)``
                hidden states, detected by ``tok_embeddings`` having been split
                out (replaced by an ``nn.Identity``).
            positions: ``(T,)`` per-token positions, resetting at document
                boundaries. Drives RoPE. Defaults to ``arange``, which is correct
                only when the sequence is a single document. Under CP, the
                matching shard of the full positions.
            attention_masks: a prebuilt BlockMask. Only the flex backend consumes
                it (see ``_apply_attention``); with sdpa the decoder is left to
                its own causal default. Under CP, a full-length mask already
                Q-sharded by ``shard_attention_mask_for_cp``.
            skip_lm_head: return the ``(T, H)`` hidden states instead of logits.
                The chunked-loss path uses this so the trainer can run lm_head +
                cross-entropy per sequence chunk (see
                ``components.loss.chunked_lm_head_cross_entropy``) rather than
                materialize the full ``T * V`` logits.
        """
        if isinstance(self.tok_embeddings, nn.Identity):
            # Non-first pipeline stage: the input IS the previous stage's
            # hidden states, so the embedding lookup is skipped by feeding the
            # decoder ``inputs_embeds`` directly. PP does not shard the
            # sequence, so the local ``arange`` positions default stays right.
            local_seq_len = input_ids.shape[0]
            decoder_input = {"inputs_embeds": input_ids.unsqueeze(0)}
        else:
            local_seq_len = input_ids.shape[0]
            decoder_input = {"input_ids": input_ids.unsqueeze(0)}
        if positions is None:
            positions = torch.arange(local_seq_len, device=input_ids.device)

        kwargs = self._apply_attention(positions, attention_masks)

        # A HF decoder expects a batch dim; the wrapper's contract is flat.
        hidden_states = self._decoder(
            **decoder_input,
            position_ids=positions.unsqueeze(0),
            use_cache=False,
            **kwargs,
        ).last_hidden_state.squeeze(0)

        if (
            self.lm_head is not None
            and not isinstance(self.lm_head, nn.Identity)
            and not skip_lm_head
        ):
            logits = self.lm_head(hidden_states)
        else:
            # Non-final pipeline stage, or a chunked-loss forward: the output
            # must own its storage rather than be a view of the decoder's
            # ``last_hidden_state`` (squeeze above), because split-backward
            # schedules (ZBVZeroBubble) call ``detach_()`` on stage outputs,
            # which views do not support. The chunked-loss caller does not
            # detach in place, but shares the branch so both skip reasons stay
            # one code path -- the clone is one T*H copy per forward.
            logits = hidden_states.clone()

        _dump_dir = os.environ.get("HF_BACKEND_LOGIT_DUMP")
        if _dump_dir is not None:
            self._maybe_dump_logits(_dump_dir, logits)

        return logits

    def _apply_attention(
        self, positions: torch.Tensor, attention_masks
    ) -> dict[str, Any]:
        """Cross the ROLE-IN / CONVENTION-OUT seam: decide what to hand the decoder.

        Roles are fixed: this wrapper ALWAYS routes through an attention
        implementation, and that implementation is ALWAYS fed a mask describing
        how tokens may attend -- flex consumes it as a ``BlockMask``. How a
        *backend* wants its mask is a different question (sdpa wants a boolean
        tensor, and derives causality from the mask's presence), so it is settled
        here and nowhere else.

        Consequently the mask is built unconditionally, even on the backend that
        ends up discarding it: not building it would make the two paths differ in
        more than the backend, and would hide the packing gap this fallback
        leaves open.
        """
        if attention_masks is None:
            if self.cp_mesh is None:
                attention_masks = self.get_attention_masks(positions=positions)
            else:
                attention_masks = self._get_cp_attention_masks(positions)

        # is_causal is the flex-only spelling: it selects which mod the BlockMask
        # runs, so it is withheld from every other backend (see below).
        if self.model.config._attn_implementation == _ATTN_IMPLEMENTATION:
            return {"attention_mask": attention_masks, "is_causal": False}

        # Every other backend wants a tensor mask (or none). There is no generic
        # conversion from a BlockMask, and one is not wanted: the only hpmesh
        # case that needs a tensor mask is packing, which the flex fallback
        # cannot express anyway. Fail loudly rather than run with the mask
        # silently dropped -- that would turn the packed path into full attention.
        #
        # Packing is detected from ``attn_mask_type``, not from scanning
        # ``positions`` for restarts: a restart scan misses a boundary after a
        # length-1 document (positions ``[0, 0, ...]`` never descend), which
        # would let sdpa run plain causal attention across the boundary with no
        # complaint. The flag is derived from the corpus at config build time
        # (``build_model_config_for``), so it cannot disagree with the batch.
        if getattr(self.model.config, "attn_mask_type", "causal") == "block_causal":
            raise ValueError(
                f"Attention backend {self.model.config._attn_implementation!r} "
                "cannot express the mask for a packed sequence: this run's "
                "corpus is packed (attn_mask_type='block_causal'), and a "
                "tensor-mask backend has no way to keep attention from "
                "crossing a document boundary. Sequence packing requires CUDA "
                "and the flex attention kernel."
            )

        # A single causal document. Hand the decoder NOTHING and let HF follow its
        # own path: it builds a 4D mask only when it must (sliding-window layers,
        # padding) and otherwise returns None, and its sdpa wrapper derives
        # ``is_causal`` from whether a mask is present. Passing ``is_causal=True``
        # ourselves would double up with the mask HF does build -- sdpa rejects
        # ``attn_mask`` together with ``is_causal=True`` -- which is what an
        # earlier version of this method got wrong.
        return {"attention_mask": None}

    def _maybe_dump_logits(self, dump_dir: str, logits: torch.Tensor) -> None:
        """Append this rank's logits (one entry per forward) for numerical tests."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        cp_coord = self.cp_mesh.get_local_rank() if self.cp_mesh is not None else 0
        recs = getattr(self, "_logit_dump_recs", None)
        if recs is None:
            recs = self._logit_dump_recs = []
        recs.append((cp_coord, logits.detach().float().cpu()))
        torch.save(recs, os.path.join(dump_dir, f"logits_rank{rank}.pt"))

    def __setattr__(self, name, value) -> None:
        """Route property-backed names through their setters.

        ``nn.Module.__setattr__`` registers modules directly, which would bypass
        the property setters defined above (``self.layers = ...`` would create a
        new child instead of replacing the decoder's).
        """
        prop = getattr(type(self), name, None)
        if isinstance(prop, property) and prop.fset is not None:
            prop.fset(self, value)
            return
        super().__setattr__(name, value)
