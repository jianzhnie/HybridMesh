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
added and removed internally.
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

from ..utils.batch_invariant import is_in_batch_invariant_mode
from .common.masks import (
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
)

logger = logging.getLogger(__name__)

__all__ = ["HFTransformerModel", "build_model_config", "build_model_config_for"]

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
    """
    offline = cfg.hf_model.count("/") != 1
    overrides = (
        {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
        }
        if offline
        else None
    )
    return build_model_config(
        cfg.hf_model, seq_len=cfg.max_seq_len, arch_overrides=overrides
    )


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


class HFTransformerModel(nn.Module):
    """A HF decoder stack behind a uniform training forward.

    Args:
        config: the HF config to instantiate (see ``build_model_config``).
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        config = _unwrap_text_config(config)
        config._attn_implementation = _flex_supported()
        AttentionInterface._global_mapping[_ATTN_IMPLEMENTATION] = _flex_attention_hf

        model_cls = _resolve_model_class(config)
        self.model = model_cls(config=config)
        self.model.config._attn_implementation = config._attn_implementation

        self.max_seq_len = getattr(config, "max_position_embeddings", None)
        self.cp_mesh = None

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

    def set_cp_mesh(self, mesh) -> None:
        """Record the CP mesh so logit dumps can tag their CP coordinate."""
        self.cp_mesh = mesh

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
        so ``state_dict`` keys would all carry a ``model.`` prefix -- and the
        parallelism layer, which walks children, would see a single opaque blob.
        Yielding the parts flattens both.
        """
        yield "tok_embeddings", self.tok_embeddings
        yield "layers", self.layers
        yield "norm", self.norm
        if self.lm_head is not None:
            yield "lm_head", self.lm_head
        if self.rotary_emb is not None:
            yield "rotary_emb", self.rotary_emb

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

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        attention_masks=None,
    ) -> torch.Tensor:
        """Run the decoder over one packed sequence and return logits.

        Args:
            input_ids: ``(T,)`` flat token ids.
            positions: ``(T,)`` per-token positions, resetting at document
                boundaries. Drives RoPE. Defaults to ``arange``, which is correct
                only when the sequence is a single document.
            attention_masks: a prebuilt BlockMask. Only the flex backend consumes
                it (see ``_apply_attention``); with sdpa the decoder is left to
                its own causal default.
        """
        local_seq_len = input_ids.shape[0]
        if positions is None:
            positions = torch.arange(local_seq_len, device=input_ids.device)

        kwargs = self._apply_attention(positions, attention_masks)

        # A HF decoder expects a batch dim; the wrapper's contract is flat.
        hidden_states = self._decoder(
            input_ids.unsqueeze(0),
            position_ids=positions.unsqueeze(0),
            use_cache=False,
            **kwargs,
        ).last_hidden_state.squeeze(0)

        logits = (
            self.lm_head(hidden_states) if self.lm_head is not None else hidden_states
        )

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
            attention_masks = self.get_attention_masks(positions=positions)

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
        # Packing is detected from ``positions``, not from the sequence length: a
        # document boundary is exactly where the position counter restarts, which
        # is the same convention ``get_attention_masks`` masks on. Length would be
        # the wrong test -- a long-context model may legitimately run a short
        # single-document sequence.
        if bool((positions[1:] < positions[:-1]).any()):
            raise ValueError(
                f"Attention backend {self.model.config._attn_implementation!r} "
                "cannot express the mask for a packed sequence: positions restart "
                "mid-sequence, so the batch holds more than one document. "
                "Sequence packing requires CUDA and the flex attention kernel."
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
