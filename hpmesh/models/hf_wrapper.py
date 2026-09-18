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

__all__ = ["HFTransformerModel", "build_model_config"]

# HF picks its attention function off ``config._attn_implementation``. Registering
# a name of our own lets us route through ``_flex_attention_hf`` without tripping
# HF's per-model ``_supports_flex_attn`` gate -- some models support flex but do
# not advertise it.
_ATTN_IMPLEMENTATION = "flex_torchtitan"


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
        config._attn_implementation = _ATTN_IMPLEMENTATION
        AttentionInterface._global_mapping[_ATTN_IMPLEMENTATION] = _flex_attention_hf

        model_cls = _resolve_model_class(config)
        self.model = model_cls(config=config)
        self.model.config._attn_implementation = _ATTN_IMPLEMENTATION

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

    # -- HF integration hooks --------------------------------------------------

    def set_cp_mesh(self, mesh) -> None:
        """Record the CP mesh so logit dumps can tag their CP coordinate."""
        self.cp_mesh = mesh

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
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the decoder over one packed sequence and return logits.

        Args:
            input_ids: ``(T,)`` flat token ids.
            positions: ``(T,)`` per-token positions, resetting at document
                boundaries. Drives RoPE. Defaults to ``arange``, which is correct
                only when the sequence is a single document.
            attention_masks: a prebuilt BlockMask; built here from ``positions``
                when omitted.
            labels: accepted for interface compatibility. Loss is computed
                outside this wrapper.
        """
        del labels  # the training loop owns the loss
        local_seq_len = input_ids.shape[0]
        if positions is None:
            positions = torch.arange(local_seq_len, device=input_ids.device)

        # Build the mask only when positions were supplied, mirroring the
        # preprocess step that normally builds it: positions are what mark packed
        # document boundaries, so without them there is nothing to mask. Flex
        # attention with no mask computes *full* attention, so a trainer using
        # this path must supply positions (or an explicit mask).
        if attention_masks is None:
            attention_masks = self.get_attention_masks(positions=positions)

        # A HF decoder expects a batch dim; the wrapper's contract is flat.
        hidden_states = self._decoder(
            input_ids.unsqueeze(0),
            position_ids=positions.unsqueeze(0),
            attention_mask=attention_masks,
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
