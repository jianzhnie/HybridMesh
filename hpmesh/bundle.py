"""Model bundle -- wraps a HF model into the framework's single model abstraction.

Design intent (FRAMEWORK_DESIGN.md section 2.2, ModelBundle): the framework's ONLY
model-facing abstraction is "a HF model + the functions that parallelize it". We do
NOT reimplement the model; we wrap AutoModelForCausalLM and expose a uniform
forward that returns a scalar loss (HF already computes it when given labels).

Learning note: keep this thin. The interesting distributed machinery lives in
parallel/, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .trainer.config import HybridMeshConfig


def _resolve(module: nn.Module, names: tuple[str, ...], what: str) -> nn.Module:
    """Return the first of ``names`` that ``module`` has.

    HF spells the same submodule differently across families (``embed_tokens``
    vs ``wte``, ``norm`` vs ``ln_f``), while the FSDP layer reads one fixed
    name. Resolving here fails at the point of the bad model rather than deep
    inside a distributed call.
    """
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"{type(module).__name__} has no {what} under any of {names}")


@dataclass
class Batch:
    """One micro-batch. input_ids/labels are (batch, seq) on the model's device."""

    input_ids: torch.Tensor
    labels: torch.Tensor


class HFModelWrapper(nn.Module):
    """Uniform (input_ids, labels) -> scalar loss interface over a HF CausalLM.

    HF's ForCausalLM already shifts labels and computes cross-entropy when labels
    are provided, so forward is a thin passthrough that surfaces the scalar loss.

    The remaining members exist because the FSDP layer reads the decoder's parts
    off the top-level module under torchtitan's names, while HF nests them under
    ``model.model``. Exposing them here -- once, on the wrapper -- is what keeps
    the FSDP layer from having to re-derive the HF layout per call.
    """

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, labels=labels)
        return out.loss

    # -- decoder parts, under the names the FSDP layer reads --------------------

    @property
    def _decoder(self) -> nn.Module:
        return self.model.model

    @property
    def tok_embeddings(self) -> nn.Module:
        return _resolve(self._decoder, ("embed_tokens", "wte"), "token embedding")

    @property
    def layers(self) -> nn.Module:
        return self._decoder.layers

    @property
    def norm(self) -> nn.Module:
        return _resolve(self._decoder, ("norm", "ln_f"), "final norm")

    @property
    def lm_head(self) -> nn.Module | None:
        return getattr(self.model, "lm_head", None)

    @property
    def enable_weight_tying(self) -> bool:
        """Whether ``lm_head`` and the embedding share one ``Parameter``.

        FSDP2 forbids one Parameter being owned by two FSDP units, so the two
        modules are wrapped as one unit when this is true. Compared by identity
        rather than by ``config.tie_word_embeddings``: that flag records intent,
        and identity is the exact check FSDP2 itself performs.
        """
        head = self.lm_head
        if head is None:
            return False
        embed = self.tok_embeddings
        return embed is not None and (
            getattr(embed, "weight", None) is getattr(head, "weight", None)
        )


def build_model_config(cfg: HybridMeshConfig) -> AutoConfig:
    """Build a HF config. Offline path uses AutoConfig.for_model so the prototype
    runs with no network and no downloaded weights."""
    if cfg.hf_model.count("/") == 1:
        # Looks like a hub id ("org/name") -> real architecture from the Hub.
        return AutoConfig.from_pretrained(cfg.hf_model)
    # Offline: construct a tiny architecture locally.
    return AutoConfig.for_model(
        cfg.hf_model,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.max_seq_len,
    )


@dataclass
class ModelBundle:
    """The framework's single model abstraction (cf. Titan's ModelSpec, flattened).

    model      -- the wrapped HF model (parameters possibly on meta/CPU; materialized
                  and sharded later by the parallelism layer).
    model_config -- the HF config (kept for FLOPs/parallelism decisions).
    """

    model: HFModelWrapper
    model_config: AutoConfig


def build_bundle(cfg: HybridMeshConfig, *, device: torch.device) -> ModelBundle:
    """Instantiate the HF model (random init) and wrap it.

    Seeding happens in trainer before this is called so that all ranks build the
    SAME initial weights (required for bit-exact DP comparisons).
    """
    model_config = build_model_config(cfg)
    hf_model = AutoModelForCausalLM.from_config(model_config)
    hf_model = hf_model.to(device)
    return ModelBundle(model=HFModelWrapper(hf_model), model_config=model_config)
