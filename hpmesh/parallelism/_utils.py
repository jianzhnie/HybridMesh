"""Shared helpers for the parallelism dimensions."""

from __future__ import annotations

import torch.nn as nn


def decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Best-effort lookup of the HF model's repeated decoder block list."""
    inner = getattr(model, "model", model)  # unwrap HFModelWrapper -> ForCausalLM
    base = getattr(inner, "model", inner)   # ForCausalLM -> base model
    layers = getattr(base, "layers", None)
    return list(layers) if layers is not None else []
