"""The one model-facing abstraction: a HF model plus the callables that size it.

Design intent (docs/FRAMEWORK_DESIGN.md section 1.1): the experiment exposes a
ModelSpec-shaped bundle and that shape is what hpmesh keeps. Everything a caller
needs in order to train a model travels in this one object:

* ``model``           -- the HF config the wrapper builds from, and its architecture.
* ``parallelize_fn``  -- how to spread it across the mesh.
* ``pipelining_fn``   -- how to split it into stages (optional; PP is not built yet).
* ``state_dict_adapter`` -- how to translate between HF and checkpoint keys (optional).

Note what is NOT here: torchtitan's ``ModelSpec.traverse``. That method exists
only so a ``Configurable`` override tree can reach the nested model config, and
hpmesh has no such mechanism -- the config travels as a plain dataclass.

``max_context_length`` and the callables carry no defaults on purpose: a spec
that silently forgot its parallelize function would fail much later, inside the
trainer, with a confusing error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

__all__ = ["HPModelSpec"]

# (model, parallel_dims, cfg) -> the (possibly wrapped) model.
# Declared with a bare Callable rather than a parameterised alias because the
# concrete argument types differ per model backend.
ParallelizeFunction = Callable[..., nn.Module]
PipeliningFunction = Callable[..., Any]
PostOptimizerBuildFunction = Callable[..., None]


@dataclass
class HPModelSpec:
    """Per-model bundle: the architecture config plus how to parallelize it."""

    name: str
    """Registry key, e.g. ``"hf_qwen3"``. Identifies the spec in logs and presets."""

    model: Any
    """The architecture config the wrapper builds from (for HF, an ``AutoConfig``)."""

    max_context_length: int
    """Longest sequence the model was built for; caps ``max_seq_len``."""

    parallelize_fn: ParallelizeFunction
    """Applies TP/EP/CP/FSDP ordering to one freshly built model."""

    pipelining_fn: PipeliningFunction | None = None
    """Splits the model into pipeline stages. ``None`` until PP is implemented."""

    post_optimizer_build_fn: PostOptimizerBuildFunction | None = None
    """Hook run after the optimizer exists (e.g. registering MoE load balancing)."""

    state_dict_adapter: type | None = None
    """Translates between the model's key layout and checkpoint keys."""
