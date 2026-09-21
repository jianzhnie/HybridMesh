"""Activation checkpointing: recompute each decoder layer during backward.

Vendored in shape from torchtitan's ``distributed/activation_checkpoint.py``,
cut down to its ``FullAC``: every decoder layer is wrapped in torch's
non-reentrant ``checkpoint_wrapper``, so a forward keeps only the layer's
inputs and recomputes its activations inside backward -- one extra forward per
layer in exchange for the layer's activation memory.

The wrapper factory is the same one torchtitan uses
(``torch.distributed.algorithms._checkpoint.checkpoint_wrapper``), with the
same two non-default knobs: ``preserve_rng_state=True`` so the recompute sees
the RNG state the original forward saw (bitwise-identical logits and
gradients), and ``early_stop=False`` so a checkpointed region inside the layer
cannot end the recompute early.

Selective/per-op AC (torchtitan's ``SelectiveAC``) is deliberately not ported:
``mode`` is the extension point, and any mode other than ``"none"``/``"full"``
raises rather than silently running uncheckpointed.
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

logger = logging.getLogger(__name__)

__all__ = ["VALID_AC_MODES", "apply_ac"]

VALID_AC_MODES = ("none", "full")


def apply_ac(
    model: nn.Module,
    mode: str = "none",
    *,
    preserve_rng_state: bool = True,
) -> nn.Module:
    """Wrap every decoder layer of ``model`` in a checkpoint wrapper.

    No-op when ``mode == "none"`` (the model is handed back untouched), so the
    caller needs no mode check of its own. ``mode == "full"`` checkpoints the
    whole layer: its forward activations are dropped and recomputed during
    backward. Any other mode is a loud error -- the string is where a future
    selective mode plugs in, not a value to guess around.

    Apply after TP/EP/CP and before compile/FSDP (torchtitan's order in
    ``parallelize_llama``): the wrapper must enclose the TP-sharded layer, and
    FSDP has to wrap the checkpointed block so the recompute runs with
    all-gathered parameters instead of re-triggering the gather.
    """
    if mode == "none":
        return model
    if mode != "full":
        raise ValueError(
            f"Unknown activation checkpointing mode {mode!r}; expected one of "
            f"{VALID_AC_MODES}. Only 'full' (per-layer, torchtitan's FullAC) is "
            "implemented."
        )

    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"apply_ac expects a HFTransformerModel (with .layers); got "
            f"{type(model).__name__}."
        )
    for layer_id, transformer_block in layers.named_children():
        layers.register_module(
            layer_id,
            ptd_checkpoint_wrapper(
                transformer_block,
                preserve_rng_state=preserve_rng_state,
                early_stop=False,
            ),
        )
    logger.info("Applied full activation checkpointing to %d layers", len(layers))
    return model
