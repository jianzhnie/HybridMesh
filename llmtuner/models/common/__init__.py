"""The model vocabulary, shared across architectures.

Vendored from torchtitan ``models/common/``: the pieces an HF text model is put
together from, with nothing in here knowing which family it belongs to. The two
families big enough to need more than one file are subpackages -- ``attention/``
(the fused QKV projection, the mask modifiers) and ``moe/`` (router, experts,
dispatcher, block, balance loss, balancing hooks) -- and the rest are one node
per file: ``activation``, ``async_linear`` (TP-overlapped GEMMs), ``aux_loss``
(the gradient carrier), ``cast_linear``, ``embedding``, ``feed_forward``,
``linear``, ``multimodal``, ``rope``, ``scatter_add``.

Two conventions this package depends on:

* **Import from the leaf module, not from here.** This file is an index, and the
  leaf modules do not import each other through it -- so a consumer that needs
  one node never drags in the MoE stack. The ``__all__`` below is for
  discoverability.
* **One module per component family.** A component with real behavior of its own
  belongs in its own file (as ``activation.py`` does for the gated activations)
  rather than being folded into a grab-bag module.
"""

from __future__ import annotations

from .activation import ActivationFn, SwiGLU
from .attention.qkv import QKVLinear, local_head_split
from .aux_loss import AuxLoss, collect_aux_loss_metrics, register_aux_loss_zero_hook
from .feed_forward import (
    FeedForward,
    SigmoidGatedFeedForward,
    compute_ffn_hidden_dim,
)
from .linear import PartialBiasRowwiseLinear, RouterGateLinear
from .moe.block import MoE
from .moe.experts import GroupedExperts
from .moe.load_balance import MicrobatchWiseLoadBalanceLoss
from .rope import ComplexRoPE, CosSinRoPE, RoPE, RoPEConfig

__all__ = [
    "ActivationFn",
    "AuxLoss",
    "collect_aux_loss_metrics",
    "ComplexRoPE",
    "compute_ffn_hidden_dim",
    "CosSinRoPE",
    "FeedForward",
    "GroupedExperts",
    "local_head_split",
    "MicrobatchWiseLoadBalanceLoss",
    "MoE",
    "PartialBiasRowwiseLinear",
    "QKVLinear",
    "register_aux_loss_zero_hook",
    "RoPE",
    "RoPEConfig",
    "RouterGateLinear",
    "SigmoidGatedFeedForward",
    "SwiGLU",
]
