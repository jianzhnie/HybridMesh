"""Model components shared across architectures.

Vendored from torchtitan ``models/common/``. The package is the model
vocabulary -- attention pieces, MoE, feed-forward, norms and activations, RoPE,
the multimodal glue.

Two conventions this package depends on:

* **Import from the leaf module, not from here.** This file is an index, and the
  leaf modules do not import each other through it -- so a consumer that needs
  one node never drags in ``token_dispatcher``. The ``__all__`` below is for
  discoverability.
* **One module per component family.** A component with real behavior of its own
  belongs in its own file (as ``activation.py`` does for the gated activations)
  rather than being folded into a grab-bag module.
"""

from __future__ import annotations

from .activation import ActivationFn, SwiGLU
from .aux_loss import AuxLoss, collect_aux_loss_metrics, register_aux_loss_zero_hook
from .feed_forward import (
    FeedForward,
    SigmoidGatedFeedForward,
    compute_ffn_hidden_dim,
)
from .grouped_experts import GroupedExperts
from .linear import PartialBiasRowwiseLinear, RouterGateLinear
from .moe import MicrobatchWiseLoadBalanceLoss, MoE
from .qkv import QKVLinear, local_head_split
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
