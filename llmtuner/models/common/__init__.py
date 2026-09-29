"""The model vocabulary, shared across architectures.

Vendored from torchtitan ``models/common/``: the pieces an HF text model is put
together from, with nothing in here knowing which family it belongs to. The two
families big enough to need more than one file are subpackages -- ``attention/``
(the fused QKV projection, the mask builders) and ``moe/`` (router, experts,
dispatcher, block, balance loss, balancing hooks) -- and the rest are one node
per file: ``activation``, ``async_linear`` (TP-overlapped GEMMs), ``aux_loss``
(the gradient carrier), ``cast_linear``, ``embedding``, ``feed_forward``,
``linear``, ``multimodal``, ``rope``, ``scatter_add``.

Two conventions this package depends on:

* **Import from the leaf module, not from here.** The index below is a
  convenience for callers that want several nodes at once (and the discoverability
  surface); the leaf modules never import each other through it.
* **One module per component family.** A component with real behavior of its own
  belongs in its own file (as ``activation.py`` does for the gated activations)
  rather than being folded into a grab-bag module.

The index is **lazy** (``__getattr__``, PEP 562): naming a node here must not
import it. An eager version would make ``import llmtuner.models.common.rope``
pull in the MoE stack, the fused QKV projection and the rest -- and on a torch
build without ``DTensor`` or flex attention that is the difference between "a
mask helper is importable" and "nothing in this package is".
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # The static view of the lazy table below, for type checkers and IDEs.
    from .activation import ActivationFn, SwiGLU
    from .attention.qkv import QKVLinear, local_head_split
    from .aux_loss import (
        AuxLoss,
        collect_aux_loss_metrics,
        register_aux_loss_zero_hook,
    )
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

#: Public name -> the leaf module that defines it.
_EXPORTS = {
    "ActivationFn": "activation",
    "AuxLoss": "aux_loss",
    "collect_aux_loss_metrics": "aux_loss",
    "ComplexRoPE": "rope",
    "compute_ffn_hidden_dim": "feed_forward",
    "CosSinRoPE": "rope",
    "FeedForward": "feed_forward",
    "GroupedExperts": "moe.experts",
    "local_head_split": "attention.qkv",
    "MicrobatchWiseLoadBalanceLoss": "moe.load_balance",
    "MoE": "moe.block",
    "PartialBiasRowwiseLinear": "linear",
    "QKVLinear": "attention.qkv",
    "register_aux_loss_zero_hook": "aux_loss",
    "RoPE": "rope",
    "RoPEConfig": "rope",
    "RouterGateLinear": "linear",
    "SigmoidGatedFeedForward": "feed_forward",
    "SwiGLU": "activation",
}

# Written out (rather than ``sorted(_EXPORTS)``) so that the type-only imports
# above read as re-exports to linters; a test pins the two lists together.
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


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
