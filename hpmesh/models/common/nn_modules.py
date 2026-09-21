"""The ``nn`` modules a ``models/common`` model composes, under one vocabulary.

Vendored from torchtitan ``models/common/nn_modules.py``. Upstream each class
used diamond inheritance (``nn.X`` + ``Module``) so it could carry a config and
be sharded by the parallelism engine; hpmesh has neither protocol, and the
constructions here are all plain ``nn.X`` calls, so almost nothing is left over.

Be clear about what this file is: with the protocol removed, these ARE the
stock ``nn`` modules. The value is having one spelling for the components a
``models/common`` model is built from -- a model assembled out of
``feed_forward`` / ``moe`` / ``rope`` does not have to reach into ``torch.nn``
for its norms and activations and hope it picked the same variant the rest of
the stack did. Anything here that would need real behavior belongs in its own
module (as ``activation.py`` does for the gated activations), not added here.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = [
    "GELU",
    "GroupNorm",
    "Identity",
    "LayerNorm",
    "RMSNorm",
    "SiLU",
]


class Identity(nn.Identity):
    """``nn.Identity`` -- a placeholder in a module list that must keep its index."""


class LayerNorm(nn.LayerNorm):
    """``nn.LayerNorm``: the epsilon is inside the variance, as HF computes it."""


class RMSNorm(nn.RMSNorm):
    """``nn.RMSNorm``: eps added after the mean-square, matching HF's RMSNorm."""


class GELU(nn.GELU):
    """``nn.GELU``; ``approximate`` selects exact or the tanh approximation."""

    def __init__(self, approximate: str = "none") -> None:
        super().__init__(approximate=approximate)


class SiLU(nn.SiLU):
    """``nn.SiLU`` -- the ungated activation, not the SwiGLU used by the FFN."""


class GroupNorm(nn.GroupNorm):
    """``nn.GroupNorm``, used by the vision towers rather than by the decoder."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__(num_groups, num_channels, eps=eps)
