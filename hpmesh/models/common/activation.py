"""Activation functions used by the feed-forward and expert layers.

Vendored from torchtitan ``models/common/activation.py``. Two removals:

* The ``ActivationFn`` base class is gone. Upstream it exists to hang a
  ``Configurable.Config`` dataclass off, so a model config can name an
  activation by class; hpmesh constructs modules directly, so a plain
  ``nn.Module`` with the same call signature is enough.
* ``SiTUGLU`` is dropped -- it is Kimi-specific and nothing in hpmesh uses it.

The call signature is ``(gate, up)`` rather than a single tensor because these
activation the two halves of a fused gate-and-up projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SwiGLU"]


class SwiGLU(nn.Module):
    """``silu(gate) * up`` -- the standard gated activation.

    Both inputs have the same shape; the result matches, so it can be fed
    straight into the down projection.
    """

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up
