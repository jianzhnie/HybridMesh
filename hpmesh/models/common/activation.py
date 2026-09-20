"""Activation functions used by the feed-forward and expert layers.

Vendored from torchtitan ``models/common/activation.py``. Two changes:

* The ``Function`` base is gone. Upstream it exists so a model config can name
  an activation by class and have it built; hpmesh constructs modules directly,
  so a plain ``nn.Module`` with the same call signature is enough.
* ``ActivationFn`` survives as an ``ABC`` -- it is the shared type for
  "consumes the gate and up halves of a fused projection", which lets a caller
  swap activations without caring which one it holds.

The call signature is ``(gate, up)`` rather than a single tensor because these
consume the two halves of a fused gate-and-up projection separately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ActivationFn", "SiTUGLU", "SwiGLU"]


class ActivationFn(nn.Module, ABC):
    """A gated activation: ``(gate, up)`` in, one tensor out.

    Both inputs have the same shape, and so does the result, so it can be fed
    straight into the down projection.
    """

    @abstractmethod
    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Combine the gate and up halves of a fused projection."""
        raise NotImplementedError


class SwiGLU(ActivationFn):
    """``silu(gate) * up`` -- the standard gated activation."""

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up


class SiTUGLU(ActivationFn):
    """Kimi's SiTU-GLU activation, evaluated in FP32.

    ``beta`` bounds the gate through a tanh, which keeps the activation from
    saturating as the gate grows. ``linear_beta`` applies the same bound to the
    up half; leaving it ``None`` bounds the gate only.

    The FP32 promotion is not incidental: ``tanh`` and ``sigmoid`` on bf16
    inputs lose enough precision to move the loss, so the compute runs wide and
    casts back on the way out.
    """

    def __init__(self, *, beta: float = 1.0, linear_beta: float | None = None) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        input_dtype = gate.dtype
        gate_f = gate.float()
        up_f = up.float()
        gate_f = self.beta * torch.tanh(gate_f / self.beta) * torch.sigmoid(gate_f)
        if self.linear_beta is not None:
            up_f = self.linear_beta * torch.tanh(up_f / self.linear_beta)
        return (gate_f * up_f).to(input_dtype)
