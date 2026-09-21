"""SwiGLU feed-forward layers.

Vendored from torchtitan ``models/common/feed_forward.py``. What was dropped is
the ``Module`` protocol and its nested ``Config`` dataclasses -- hpmesh builds
the module directly, so there is nothing to build from -- and ``torch_remat``.
Upstream wraps each projection in ``remat.region(..., recompute=...)``, which
only steers activation checkpointing; calling the projection directly is the
same arithmetic. The checkpoint hooks are kept: without them an external
checkpoint that stores the logical ``w1`` / ``w3`` projections cannot load.

This is the base class ``dist_gemm.DistGEMMFeedForward`` subclasses: that module
is the same network with the TP collectives folded into the two GEMMs, so it
overrides how the projections run and reuses the fused-weight layout contract
(``w13`` interleaved gate/up) and the activation split established here.

Shape suffix legend, scoped to this file:
  T = token dimensions, D = model dimension, F = feed-forward hidden dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .activation import ActivationFn, SwiGLU

__all__ = ["FeedForward", "SigmoidGatedFeedForward", "compute_ffn_hidden_dim"]


def compute_ffn_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 1,
    ffn_dim_multiplier: float | None = None,
) -> int:
    """Compute the SwiGLU hidden dimension for Llama3/4-style models.

    This applies the 2/3 scaling, optional multiplier, and rounds up to multiple_of.
    """
    hidden_dim = int(2 * 4 * dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class FeedForward(nn.Module):
    """SwiGLU feed-forward with one physical gate-and-up projection.

    ``w13`` stores the interleaved gate and up projections. The constructor
    takes the **final** hidden_dim (no internal 2/3 scaling). Use
    :func:`compute_ffn_hidden_dim` for Llama3/4-style dim computation.

    Args:
        w13: the fused gate-and-up projection, ``dim -> 2 * hidden_dim``.
        w2: the down projection, ``hidden_dim -> dim``.
        activation_fn: the gated activation; defaults to SwiGLU.
    """

    def __init__(
        self,
        *,
        w13: nn.Module,
        w2: nn.Module,
        activation_fn: ActivationFn | None = None,
    ) -> None:
        super().__init__()
        self.w13 = w13
        self.w2 = w2
        self.activation_fn = activation_fn if activation_fn is not None else SwiGLU()
        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Expose fused parameters under the logical w1/w3 checkpoint keys."""
        for param_name in ("weight", "bias"):
            fused_key = f"{prefix}w13.{param_name}"
            if fused_key not in state_dict:
                continue
            gate_up = state_dict.pop(fused_key).unflatten(0, (-1, 2))
            state_dict[f"{prefix}w1.{param_name}"] = gate_up[:, 0].contiguous()
            state_dict[f"{prefix}w3.{param_name}"] = gate_up[:, 1].contiguous()

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        """Pack logical w1/w3 checkpoint entries into the fused parameter."""
        for param_name in ("weight", "bias"):
            gate_key = f"{prefix}w1.{param_name}"
            up_key = f"{prefix}w3.{param_name}"
            if gate_key not in state_dict or up_key not in state_dict:
                continue
            state_dict[f"{prefix}w13.{param_name}"] = torch.stack(
                [state_dict.pop(gate_key), state_dict.pop(up_key)], dim=1
            ).flatten(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation_fn(*self._split_gate_up(self.w13(x))))

    def _split_gate_up(
        self, gate_up_TF: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the interleaved gate/up halves along the feature dim.

        The split is elementwise on feature-sharded activations, so it needs no
        collective even when the feature dim is TP-sharded. A subclass whose
        projections run fused collectives overrides ``forward`` but reuses this.
        """
        return gate_up_TF.unflatten(-1, (-1, 2)).unbind(-1)


class SigmoidGatedFeedForward(FeedForward):
    """SwiGLU feed-forward with a per-token sigmoid gate.

    The output is ``sigmoid(gate(x)) * ffn(x)``. It uses FeedForward's fused
    ``w13`` and ``w2`` projections and adds a separate ``gate`` projection.

    Args:
        w13: the fused gate-and-up projection, ``dim -> 2 * hidden_dim``.
        w2: the down projection, ``hidden_dim -> dim``.
        gate: the sigmoid gate projection, ``dim -> dim``.
        activation_fn: the gated activation; defaults to SwiGLU.
    """

    def __init__(
        self,
        *,
        w13: nn.Module,
        w2: nn.Module,
        gate: nn.Module,
        activation_fn: ActivationFn | None = None,
    ) -> None:
        super().__init__(w13=w13, w2=w2, activation_fn=activation_fn)
        self.gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_TD = super().forward(x)
        return torch.sigmoid(self.gate(x)) * out_TD
