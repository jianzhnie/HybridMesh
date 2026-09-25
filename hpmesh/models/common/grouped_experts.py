"""Grouped experts: one weight tensor per projection, holding every expert.

Vendored from torchtitan ``models/common/moe.py`` (the ``GroupedExperts`` class
only). Removals: the nested ``Config`` dataclass, ``torch_remat``, and the
``spmd_types`` type-checking block in ``forward`` -- none of them affect the
arithmetic.

Shape legend (Noam Shazeer's convention, scoped to this file):

* ``E`` -- number of experts (a local shard of it under EP).
* ``D`` -- model dimension.
* ``F`` -- expert hidden dimension (the per-expert FFN width).
* ``R`` -- routed tokens assigned to the experts on this rank.
* ``O`` / ``I`` -- the grouped-GEMM weight's out/in feature roles. Kept as
  roles rather than model dims because the stored weights are ``(F, D)`` for
  the up/gate projections and ``(D, F)`` for the down one.

Weights are three stacked tensors rather than a ``ModuleList`` of small
modules. That is what makes the expert computation a single grouped GEMM
instead of E separate ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...accelerator.capabilities import has
from .activation import SwiGLU

__all__ = ["GroupedExperts"]


def _grouped_mm_available() -> bool:
    """Whether ``torch._grouped_mm`` can run here.

    The probe (run the op once on a real-shaped dummy -- a version or device
    test would be wrong on both counts) lives in the capability registry as
    ``torch_grouped_mm``; this wrapper keeps the local name the constructor
    and the tests use.
    """
    return has("torch_grouped_mm")


class GroupedExperts(nn.Module):
    """All experts' weights as three ``(E, O, I)`` tensors.

    Args:
        dim: model dimension (``D``).
        hidden_dim: expert hidden dimension (``F``).
        num_experts: number of experts (``E``).
        activation_fn: the gated activation; defaults to SwiGLU.
        use_grouped_mm: run the expert GEMMs as one ``torch._grouped_mm``
            instead of a per-expert loop. ``None`` (the default) decides by
            probing the op, which is what a run wants; pass a bool to pin it,
            which is what a test wants. See ``_grouped_mm`` for the trade-off.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        *,
        activation_fn: nn.Module | None = None,
        use_grouped_mm: bool | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.use_grouped_mm = (
            _grouped_mm_available() if use_grouped_mm is None else use_grouped_mm
        )
        self.activation_fn = activation_fn if activation_fn is not None else SwiGLU()

        # ``w1``/``w3`` are the gate and up projections, ``w2`` the down one.
        # Empty rather than random: the weights are filled by the caller (the
        # EP swap copies them out of the HF block; a from-scratch build would
        # run an initializer of its own).
        self.w1_EFD = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w3_EFD = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2_EDF = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Run every expert over the tokens routed to it.

        ``x_RD`` holds each expert's tokens contiguously, in expert order -- the
        dispatcher's job is to produce that layout. ``num_tokens_per_expert_E``
        gives the per-expert counts that delimit the segments.
        """
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)

        gate_RF = self._grouped_mm(x_RD, self.w1_EFD, offsets_E)
        up_RF = self._grouped_mm(x_RD, self.w3_EFD, offsets_E)
        h_RF = self.activation_fn(gate_RF, up_RF)
        return self._grouped_mm(h_RF, self.w2_EDF, offsets_E).type_as(x_RD)

    def _grouped_mm(
        self,
        x_TD: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``x @ weight_EOI.transpose(-2, -1)`` per expert.

        ``offsets_E`` is the exclusive cumulative token count (length ``E``),
        so segment ``e`` spans ``offsets_E[e-1] : offsets_E[e]`` -- exactly what
        ``torch._grouped_mm`` expects. Pass an empty tensor for expert 0.

        Two implementations sit behind this one call:

        * ``torch._grouped_mm`` -- one kernel for all experts, the shape
          torchtitan's MoE is written against, and the one taken whenever the op
          is available (``use_grouped_mm``) *and* the activations are already
          bf16 or narrower.
        * a per-expert loop of ``F.linear`` -- the fallback. It is the *same*
          arithmetic HF's own MoE does (v5's ``Qwen3MoeExperts.forward`` also
          loops with ``nn.functional.linear``), which is what makes a swapped-in
          MoE checkable against the HF block it replaced, and it keeps the
          input dtype.

        The dtype gate is not a nicety. ``torch._grouped_mm`` is bf16-only, so
        on an fp32 or fp64 model the fused path would round every expert input
        and weight to 8 mantissa bits: measured 1.4e-1 relative error on fp32
        and 1.3e0 on fp64, silently, which is exactly the class of change a
        numeric baseline cannot see. At or below bf16 the cast is the identity
        or a widening, so there is nothing to lose.
        """
        if self.use_grouped_mm and x_TD.dtype in (torch.bfloat16, torch.float16):
            return torch._grouped_mm(
                x_TD.bfloat16(),
                weight_EOI.bfloat16().transpose(-2, -1),
                offs=offsets_E,
            ).type_as(x_TD)

        counts = offsets_E.clone()
        counts[1:] = offsets_E[1:] - offsets_E[:-1]
        out_TD = x_TD.new_empty(x_TD.shape[0], weight_EOI.shape[1])
        start = 0
        for expert, count in enumerate(counts.tolist()):
            if count:
                end = start + count
                out_TD[start:end] = F.linear(x_TD[start:end], weight_EOI[expert])
                start = end
        return out_TD
