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

from .activation import SwiGLU

__all__ = ["GroupedExperts"]


class GroupedExperts(nn.Module):
    """All experts' weights as three ``(E, O, I)`` tensors.

    Args:
        dim: model dimension (``D``).
        hidden_dim: expert hidden dimension (``F``).
        num_experts: number of experts (``E``).
        activation_fn: the gated activation; defaults to SwiGLU.
        use_grouped_mm: dispatch the expert GEMMs to ``torch._grouped_mm``
            instead of a per-expert loop. See ``_grouped_mm`` for why the
            default is the loop.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        *,
        activation_fn: nn.Module | None = None,
        use_grouped_mm: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.use_grouped_mm = use_grouped_mm
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

        Two implementations sit behind this one call, and the choice is a
        correctness decision, not a performance one:

        * ``torch._grouped_mm`` -- one kernel for all experts, the shape
          torchtitan's FSDP and MoE code is written against. It casts to bf16,
          which is what makes it fast.
        * a per-expert loop of ``F.linear`` -- slower, but it is the *same*
          arithmetic HF's own MoE does (v5's ``Qwen3MoeExperts.forward`` also
          loops with ``nn.functional.linear``), and it keeps the input dtype.
          That is what lets a swapped-in MoE be checked against the HF block it
          replaced on a machine with no GPU.

        The seam lives here so the low-precision variants can be swapped in
        without touching the forward, and so an FX tracer sees the op.
        """
        if self.use_grouped_mm:
            return torch._grouped_mm(
                x_TD.bfloat16(),
                weight_EOI.bfloat16().transpose(-2, -1),
                offs=offsets_E,
            )

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
