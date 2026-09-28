"""The per-forward MoE load-balance loss.

One node of the ``moe`` package: ``MicrobatchWiseLoadBalanceLoss`` is the
DeepSeek-V3 sequence-wise auxiliary loss, run by a router on each training
forward through its ``aux_loss`` slot (the router owns the slot; this file owns
the loss). Corpus-level balance is the separate auxiliary-loss-free path in
``block.py``.

Vendored from torchtitan ``models/common/moe.py``. What changed: the
``spmd_types`` blocks are gone (no runtime effect), and the Partial -> Invariant
reduction is the ``_PartialToInvariantAllReduce`` autograd Function (all-reduce
forward, identity backward) instead of ``spmd.redistribute`` -- the same
semantics, which ``torch.distributed.nn.all_reduce`` would NOT give: its
backward is a second all-reduce, multiplying the injected gradient by the group
size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llmtuner.accelerator.dist import all_reduce
from llmtuner.accelerator.spmd_context import spmd_mesh_group, spmd_sparse_mesh

from ..aux_loss import AuxLoss

__all__ = ["MicrobatchWiseLoadBalanceLoss"]


class _PartialToInvariantAllReduce(torch.autograd.Function):
    """All-reduce in forward, identity in backward (Partial -> Invariant).

    The reduced sum is identical on every rank of the group, and every rank
    computes the same downstream loss from it, so the gradient of that loss
    w.r.t. one rank's partial equals the gradient w.r.t. the sum itself
    (``d sum / d partial = 1``). ``torch.distributed.nn.all_reduce`` instead
    all-reduces on the backward too, summing every rank's identical gradient
    and multiplying what reaches the router by the group size.
    """

    @staticmethod
    def forward(ctx, partial_E, group):  # pyrefly: ignore[bad-override]
        reduced_E = partial_E.clone()
        all_reduce(reduced_E, group=group)
        return reduced_E

    @staticmethod
    def backward(ctx, grad_out):  # pyrefly: ignore[bad-override]
        return grad_out, None


class MicrobatchWiseLoadBalanceLoss(AuxLoss):
    """Per-forward MoE load-balance gradient (DeepSeek-V3 Sec 2.1.2 Eqs 17-20).

    The balancing unit is one forward's folded token stream (a DP-local
    microbatch). Global (corpus-level) balance is left to the
    auxiliary-loss-free bias path (``expert_bias_E``); this loss only
    discourages extreme load imbalance within individual forwards (samples),
    per the DeepSeek-V3 design ("Complementary Sequence-Wise Auxiliary Loss").

    With ``E`` experts, top-``K`` selection and ``T`` valid tokens per forward:

    Eq. 18: ``f_i = (E / (K T)) * sum_t 1[token t routes to expert i]``
    Eq. 19: ``p_i = (1 / T) * sum_t s'_t,i``, where
            ``s'_t,i = s_t,i / sum_j s_t,j`` is the per-token normalized score.
    Eq. 17: ``L_bal = sum_i f_i * p_i``

    The value returned through ``inject`` is ``T * L_bal``: Eqs 17-20 define a
    per-token-normalized value, while ``AuxLoss`` scales every auxiliary loss by
    ``1 / global_valid_tokens``, so the sum-type form keeps the injected weight
    at ``coeff * L_bal``.

    The counts (Eq. 18) and normalized-score sums (Eq. 19) are sums over the
    folded token dim, hence Partial over the mesh axes that shard it. They are
    all-reduced before the formula so every rank computes the same per-forward
    loss. The one-hot counts are non-differentiable: the gradient reaches the
    router only through the normalized-score sums and the top-k score carrier.

    ``T`` never appears explicitly: Eq. 18 is evaluated in the T-free form
    ``f_i = E * counts_i / sum_j counts_j``, which equals ``(E / (K T)) *
    counts_i`` because each token contributes K entries, so ``sum_j counts_j =
    K T``. That needs no shape or mesh-degree assumption and follows any
    masking the router applies to the routing map.

    Args:
        coeff: scales the injected gradient, as for any ``AuxLoss``.
    """

    def __init__(self, *, coeff: float) -> None:
        # "batch" (dp) rather than "loss" (dp, cp and tp): the value is already
        # the same on every CP coordinate by construction -- and on every TP
        # one, since llmtuner's TP is sequence-parallel -- so summing over
        # either would count it more than once per layer.
        super().__init__(coeff=coeff, reduce_mesh="batch")

    def _reduce_token_partials(
        self, partial_E: torch.Tensor, axes: tuple[str, ...]
    ) -> torch.Tensor:
        """Partial -> Invariant all-reduce over the token-partition axes.

        An all-reduce in forward with an identity backward: the reduced sums,
        and hence the loss and its gradient, are identical on every rank of the
        group, and each rank's local partial is one summand of them, so the
        gradient w.r.t. the partial is the gradient w.r.t. the sum.

        Axes are resolved by name through ``spmd_mesh_group``, so no DeviceMesh
        escapes into model code and an inactive axis is skipped rather than run
        as a size-1 no-op collective. A ``None`` group means the axis is not
        active -- either the axis is size 1, or no SPMD mesh is registered for
        this process (the trainer registers one via ``spmd_context``; a bare
        single-process run has none, and no reduction is correct there).
        """
        for axis in axes:
            group = spmd_mesh_group(axis)
            if group is None:
                continue
            partial_E = _PartialToInvariantAllReduce.apply(partial_E, group)
        return partial_E

    def forward(
        self,
        scores_TE: torch.Tensor,
        routing_map_TE: torch.Tensor,
        *,
        carrier: torch.Tensor,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the per-forward balance loss and inject its gradient.

        Args:
            scores_TE: router scores ``(T, E)`` for the forward's tokens.
            routing_map_TE: one-hot routing map ``(T, E)`` for the same tokens,
                as counted by the router -- padding-filtered when the router
                was given a mask.
            carrier: tensor whose backward path carries the injected gradient
                (the router's top-k scores).
            padding_mask_T: optional boolean ``(T,)`` mask, true for padding.
                Padding rows contribute nothing to the normalized-score sum
                (Eq. 19), so ``T`` in both equations is the VALID token count.

        Returns:
            ``carrier`` unchanged (identity forward).
        """
        # DP is deliberately not reduced: each DP rank owns an independent
        # token stream, so only the axes that shard one stream are summed over.
        # ``tp`` belongs under EP for the same reason as upstream: EP borrows
        # ranks from TP, so an EP'd MoE sees a tp-partial token stream. Under
        # MoE-under-TP (tp>1, ep=1) no reduction is needed: the block-boundary
        # all-gather means the router already sees the full token stream, and
        # the aux loss only exists on the llmtuner MoE stack, which the TP path
        # does not install.
        axes = ("cp", "tp") if spmd_sparse_mesh() is not None else ("cp",)

        # Eq. 18: per-expert routing counts, then f_i = E * counts_i /
        # sum_j counts_j, so sum_i f_i = E. The map is cast to float before the
        # sum because a bool tensor has no gradient path and a Partial cast is
        # non-linear under spmd_types.
        counts_E = self._reduce_token_partials(
            routing_map_TE.to(scores_TE.dtype).sum(dim=0), axes
        )
        f_E = F.normalize(counts_E, p=1, dim=0) * scores_TE.size(-1)

        # Eq. 19: p_i = (1/T) sum_t s'_t,i, the per-token L1-normalized scores.
        # F.normalize's eps clamp only guards an all-zero score row: the scores
        # are non-negative, so the norm is a plain sum. Padding rows are zeroed
        # after the per-token normalization, dropping them from the sum.
        probs_TE = F.normalize(scores_TE, p=1, dim=-1)
        if padding_mask_T is not None:
            probs_TE = probs_TE * ~padding_mask_T.unsqueeze(-1)
        p_E = self._reduce_token_partials(probs_TE.sum(dim=0), axes)

        # Eq. 17: L_bal = sum_i f_i * p_i
        loss = (f_E * p_E).sum()
        return self.inject(loss, carrier=carrier)
