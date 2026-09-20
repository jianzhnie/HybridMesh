"""MoE layer: route tokens to experts, compute, combine.

Vendored from torchtitan ``models/common/moe.py`` -- ``RoutedExperts``,
``TokenChoiceTopKRouter``, ``MoE`` and ``MicrobatchWiseLoadBalanceLoss``. What
changed:

* nested ``Config`` dataclasses are gone; the modules are constructed directly.
* ``torch_remat`` is gone. Upstream wraps the routing decision in
  ``remat.region(..., recompute=False)``, which only tells the activation-
  checkpointing pass not to recompute it; calling ``_select_experts`` directly is
  the same arithmetic.
* ``RouterGateLinear`` moved to ``linear.py`` and now pins the *backward* GEMMs to
  fp32 too, through the autograd Function upstream uses (this file previously
  rested with an explicit-cast forward that matched it only in precision, not in
  dtype). The router imports it from there, as upstream does.
* the ``spmd_types`` blocks are gone (no runtime effect), and
  ``MicrobatchWiseLoadBalanceLoss``'s Partial -> Invariant reduction is a plain
  autograd-aware ``all_reduce`` (see its ``_reduce_token_partials``) instead of
  ``spmd.redistribute``. The arithmetic is the same: an all-reduce forward with
  an all-reduce backward.
* ``MicrobatchWiseLoadBalanceLoss`` is ported and wired: the router runs it
  on each training forward through its ``aux_loss`` slot, matching upstream.

Shape legend, scoped to this file: ``T`` = tokens, ``D`` = model dimension,
``E`` = experts, ``K`` = experts per token (top-k), ``e`` = local experts under
EP, ``R`` = routed tokens landing on this rank's experts, ``F`` = expert hidden
dimension.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F

from hpmesh.utils.spmd_context import spmd_mesh_group, spmd_sparse_mesh

from .aux_loss import AuxLoss
from .grouped_experts import GroupedExperts
from .linear import RouterGateLinear
from .token_dispatcher import LocalTokenDispatcher

__all__ = [
    "MoE",
    "MicrobatchWiseLoadBalanceLoss",
    "RoutedExperts",
    "TokenChoiceTopKRouter",
]


class TokenChoiceTopKRouter(nn.Module):
    """Token-choice top-K routing: each token picks its own K experts.

    Args:
        num_experts: total experts (E).
        dim: model dimension (D).
        top_k: experts per token (K).
        score_func: how the raw gate logits become scores -- ``"sigmoid"``,
            ``"softmax"``, or ``"sqrtsoftplus"``.
        route_norm: renormalize the selected K scores to sum to 1.
        route_scale: multiply the final scores, e.g. DeepSeek-V3's
            ``routed_scaling_factor``.
        aux_loss: an optional ``AuxLoss`` (e.g.
            ``MicrobatchWiseLoadBalanceLoss``) run on the scores each training
            forward; its gradient is injected on the top-k scores' backward
            path. ``None`` disables it.
    """

    def __init__(
        self,
        num_experts: int,
        dim: int,
        top_k: int = 1,
        *,
        score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sigmoid",
        route_norm: bool = False,
        route_scale: float = 1.0,
        aux_loss: AuxLoss | None = None,
    ) -> None:
        super().__init__()
        self.gate = RouterGateLinear(dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.aux_loss = aux_loss

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Top-k expert ids, using the load-balancing bias on top of the scores.

        ``sorted=False`` matches upstream: the ids come back in top-k order but
        the scores are gathered separately, so no ordering is relied on.
        """
        scores_for_choice_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        )
        return torch.topk(
            scores_for_choice_TE, k=self.top_k, dim=-1, sorted=False
        ).indices

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_TD: input tokens ``(T, D)``.
            expert_bias_E: optional load-balancing bias ``(E,)``. It shifts which
                experts win but not the score a token carries to them.

        Returns:
            topk_scores_TK: routing scores ``(T, K)``.
            topk_expert_ids_TK: expert indices ``(T, K)``.
            routing_map_TE: one-hot boolean map ``(T, E)``.
        """
        scores_TE = self.gate(x_TD)

        # Done in fp32 (the gate already returns fp32); a sigmoid over low
        # precision scores can underflow and collapse the routing.
        if self.score_func == "sigmoid":
            scores_TE = torch.sigmoid(scores_TE)
        elif self.score_func == "softmax":
            scores_TE = F.softmax(scores_TE, dim=-1)
        elif self.score_func == "sqrtsoftplus":
            scores_TE = F.softplus(scores_TE).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        topk_expert_ids_TK = self._select_experts(scores_TE, expert_bias_E)
        # The bias only picks experts; the weight a token carries is the score of
        # the expert it actually landed on, bias excluded.
        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)

        if self.route_norm:
            denominator = topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_TK = topk_scores_TK / denominator
        topk_scores_TK = topk_scores_TK * self.route_scale

        # One-hot map marking each token's chosen experts. Built by scatter so
        # a token choosing the same expert twice (route_norm edge case) still
        # counts once.
        routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1,
            topk_expert_ids_TK,
            True,
        )

        # The aux loss reads the pre-topk scores and the routing map; its
        # gradient rides back on the top-k scores (identity forward, so the
        # routing arithmetic is unchanged). Training only: an eval forward has
        # no backward to inject into, and no step denominator is set there.
        if self.training and self.aux_loss is not None:
            topk_scores_TK = self.aux_loss(
                scores_TE,
                routing_map_TE,
                carrier=topk_scores_TK,
            )

        return topk_scores_TK, topk_expert_ids_TK, routing_map_TE


class RoutedExperts(nn.Module):
    """The dispatch/combine pair wrapped around the grouped expert weights.

    Split out from ``MoE`` so the routing (which is model-specific) and the
    expert compute (which is not) can vary independently.

    Args:
        grouped_experts: the expert weights.
        dispatcher: moves tokens to the ranks holding their experts.
    """

    def __init__(
        self,
        grouped_experts: GroupedExperts,
        dispatcher: LocalTokenDispatcher,
    ) -> None:
        super().__init__()
        self.inner_experts = grouped_experts
        self.token_dispatcher = dispatcher

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, run them, and combine the results."""
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        routed_output_RD = self.inner_experts(
            routed_input_RD, num_global_tokens_per_local_expert_e
        )
        return self.token_dispatcher.combine(
            routed_output_RD,
            metadata,
            x_TD,
        )


class MoE(nn.Module):
    """A mixture-of-experts block.

    ``forward`` runs: route -> dispatch -> expert compute -> combine -> (shared
    experts) -> sum. With EP the dispatch and combine halves are all-to-alls; at
    EP=1 they are local reorderings and the arithmetic is unchanged.

    Args:
        num_experts: total experts (E).
        routed_experts: the dispatch/combine + expert-weight bundle.
        router: decides each token's experts.
        load_balance_coeff: strength of the auxiliary-loss-free bias update, or
            ``None`` to disable it. The bias is updated outside the model, by an
            optimizer hook, so it survives gradient accumulation.
        shared_experts: an optional dense FFN every token passes through.
    """

    def __init__(
        self,
        num_experts: int,
        routed_experts: RoutedExperts,
        router: TokenChoiceTopKRouter,
        *,
        load_balance_coeff: float | None = 1e-3,
        shared_experts: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.routed_experts = routed_experts
        self.router = router
        self.shared_experts = shared_experts
        self.load_balance_coeff = load_balance_coeff

        # Auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664):
        # a per-expert bias nudged by observed load, updated by an optimizer hook
        # so it sees a whole step rather than one microbatch.
        if load_balance_coeff is not None:
            if load_balance_coeff <= 0.0:
                raise ValueError(
                    f"load_balance_coeff must be positive, got {load_balance_coeff}"
                )
            self.register_buffer(
                "expert_bias_E",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias_E = None
        # Expert usage counters. Non-persistent: they are scratch, not checkpoint
        # state.
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route, run the experts, and sum the routed and shared outputs.

        Accepts ``(T, D)`` or any leading-dimension form ``(..., T, D)``. The
        HF decoder layer calls its MoE as ``self.mlp(hidden_states)`` with a
        ``(batch, seq, D)`` tensor, so a swapped-in block has to take that shape
        and give it back; the routing itself works on flattened tokens.
        """
        if x.dim() > 2:
            lead = x.shape[:-1]
            out = self._forward_tokens(x.reshape(-1, x.shape[-1]))
            return out.reshape(*lead, out.shape[-1])
        return self._forward_tokens(x)

    def _forward_tokens(self, x_TD: torch.Tensor) -> torch.Tensor:
        """The MoE computation over a flat ``(T, D)`` token stream."""
        # (T, K) scores and ids; (T, E) map of which experts each token picked.
        (
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
        ) = self.router(x_TD, self.expert_bias_E)
        num_local_tokens_per_expert_E = routing_map_TE.sum(dim=0)

        if self.training:
            with torch.no_grad():
                # NOTE: activation checkpointing runs the forward twice, so this
                # counts a token twice on recompute. The bias update uses
                # sign(), so the doubled count does not change its direction.
                self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        out_TD = self.routed_experts(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )

        if self.shared_experts is not None:
            out_TD = out_TD + self.shared_experts(x_TD)
        return out_TD


class MicrobatchWiseLoadBalanceLoss(AuxLoss):
    """Per-forward MoE load-balance gradient (DeepSeek-V3 Sec 2.1.2 Eqs 17-20).

    The balancing unit is one forward's folded token stream (a DP-local
    microbatch). Global (corpus-level) balance is left to the
    auxiliary-loss-free bias path (``expert_bias_E``); this loss only
    discourages extreme load imbalance within individual forwards (samples),
    per the DeepSeek-V3 design ("Complementary Sequence-Wise Auxiliary Loss").

    With ``E`` experts, top-``K`` selection and ``T`` tokens per forward:

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
        # "batch" (dp) rather than "loss" (dp+cp): the value is already the
        # same on every CP coordinate by construction, so summing over cp too
        # would count it more than once per layer.
        super().__init__(coeff=coeff, reduce_mesh="batch")

    def _reduce_token_partials(
        self, partial_E: torch.Tensor, axes: tuple[str, ...]
    ) -> torch.Tensor:
        """Partial -> Invariant all-reduce over the token-partition axes.

        An all-reduce in forward with an all-reduce backward: the reduced sums,
        and hence the loss and its gradient, are identical on every rank of the
        group, so each rank's backward contributes the same gradient and the
        sum of those contributions is what the router needs.

        Axes are resolved by name through ``spmd_mesh_group``, so no DeviceMesh
        escapes into model code and an inactive axis is skipped rather than run
        as a size-1 no-op collective. A ``None`` group means the axis is not
        active -- either the axis is size 1, or no SPMD mesh has been registered
        for this process. The latter is the case for hpmesh today: the trainer
        does not call ``set_spmd_meshes``, so this degrades to no reduction,
        which is correct while nothing shards the token dim.
        """
        for axis in axes:
            group = spmd_mesh_group(axis)
            if group is None:
                continue
            partial_E = dist_nn.all_reduce(partial_E, op=dist.ReduceOp.SUM, group=group)
        return partial_E

    def forward(
        self,
        scores_TE: torch.Tensor,
        routing_map_TE: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-forward balance loss and inject its gradient.

        Args:
            scores_TE: router scores ``(T, E)`` for the forward's tokens.
            routing_map_TE: one-hot routing map ``(T, E)`` for the same tokens,
                as counted by the router.
            carrier: tensor whose backward path carries the injected gradient
                (the router's top-k scores).

        Returns:
            ``carrier`` unchanged (identity forward).
        """
        # DP is deliberately not reduced: each DP rank owns an independent
        # token stream, so only the axes that shard one stream are summed over.
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
        # are non-negative, so the norm is a plain sum.
        probs_TE = F.normalize(scores_TE, p=1, dim=-1)
        p_E = self._reduce_token_partials(probs_TE.sum(dim=0), axes)

        # Eq. 17: L_bal = sum_i f_i * p_i
        loss = (f_E * p_E).sum()
        return self.inject(loss, carrier=carrier)
