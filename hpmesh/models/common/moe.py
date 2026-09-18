"""MoE layer: route tokens to experts, compute, combine.

Vendored from torchtitan ``models/common/moe.py`` -- ``RoutedExperts``,
``TokenChoiceTopKRouter`` and ``MoE``. What changed:

* nested ``Config`` dataclasses are gone; the modules are constructed directly.
* ``torch_remat`` is gone. Upstream wraps the routing decision in
  ``remat.region(..., recompute=False)``, which only tells the activation-
  checkpointing pass not to recompute it; calling ``_select_experts`` directly is
  the same arithmetic.
* the ``spmd_types`` blocks are gone (no runtime effect).
* ``MicrobatchWiseLoadBalanceLoss`` (the *auxiliary* load-balance loss) is not
  ported yet -- it needs the gradient-injection machinery in
  ``aux_loss.py``, and routing correctness does not depend on it. The
  auxiliary-loss-free bias path (``expert_bias_E``) is ported, since it is just
  a buffer an optimizer hook nudges.

Shape legend, scoped to this file: ``T`` = tokens, ``D`` = model dimension,
``E`` = experts, ``K`` = experts per token (top-k), ``e`` = local experts under
EP, ``R`` = routed tokens landing on this rank's experts, ``F`` = expert hidden
dimension.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grouped_experts import GroupedExperts
from .token_dispatcher import LocalTokenDispatcher

__all__ = ["TokenChoiceTopKRouter", "RoutedExperts", "MoE"]


class RouterGateLinear(nn.Module):
    """The router's projection: one score per expert, always in fp32.

    A plain ``nn.Linear`` would return the input dtype. Routing decisions are
    made on these scores, and a bf16 score can reorder a top-k on close calls,
    so the projection is computed in fp32 regardless of the model's dtype.

    Upstream reaches the same result through a custom autograd Function that also
    pins the backward GEMMs to fp32. The explicit casts here give the identical
    forward; autograd then derives the backward from those casts.

    Args:
        dim: model dimension (D).
        num_experts: number of experts (E).
    """

    def __init__(self, dim: int, num_experts: int) -> None:
        super().__init__()
        self.in_features = dim
        self.out_features = num_experts
        self.weight = nn.Parameter(torch.empty(num_experts, dim))

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return F.linear(x_TD.float(), self.weight.float())


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
    ) -> None:
        super().__init__()
        self.gate = RouterGateLinear(dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale

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
