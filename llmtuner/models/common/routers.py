"""Router modules for the MoE stack: token-choice top-K routing.

Split out of ``moe.py`` (which keeps the MoE block itself and the aux-loss):
``TokenChoiceTopKRouter`` is the base router, ``QuantileBalancedTopKRouter``
the quantile-balanced variant, and ``QuantileBalancer`` its per-step bias
estimator. Nothing here knows about EP dispatch or the bias-update hooks --
those live in ``moe.py`` and ``balancing.py`` respectively.

Shape legend matches ``moe.py``: ``T`` = tokens, ``D`` = model dimension,
``E`` = experts, ``K`` = experts per token, ``B`` = quantile-histogram bins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import RouterGateLinear

if TYPE_CHECKING:
    from .aux_loss import AuxLoss


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
        num_expert_groups: split the ``E`` experts into this many contiguous
            groups and restrict each token's top-K to ``num_limited_groups`` of
            them -- DeepSeek-V3's node-limited routing, where a group is a node
            and the restriction caps inter-node all-to-all. ``None`` disables
            grouping and leaves plain top-K over all ``E``.
        num_limited_groups: how many groups a token may draw from. Required
            when ``num_expert_groups`` is set.
        aux_loss: an optional ``AuxLoss`` (e.g.
            ``MicrobatchWiseLoadBalanceLoss``) run on the scores each training
            forward; its gradient is injected on the top-k scores' backward
            path. ``None`` disables it.
        _debug_force_load_balance: replace the routing decision with a
            round-robin assignment that lands exactly the same number of tokens
            on every expert, so a load-imbalance bug can be told apart from a
            bias/score bug. Debug only: the gate still runs and its scores are
            gathered for the chosen experts, but nothing about them (or the
            bias, or the group restriction) influences the choice.
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
        num_expert_groups: int | None = None,
        num_limited_groups: int | None = None,
        aux_loss: AuxLoss | None = None,
        _debug_force_load_balance: bool = False,
    ) -> None:
        super().__init__()
        if num_expert_groups is not None:
            if num_limited_groups is None:
                raise ValueError(
                    "num_limited_groups must be set when num_expert_groups is set"
                )
            if num_limited_groups > num_expert_groups:
                raise ValueError(
                    f"num_limited_groups ({num_limited_groups}) cannot exceed "
                    f"num_expert_groups ({num_expert_groups})"
                )
            if num_experts % num_expert_groups != 0:
                raise ValueError(
                    f"num_experts ({num_experts}) must be divisible by "
                    f"num_expert_groups ({num_expert_groups})"
                )
            # The group score is the sum of each group's top-2 expert scores,
            # so a one-expert group has no second score to add.
            if num_experts // num_expert_groups < 2:
                raise ValueError(
                    f"num_experts_per_group ({num_experts // num_expert_groups}) "
                    "must be >= 2 to form a group score"
                )
        self.gate = RouterGateLinear(dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.num_expert_groups = num_expert_groups
        self.num_limited_groups = num_limited_groups
        self.aux_loss = aux_loss
        self._debug_force_load_balance = _debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores_TE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.

        Token ``t``'s ``k``-th slot gets expert ``(t * K + k) % E``, so over a
        folded token stream every expert wins exactly ``ceil``/``floor`` of
        ``T * K / E`` slots regardless of the scores. The gating *value* still
        comes from the real scores (gathered, bias excluded), matching the
        normal path -- only the choice is forced.

        Returns expert ids and scores, both ``(T, K)``.
        """
        num_tokens = scores_TE.shape[0]
        topk_expert_ids_TK = (
            torch.arange(
                num_tokens * self.top_k,
                device=scores_TE.device,
                dtype=torch.int64,
            ).reshape(num_tokens, self.top_k)
            % self.num_experts
        )
        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)
        return topk_expert_ids_TK, topk_scores_TK

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Top-k expert ids, using the load-balancing bias on top of the scores.

        ``sorted=False`` matches upstream: the ids come back in top-k order but
        the scores are gathered separately, so no ordering is relied on.

        ``padding_mask_T`` is accepted so subclasses (the quantile router) can
        filter their observations to valid tokens; the choice itself never
        depends on it -- padding tokens route like any other token.
        """
        del padding_mask_T
        scores_for_choice_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        )
        if self.num_expert_groups is None:
            return torch.topk(
                scores_for_choice_TE, k=self.top_k, dim=-1, sorted=False
            ).indices
        return self._select_experts_within_groups(scores_for_choice_TE)

    def _select_experts_within_groups(
        self, scores_for_choice_TE: torch.Tensor
    ) -> torch.Tensor:
        """Node-limited top-K: pick the top groups first, then the top experts.

        A group's score is the sum of its two highest expert scores (DeepSeek-V3
        Sec 2.1.1), so the groups that win are the ones with a strong pair of
        experts in them rather than a single lucky one.

        Everything runs on ``scores_for_choice`` -- the sigmoid scores with the
        load-balancing bias already added -- because the bias is what steers
        which experts win. The caller still gathers the routing *weight* from
        the unbiased scores, so the bias shifts the choice and never the value.

        ``E`` is laid out as ``num_expert_groups`` contiguous runs of equal size,
        which is what makes the restriction a statement about where the experts
        physically live.
        """
        assert self.num_expert_groups is not None
        assert self.num_limited_groups is not None
        num_experts_per_group = self.num_experts // self.num_expert_groups

        scores_TGP = scores_for_choice_TE.unflatten(
            -1, (self.num_expert_groups, num_experts_per_group)
        )
        group_scores_TG = scores_TGP.topk(2, dim=-1).values.sum(dim=-1)
        selected_group_ids_TL = torch.topk(
            group_scores_TG, k=self.num_limited_groups, dim=-1, sorted=False
        ).indices

        unselected_groups_TG = torch.ones_like(group_scores_TG, dtype=torch.bool)
        unselected_groups_TG.scatter_(-1, selected_group_ids_TL, False)
        # ``-inf``, which is what torchtitan (``models/deepseek_v3/moe.py``)
        # uses and what HF's DeepSeek-V3, GLM4 and OLMoE use. HF's DeepSeek-V2
        # and Mistral4 are the exceptions: they mask to ``0.0``, which breaks
        # once the load-balancing bias can push ``scores_for_choice`` negative
        # -- a masked 0.0 then outranks a real expert inside a selected group.
        # V2 never reaches this code (its ``topk_method`` is "greedy", so
        # ``swap.py`` builds it no group restriction), but matching the majority
        # spelling is the right default for the ones that do.
        scores_for_choice_TE = scores_TGP.masked_fill(
            unselected_groups_TG.unsqueeze(-1), float("-inf")
        ).flatten(-2)
        return torch.topk(
            scores_for_choice_TE, k=self.top_k, dim=-1, sorted=False
        ).indices

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_TD: input tokens ``(T, D)``.
            expert_bias_E: optional load-balancing bias ``(E,)``. It shifts which
                experts win but not the score a token carries to them.
            padding_mask_T: optional boolean ``(T,)`` mask, true for padding.
                Padding tokens are routed, dispatched and computed like any
                other token; the mask filters only the load-balancing
                statistics (the aux loss's f/p terms and, for the quantile
                router, the histogram observation).

        Returns:
            topk_scores_TK: routing scores ``(T, K)``.
            topk_expert_ids_TK: expert indices ``(T, K)``.
            routing_map_TE: one-hot boolean map ``(T, E)``, over ALL tokens --
                the dispatch contract. The masked view used for statistics is
                built inside and never leaves the router.
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

        if padding_mask_T is not None:
            if padding_mask_T.dtype != torch.bool:
                raise ValueError(
                    "padding_mask_T must have dtype bool, "
                    f"got {padding_mask_T.dtype}."
                )
            if padding_mask_T.shape != scores_TE.shape[:-1]:
                raise ValueError(
                    "padding_mask_T must have shape matching the routing-map "
                    f"token axis, got {tuple(padding_mask_T.shape)} for scores "
                    f"{tuple(scores_TE.shape)}."
                )

        if self._debug_force_load_balance:
            # The bias and the group restriction are both bypassed: the point
            # of the flag is a routing decision nothing downstream can skew.
            (
                topk_expert_ids_TK,
                topk_scores_TK,
            ) = self._debug_force_load_balance_routing(scores_TE)
        else:
            topk_expert_ids_TK = self._select_experts(
                scores_TE, expert_bias_E, padding_mask_T=padding_mask_T
            )
            # The bias only picks experts; the weight a token carries is the
            # score of the expert it actually landed on, bias excluded.
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

        # The aux loss reads the pre-topk scores and a padding-filtered view
        # of the routing map; its gradient rides back on the top-k scores
        # (identity forward, so the routing arithmetic is unchanged). Training
        # only: an eval forward has no backward to inject into, and no step
        # denominator is set there. The full map is what is returned: dispatch
        # counts every token, statistics count valid ones.
        if self.training and self.aux_loss is not None:
            masked_routing_map_TE = (
                routing_map_TE
                if padding_mask_T is None
                else routing_map_TE & ~padding_mask_T.unsqueeze(-1)
            )
            topk_scores_TK = self.aux_loss(
                scores_TE,
                masked_routing_map_TE,
                carrier=topk_scores_TK,
                padding_mask_T=padding_mask_T,
            )

        return topk_scores_TK, topk_expert_ids_TK, routing_map_TE


class QuantileBalancedTopKRouter(TokenChoiceTopKRouter):
    """Top-k router balanced by a histogram-estimated quantile bias.

    Ported from torchtitan's quantile-balanced routing (Kimi K3 technical
    report, Sec 2.3.3 and Appendix D). Each training forward computes a biased
    Top-(K+1) once: the first K experts route the token, and the (K+1)-th
    biased score is the cutoff a *required* expert bias is measured against.
    ``QuantileBalancer`` accumulates those required biases into a histogram,
    and ``register_moe_quantile_balancing_hook`` turns the histogram into the
    next mean-centred ``expert_bias_E`` once per optimizer step.

    The routing *weight* still comes from the original unbiased scores --
    the bias only shifts which experts win, as in the base router.

    The scheme is defined over sigmoid scores in ``[0, 1]`` (the histogram
    range derives from that bound), so the score function is pinned to
    sigmoid and the constructor takes no ``score_func``. Node-limited routing
    and ``_debug_force_load_balance`` are likewise not offered: the quantile
    update assumes a free Top-(K+1) over all experts, and a forced round-robin
    would make the cutoff meaningless.

    Args:
        num_bins: histogram resolution for the quantile estimate. The report
            uses 1000, which is the default.
    """

    def __init__(
        self,
        num_experts: int,
        dim: int,
        top_k: int = 1,
        *,
        num_bins: int = 1000,
        route_norm: bool = False,
        route_scale: float = 1.0,
        aux_loss: AuxLoss | None = None,
    ) -> None:
        super().__init__(
            num_experts,
            dim,
            top_k,
            score_func="sigmoid",
            route_norm=route_norm,
            route_scale=route_scale,
            aux_loss=aux_loss,
        )
        self.quantile_balancer = QuantileBalancer(num_experts, top_k, num_bins)

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Biased Top-(K+1): route on the first K, observe the cutoff."""
        if expert_bias_E is None:
            raise ValueError(
                "Quantile-balanced routing requires an expert bias; the MoE "
                "owning this router must register expert_bias_E."
            )
        if not self.training:
            # Eval routes with the plain biased top-k: no histogram is being
            # accumulated, so the cutoff is not needed and the base path is
            # the same decision.
            return super()._select_experts(scores_TE, expert_bias_E)

        biased_scores_TE = scores_TE + expert_bias_E
        topk_plus_one_scores_TK1, topk_plus_one_expert_ids_TK1 = torch.topk(
            biased_scores_TE,
            k=self.top_k + 1,
            dim=-1,
            sorted=True,
        )
        self.quantile_balancer.observe(
            scores_TE,
            topk_plus_one_scores_TK1[:, self.top_k :],
            expert_bias_E,
            padding_mask_T=padding_mask_T,
        )
        return topk_plus_one_expert_ids_TK1[:, : self.top_k].contiguous()


class QuantileBalancer(nn.Module):
    """Accumulate and recover histogram-based quantile bias updates.

    For sigmoid scores bounded in ``[0, 1]``, the bias an expert needs to win
    a token lies between the current minimum bias minus one and the maximum
    bias plus one. Each training micro-batch's required biases
    (``cutoff - score``, for every token and expert) are accumulated into
    uniform bins over that interval; ``estimate_expert_bias`` then reads the
    ``top_k / num_experts`` quantile back out, interpolated within its
    crossing bin.

    The histogram is non-persistent: scratch state that follows the module's
    device moves but never lands in a checkpoint.
    """

    def __init__(self, num_experts: int, top_k: int, num_bins: int) -> None:
        super().__init__()
        if not 0 < top_k < num_experts:
            raise ValueError(
                f"top_k ({top_k}) must be between zero and num_experts "
                f"({num_experts})"
            )
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_bins = num_bins
        self.register_buffer(
            "required_bias_histogram_EB",
            torch.zeros(num_experts, num_bins, dtype=torch.int32),
            persistent=False,
        )

    @torch.no_grad()
    def observe(
        self,
        scores_TE: torch.Tensor,
        cutoff_T1: torch.Tensor,
        expert_bias_E: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one local micro-batch's required-bias histogram.

        Padding tokens are filtered out first: a bias estimated from tokens
        that carry no loss would balance the wrong distribution.
        """
        if not self.training:
            return
        if padding_mask_T is not None:
            valid_mask_T = ~padding_mask_T
            scores_TE = scores_TE[valid_mask_T]
            cutoff_T1 = cutoff_T1[valid_mask_T]
        lower_bound = expert_bias_E.min() - 1.0
        bin_width = (
            expert_bias_E.max() - expert_bias_E.min() + 2.0
        ) / self.num_bins
        required_bias_TE = cutoff_T1 - scores_TE
        bin_indices_TE = torch.floor(
            (required_bias_TE - lower_bound) / bin_width
        ).to(torch.int64)
        bin_indices_ET = bin_indices_TE.clamp_(0, self.num_bins - 1).transpose(0, 1)
        self.required_bias_histogram_EB.scatter_add_(
            1,
            bin_indices_ET,
            torch.ones_like(
                bin_indices_ET,
                dtype=self.required_bias_histogram_EB.dtype,
            ),
        )

    def estimate_expert_bias(
        self,
        histogram_EB: torch.Tensor,
        expert_bias_E: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the next mean-centred expert bias from the histogram.

        The target is the ``top_k / num_experts`` quantile of each expert's
        required-bias distribution: the bias at which the expert would win
        exactly its uniform share of the observed tokens. The result is
        mean-centred so ``sum(expert_bias_E) == 0`` and the bias shifts which
        experts win without shifting the routed output as a whole -- the same
        invariant the sign-based update keeps.
        """
        counts_E = histogram_EB.sum(dim=-1, dtype=torch.int64)
        target_count_E = counts_E.float() * (self.top_k / self.num_experts)
        cumulative_counts_EB = histogram_EB.cumsum(dim=-1, dtype=torch.int64)
        target_rank_E = target_count_E.ceil().to(torch.int64)
        target_bin_E = (cumulative_counts_EB < target_rank_E.unsqueeze(-1)).sum(dim=-1)

        target_bin_E1 = target_bin_E.unsqueeze(-1)
        counts_in_bin_E = histogram_EB.gather(-1, target_bin_E1).squeeze(-1)
        counts_before_E = (
            cumulative_counts_EB.gather(-1, target_bin_E1).squeeze(-1) - counts_in_bin_E
        )
        fraction_E = (
            target_count_E - counts_before_E.float()
        ) / counts_in_bin_E.float()

        bin_width = (expert_bias_E.max() - expert_bias_E.min() + 2.0) / self.num_bins
        quantile_position_E = target_bin_E.float() + fraction_E
        return (quantile_position_E - quantile_position_E.mean()) * bin_width
