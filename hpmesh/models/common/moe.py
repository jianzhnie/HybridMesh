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
  ``MicrobatchWiseLoadBalanceLoss``'s Partial -> Invariant reduction is the
  ``_PartialToInvariantAllReduce`` autograd Function (all-reduce forward,
  identity backward) instead of ``spmd.redistribute`` -- the same semantics,
  which ``torch.distributed.nn.all_reduce`` would NOT give: its backward is a
  second all-reduce, multiplying the injected gradient by the group size.
* ``MicrobatchWiseLoadBalanceLoss`` is ported and wired: the router runs it
  on each training forward through its ``aux_loss`` slot, matching upstream.
* the auxiliary-loss-free bias is updated by ``MoE.update_expert_bias``, which
  the *trainer* calls once per optimizer step (torchtitan reaches the same state
  through an optimizer hook; hpmesh has no hook registry, and the trainer
  already owns the step boundary). The rule is the same sign-based, mean-centred
  nudge, and the counter is drained there.
* the ``_debug_force_load_balance`` switch is kept: a constructor argument
  here rather than a Config field, with the same round-robin semantics.
* quantile-balanced routing (Kimi K3 Sec 2.3.3 / Appendix D) is ported as
  ``QuantileBalancedTopKRouter`` + ``QuantileBalancer``, with the per-step
  bias solve in ``register_moe_quantile_balancing_hook`` next to the
  sign-based one. It keeps the parameterized-constructor style of this file:
  the sigmoid score function, no group restriction and no debug round-robin
  are enforced structurally, by not taking those parameters.
* every router takes an optional ``padding_mask_T`` (true for padding). It never
  changes the routing decision, the dispatch, or the expert compute -- those run
  on the full token stream. It filters only the load-balancing *statistics*:
  ``tokens_per_expert_E``, the aux loss's f/p terms, and the quantile
  histogram all count valid tokens only, while the no-mask path is unchanged.
* node-limited routing (DeepSeek-V3's ``n_group``/``topk_group``) lives in
  ``_select_experts_within_groups``, reached by passing
  ``num_expert_groups``/``num_limited_groups``. One deliberate difference from
  torchtitan and HF: the out-of-group experts are masked to ``-inf`` rather than
  to ``0.0``, because those two only agree while the additive ``expert_bias_E``
  leaves every score positive. See ``_select_experts_within_groups``.

Shape legend, scoped to this file: ``T`` = tokens, ``D`` = model dimension,
``E`` = experts, ``K`` = experts per token (top-k), ``e`` = local experts under
EP, ``R`` = routed tokens landing on this rank's experts, ``F`` = expert hidden
dimension, ``B`` = quantile-histogram bins.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ...parallel.parallel_dims import ParallelDims

from hpmesh.accelerator.dist import all_reduce
from hpmesh.accelerator.spmd_context import spmd_mesh_group, spmd_sparse_mesh

from .aux_loss import AuxLoss
from .grouped_experts import GroupedExperts
from .linear import RouterGateLinear
from .token_dispatcher import LocalTokenDispatcher

__all__ = [
    "MOE_LAYER_ATTRS",
    "MoE",
    "MicrobatchWiseLoadBalanceLoss",
    "QuantileBalancedTopKRouter",
    "QuantileBalancer",
    "RoutedExperts",
    "TokenChoiceTopKRouter",
    "register_moe_load_balancing_hook",
    "register_moe_quantile_balancing_hook",
]

# The decoder-layer attributes that may hold a MoE block. Every model family
# transformers 5.x supports keeps it on ``mlp``, dense layers included -- a
# dense ``mlp`` is simply not a MoE and is skipped. The swap for an HF model
# replaces the block in the attribute it was found under, so this list is shared
# with it rather than duplicated: the two must agree or the expert-bias hook
# silently finds no layers on a real swapped model.
MOE_LAYER_ATTRS = ("mlp",)


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
            optimizer hook, so it survives gradient accumulation. Must be
            ``None`` when ``router`` is a ``QuantileBalancedTopKRouter``: the
            quantile update replaces the sign-based one (the two schemes
            writing the same buffer would fight), and the bias buffer is then
            registered unconditionally.
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
        # a per-expert bias nudged by observed load, updated once per optimizer
        # step (``update_expert_bias``) so it sees a whole accumulation cycle
        # rather than one microbatch.
        quantile_balanced = isinstance(router, QuantileBalancedTopKRouter)
        if quantile_balanced and load_balance_coeff is not None:
            raise ValueError(
                "A QuantileBalancedTopKRouter is balanced by the quantile "
                "update, so load_balance_coeff must be None -- the sign-based "
                "and quantile updates cannot both own expert_bias_E."
            )
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
        elif quantile_balanced:
            # The quantile scheme routes on the bias too, so the buffer exists
            # even with no sign-based update to drive it; the quantile hook
            # overwrites it once per step. Persistent for the same reason the
            # sign-based one is: a resumed run keeps the reached balance.
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
        # Staged padding mask for the next forward (see ``set_padding_mask``).
        # A plain attribute, not a buffer: it is per-microbatch input, never
        # module state, and must stay out of the state_dict.
        self._pending_padding_mask: torch.Tensor | None = None

    @torch.no_grad()
    def update_expert_bias(self) -> None:
        """Nudge ``expert_bias_E`` toward balance from the accumulated counts.

        Called once per optimizer step, after ``tokens_per_expert_E`` has been
        turned into this layer's expert counts (the caller is responsible for
        summing the counter over the axes that shard a token stream). The step
        is sign-based and then mean-centred:

        * the sign makes the step size independent of how lopsided the load is,
          so a single runaway expert cannot dominate the update -- and it is
          also what makes the doubled count from activation checkpointing
          harmless, since ``sign`` of a scaled value is the same sign;
        * centring keeps ``sum(expert_bias_E) == 0``, so the bias shifts which
          experts win without shifting the routed output as a whole.

        This mirrors torchtitan's ``_update_expert_bias``. It differs from
        Eq. 14 of the paper, which moves only the most- and least-loaded
        experts; this moves every expert by one step whose sign depends on
        whether it is above or below the mean.
        """
        if self.expert_bias_E is None or self.load_balance_coeff is None:
            return
        counts_E = self.tokens_per_expert_E
        delta_E = self.load_balance_coeff * torch.sign(counts_E.mean() - counts_E)
        self.expert_bias_E.add_(delta_E - delta_E.mean())
        self.tokens_per_expert_E.zero_()

    def set_padding_mask(self, padding_mask: torch.Tensor | None) -> None:
        """Stage the padding mask for the NEXT forward of this block.

        The HF decoder layer calls its MoE as ``self.mlp(hidden_states)`` --
        its fixed signature has no slot for a mask, so the wrapper
        (``HFTransformerModel.forward``) stages the microbatch's mask on every
        swapped MoE block just before the decoder runs. The mask is consumed
        by the next ``forward`` and cleared, so a stale mask can never leak
        into a later microbatch that carried none: a forward entered without
        any staging is exactly the no-mask path.

        An explicit ``padding_mask`` passed to ``forward`` takes precedence
        over (and still consumes) a staged one.
        """
        self._pending_padding_mask = padding_mask

    def forward(
        self, x: torch.Tensor, *, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Route, run the experts, and sum the routed and shared outputs.

        Accepts ``(T, D)`` or any leading-dimension form ``(..., T, D)``. The
        HF decoder layer calls its MoE as ``self.mlp(hidden_states)`` with a
        ``(batch, seq, D)`` tensor, so a swapped-in block has to take that shape
        and give it back; the routing itself works on flattened tokens.

        ``padding_mask`` (true for padding) follows the same flattening; it
        filters only the load-balancing statistics -- the routing decision,
        dispatch and expert compute always see the full token stream, so the
        block's output is unaffected by it. ``None`` falls back to a mask
        staged via ``set_padding_mask``.
        """
        if padding_mask is None:
            padding_mask = self._pending_padding_mask
        self._pending_padding_mask = None
        if x.dim() > 2:
            lead = x.shape[:-1]
            out = self._forward_tokens(
                x.reshape(-1, x.shape[-1]),
                padding_mask_T=(
                    None if padding_mask is None else padding_mask.reshape(-1)
                ),
            )
            return out.reshape(*lead, out.shape[-1])
        return self._forward_tokens(x, padding_mask_T=padding_mask)

    def _forward_tokens(
        self,
        x_TD: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The MoE computation over a flat ``(T, D)`` token stream."""
        # (T, K) scores and ids; (T, E) map of which experts each token picked.
        (
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
        ) = self.router(x_TD, self.expert_bias_E, padding_mask_T=padding_mask_T)
        num_local_tokens_per_expert_E = routing_map_TE.sum(dim=0)

        if self.training:
            with torch.no_grad():
                # NOTE: activation checkpointing runs the forward twice, so this
                # counts a token twice on recompute. The bias update uses
                # sign(), so the doubled count does not change its direction.
                # The padding-filtered map is what is counted: padding tokens
                # are dispatched and computed, but they carry no loss, so they
                # must not steer the bias.
                counts_map_TE = (
                    routing_map_TE
                    if padding_mask_T is None
                    else routing_map_TE & ~padding_mask_T.unsqueeze(-1)
                )
                self.tokens_per_expert_E.add_(counts_map_TE.sum(dim=0))

        out_TD = self.routed_experts(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )

        if self.shared_experts is not None:
            out_TD = out_TD + self.shared_experts(x_TD)
        return out_TD


def _iter_moe_layers(model_part: nn.Module) -> list[MoE]:
    """The MoE blocks of one model part, in a stable order.

    Every model part is a ``HFTransformerModel``, whose ``layers`` is a
    ``ModuleList``. A dense layer carries no MoE -- and the swap leaves the block
    *in the attribute it already held* (``mlp``, for every family) rather than
    parking it under a new name. So the lookup has to walk the same names the
    swap writes to (``MOE_LAYER_ATTRS``); anything else finds nothing on a real
    model, which is the worst failure mode available here -- the register
    function below then no-ops and the bias is never updated at all.
    """
    layers = getattr(model_part, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        return []
    found: list[MoE] = []
    for layer in layers:
        for attr in MOE_LAYER_ATTRS:
            block = getattr(layer, attr, None)
            if isinstance(block, MoE):
                found.append(block)
                break
    return found


def _update_expert_bias(
    mappers: list[tuple[nn.Module, list[MoE]]],
    parallel_dims: ParallelDims | None,
) -> None:
    """Turn every MoE's accumulated token counts into one bias update.

    ``tokens_per_expert_E`` counted only the tokens *this* rank saw, so it is
    summed over every axis that shards one token stream before it means
    anything. Three collectives, for three different reasons:

    * ``dp`` -- each data-parallel rank processes a different slice of the
      batch, so the counts are partial over it. This is the one that matters
      for a plain FSDP run: without it every rank would balance its own shard
      and the biases would drift apart.
    * ``cp`` -- the sequence is split across the context-parallel ranks, so the
      tokens are partial there too.
    * ``tp`` -- only when EP is on: EP borrows ranks from TP, so the token
      stream is sharded over TP as well. Without EP every TP rank already sees
      the same full stream, so summing would multiply the counts by ``tp``
      (harmless for a sign-based step, but wrong for anything else reading it).

    Pipeline parallelism needs no collective here: it is a layer split, not a
    token split, so each stage's blocks see their own whole token stream.

    The expert dimension is deliberately *not* reduced. ``tokens_per_expert_E``
    is the global per-expert count -- the routing map spans all ``E`` experts on
    every EP rank, before the dispatcher narrows anything to a local shard -- so
    an EP all-reduce would double-count.

    Because every rank ends up with identical counts, the identical update is
    applied identically and ``expert_bias_E`` stays replicated, which is what
    the forward assumes.
    """
    layers_by_part = [layers for _, layers in mappers if layers]
    if not layers_by_part:
        return

    counts_LE = torch.vstack(
        [
            torch.stack([moe.tokens_per_expert_E for moe in layers])
            for layers in layers_by_part
        ]
    )

    axes = ["dp", "cp"] + (["tp"] if parallel_dims and parallel_dims.ep_enabled else [])
    for axis in axes:
        mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh(axis)
        if mesh is None:
            continue
        all_reduce(counts_LE, group=mesh.get_group())

    row = 0
    for _, layers in mappers:
        for moe in layers:
            moe.tokens_per_expert_E.copy_(counts_LE[row])
            row += 1
            moe.update_expert_bias()


def register_moe_load_balancing_hook(
    optimizer: torch.optim.Optimizer,
    model_parts: Sequence[nn.Module],
    parallel_dims: ParallelDims | None,
) -> None:
    """Register the step pre-hook that updates every MoE's expert bias.

    A *pre*-hook, so the counts it reads are a whole accumulation window's and
    the bias the next forward reads is the one the step just earned. That is
    also torchtitan's placement; hpmesh reaches the same point through PyTorch's
    own hook machinery rather than a hook registry.

    A no-op when no model part carries a MoE layer, so a dense run pays neither
    the traversal nor an empty collective. It is also a no-op when every MoE
    layer has ``load_balance_coeff=None`` -- nothing would consume the reduced
    counts, so no hook (and no per-step collective) is registered at all. A
    mixed configuration, where only some layers carry a coeff, is rejected
    outright: silently balancing a subset of the layers would look like a
    working setup while the rest drift (torchtitan makes the same check in
    ``_should_register_moe_balancing_hook``).
    """
    mappers = [(part, _iter_moe_layers(part)) for part in model_parts]
    all_layers = [moe for _, layers in mappers for moe in layers]
    if not all_layers:
        return
    load_balance_enabled = all_layers[0].load_balance_coeff is not None
    for moe in all_layers[1:]:
        if (moe.load_balance_coeff is not None) != load_balance_enabled:
            raise ValueError(
                "MoE load_balance_coeff must be configured consistently across "
                "all MoE layers. Either set it for every MoE layer or leave it "
                "unset for all MoE layers."
            )
    if not load_balance_enabled:
        return
    optimizer.register_step_pre_hook(
        lambda *args, **kwargs: _update_expert_bias(mappers, parallel_dims)
    )


@torch.no_grad()
def _update_quantile_expert_bias(
    moe_layers: list[MoE],
    parallel_dims: ParallelDims | None,
) -> None:
    """Reduce the quantile histograms and write the next expert biases.

    The histograms count the same tokens ``tokens_per_expert_E`` counts, so
    they are summed over exactly the same axes (dp, cp, and tp only when EP
    shards the token stream over it) -- see ``_update_expert_bias`` for why
    those axes and no others. Every rank then holds the identical global
    histogram, computes the identical estimate, and ``expert_bias_E`` stays
    replicated, which is what the forward assumes.

    After the update each layer's histogram and token counter are drained:
    both are per-step scratch, and the counter is otherwise never consumed
    (the sign-based ``update_expert_bias`` that drains it is disabled for
    these layers).
    """
    histograms = [
        moe.router.quantile_balancer.required_bias_histogram_EB
        for moe in moe_layers
    ]
    stacked_LEB = torch.stack(histograms)

    axes = ["dp", "cp"] + (["tp"] if parallel_dims and parallel_dims.ep_enabled else [])
    for axis in axes:
        mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh(axis)
        if mesh is None:
            continue
        all_reduce(stacked_LEB, group=mesh.get_group())

    for moe, histogram_EB in zip(moe_layers, stacked_LEB.unbind(), strict=True):
        quantile_balancer = moe.router.quantile_balancer
        moe.expert_bias_E.copy_(
            quantile_balancer.estimate_expert_bias(histogram_EB, moe.expert_bias_E)
        )
        quantile_balancer.required_bias_histogram_EB.zero_()
        moe.tokens_per_expert_E.zero_()


def register_moe_quantile_balancing_hook(
    optimizer: torch.optim.Optimizer,
    model_parts: Sequence[nn.Module],
    parallel_dims: ParallelDims | None,
) -> None:
    """Register the step pre-hook that updates quantile-balanced expert biases.

    Same placement as ``register_moe_load_balancing_hook``: a pre-hook, so the
    bias the next forward reads is earned by the whole accumulation window
    that just finished. A no-op when no MoE layer carries a
    ``QuantileBalancedTopKRouter``.

    The two balancing schemes are mutually exclusive, matching the upstream
    wiring where a model registers exactly one of the two hooks. A single
    MoE already refuses to combine them (``MoE.__init__`` raises on a
    quantile router with a coeff); a *mixed* model -- quantile routers on
    some layers, sign-based coeff on others -- is rejected here, because
    silently balancing different layers by different rules would look like a
    working setup while the load-balance hook either no-ops or raises on the
    inconsistent coeff configuration.
    """
    all_layers = [moe for part in model_parts for moe in _iter_moe_layers(part)]
    quantile_layers = [
        moe
        for moe in all_layers
        if isinstance(moe.router, QuantileBalancedTopKRouter)
    ]
    if not quantile_layers:
        return
    if len(quantile_layers) != len(all_layers) or any(
        moe.load_balance_coeff is not None for moe in all_layers
    ):
        raise ValueError(
            "Quantile-balanced routing is mutually exclusive with the "
            "sign-based load-balancing bias: every MoE layer must use a "
            "QuantileBalancedTopKRouter with load_balance_coeff=None, or "
            "none may."
        )
    optimizer.register_step_pre_hook(
        lambda *args, **kwargs: _update_quantile_expert_bias(
            quantile_layers, parallel_dims
        )
    )


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
        # "batch" (dp) rather than "loss" (dp+cp): the value is already the
        # same on every CP coordinate by construction, so summing over cp too
        # would count it more than once per layer.
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
        # ``tp`` belongs under EP for the same reason as upstream: hpmesh's TP
        # is the sequence-parallel formulation (``tensor_parallel/tp.py``
        # reduce-scatters the residual stream back to a sequence shard), so a
        # TP'd MoE would see a tp-partial token stream. Today the combination
        # is unreachable -- ``apply_tp`` refuses every supported MoE family's
        # HF tp_plan (unsupported spec strings) -- so the tp term is dormant
        # rather than wrong: dropping it would under-reduce if TP+EP ever ran.
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
