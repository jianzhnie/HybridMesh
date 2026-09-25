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

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

from hpmesh.accelerator.dist import all_reduce
from hpmesh.accelerator.spmd_context import spmd_mesh_group, spmd_sparse_mesh

from .aux_loss import AuxLoss
from .grouped_experts import GroupedExperts
from .linear import RouterGateLinear as RouterGateLinear
from .routers import (
    QuantileBalancedTopKRouter,
    QuantileBalancer,
    TokenChoiceTopKRouter,
)
from .token_dispatcher import LocalTokenDispatcher

__all__ = [
    "MOE_LAYER_ATTRS",
    "MoE",
    "MicrobatchWiseLoadBalanceLoss",
    "QuantileBalancedTopKRouter",
    "QuantileBalancer",
    "RoutedExperts",
    "TokenChoiceTopKRouter",
    "register_moe_load_balancing_hook",  # noqa: F822 - lazy re-export via __getattr__
    "register_moe_quantile_balancing_hook",  # noqa: F822 - lazy re-export via __getattr__
]

# The decoder-layer attributes that may hold a MoE block. Every model family
# transformers 5.x supports keeps it on ``mlp``, dense layers included -- a
# dense ``mlp`` is simply not a MoE and is skipped. The swap for an HF model
# replaces the block in the attribute it was found under, so this list is shared
# with it rather than duplicated: the two must agree or the expert-bias hook
# silently finds no layers on a real swapped model.
MOE_LAYER_ATTRS = ("mlp",)


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
        # ``tp`` belongs under EP for the same reason as upstream: EP borrows
        # ranks from TP, so an EP'd MoE sees a tp-partial token stream. Under
        # MoE-under-TP (tp>1, ep=1) no reduction is needed: the block-boundary
        # all-gather means the router already sees the full token stream, and
        # the aux loss only exists on the hpmesh MoE stack, which the TP path
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


_BALANCING_EXPORTS = {
    "_update_expert_bias",
    "_update_quantile_expert_bias",
    "register_moe_load_balancing_hook",  # noqa: F822 - lazy re-export via __getattr__
    "register_moe_quantile_balancing_hook",  # noqa: F822 - lazy re-export via __getattr__
}


def __getattr__(name: str):
    # Lazy re-export: balancing.py imports this module back for
    # ``_iter_moe_layers`` / ``MoE``, so an eager import here would cycle.
    if name in _BALANCING_EXPORTS:
        from . import balancing

        return getattr(balancing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
