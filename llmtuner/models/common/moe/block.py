"""The MoE block itself: route, dispatch, compute, combine, sum.

Vendored from torchtitan ``models/common/moe.py`` -- the ``MoE`` block and the
walk that finds the blocks of a model. The rest of that upstream file lives
next to this one: the routers in ``router.py``, the expert weights in
``experts.py``, the EP dispatchers in ``dispatcher.py``, the balance loss in
``load_balance.py``, the bias-update hooks in ``balancing.py``.

What changed from upstream:

* the nested ``Config`` dataclass is gone; the block is constructed directly.
* the auxiliary-loss-free bias is updated by ``MoE.update_expert_bias``, which
  the *trainer* calls once per optimizer step (torchtitan reaches the same state
  through an optimizer hook; llmtuner has no hook registry, and the trainer
  already owns the step boundary). The rule is the same sign-based, mean-centred
  nudge, and the counter is drained there; the reduction over the axes that
  shard a token stream is ``balancing.py``.
* the block is balancing-scheme-agnostic: a quantile router is driven by
  ``balancing.py``'s quantile hook instead, and the bias buffer is registered
  for it here (upstream needs a ``KimiLatentMoE`` subclass to do that). The two
  schemes cannot both own the buffer, which the constructor enforces.
* ``padding_mask`` (true for padding) is a staged per-microbatch input. It never
  changes the routing decision, the dispatch, or the expert compute -- those run
  on the full token stream. It filters only the load-balancing *statistics*:
  ``tokens_per_expert_E`` and the aux loss's f/p terms count valid tokens only,
  while the no-mask path is unchanged.

Shape legend, scoped to this file: ``T`` = tokens, ``D`` = model dimension,
``E`` = experts, ``K`` = experts per token (top-k).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .experts import RoutedExperts
from .router import QuantileBalancedTopKRouter, TokenChoiceTopKRouter

__all__ = ["MOE_LAYER_ATTRS", "iter_moe_layers", "MoE"]

# The decoder-layer attributes that may hold a MoE block. Every model family
# transformers 5.x supports keeps it on ``mlp``, dense layers included -- a
# dense ``mlp`` is simply not a MoE and is skipped. The swap for an HF model
# replaces the block in the attribute it was found under, so this list is shared
# with it rather than duplicated: the two must agree or the expert-bias hook
# silently finds no layers on a real swapped model.
MOE_LAYER_ATTRS = ("mlp",)


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

        This mirrors torchtitan's ``update_expert_bias``. It differs from
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


def iter_moe_layers(model_part: nn.Module) -> list[MoE]:
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
