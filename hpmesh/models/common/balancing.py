"""MoE load-balancing hooks: the per-step expert-bias updates.

Split out of ``moe.py``: the sign-based bias update
(``register_moe_load_balancing_hook``) and the quantile-histogram update
(``register_moe_quantile_balancing_hook``), plus their reduction helpers.
Both are optimizer step pre-hooks over the MoE blocks ``_iter_moe_layers``
finds; the blocks themselves and the routers live in ``moe.py`` /
``routers.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from hpmesh.accelerator.dist import all_reduce

from .moe import MoE, _iter_moe_layers
from .routers import QuantileBalancedTopKRouter

if TYPE_CHECKING:
    from ...parallel.parallel_dims import ParallelDims


def update_expert_bias(
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
        lambda *args, **kwargs: update_expert_bias(mappers, parallel_dims)
    )


@torch.no_grad()
def update_quantile_expert_bias(
    moe_layers: list[MoE],
    parallel_dims: ParallelDims | None,
) -> None:
    """Reduce the quantile histograms and write the next expert biases.

    The histograms count the same tokens ``tokens_per_expert_E`` counts, so
    they are summed over exactly the same axes (dp, cp, and tp only when EP
    shards the token stream over it) -- see ``update_expert_bias`` for why
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
        lambda *args, **kwargs: update_quantile_expert_bias(
            quantile_layers, parallel_dims
        )
    )
