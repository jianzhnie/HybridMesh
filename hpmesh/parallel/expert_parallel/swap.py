"""Swap a HF MoE block for hpmesh's EP-capable MoE, weights included.

Every sparse decoder layer of a transformers 5.x MoE model holds the same block
under ``layer.mlp``: a router, plus two stacked expert parameters. Expert
parallelism needs those weights in grouped-GEMM form with a token dispatcher in
front -- the stack in ``models/common/`` (``GroupedExperts`` /
``TokenChoiceTopKRouter`` / ``MoE``). This module is the bridge: it detects the
HF block, moves the weights into the hpmesh layout, and replaces ``layer.mlp``.
It also flags the layer for the FSDP MoE branch (``layer.moe_enabled = True``
plus a non-registered ``layer.moe`` alias, as upstream's swap does), without
which the expert weights would be sharded as dense parameters over the dense
DP mesh. The swap is in place and dtype-preserving, so it must run before FSDP
wraps the model and after the weights are loaded.

Both halves of the block vary by family, and the probe is duck-typed against
structures rather than class names, which are not stable across transformers
releases:

* **The router.** ``block.gate`` or ``block.router``, either a plain
  ``nn.Linear`` (DeepSeek-V2) or a bespoke module holding the same ``(E, D)``
  weight plus, for DeepSeek-V3/GLM4, an ``e_score_correction_bias`` buffer.
  Identified by its *weight tensor*, not its type (``_router_of``).
* **The expert weights.** Always ``gate_up_proj (E, 2F, D)`` and
  ``down_proj (E, D, F)``, split by ``chunk(2, dim=-1)`` after the gate+up
  GEMM. Identified by shape (``_fused_experts_of``).

Where a family keeps its routing attributes also moved in 5.x: ``top_k``,
``n_group``, ``topk_group``, ``norm_topk_prob``, ``scoring_func`` and
``routed_scaling_factor`` now sit on the *block* for every family, having been
split between block and router in 4.x. Every ``_read_*`` helper therefore takes
both owners and prefers whichever declares the field.

Weight layout: HF stores each projection as ``(out, in)`` and applies it as
``x @ W.T``; ``GroupedExperts`` stores the same orientation per expert
(``w1_EFD``/``w3_EFD`` are ``(F, D)``, ``w2_EDF`` is ``(D, F)``) and applies it
with the same ``F.linear``. Moving a weight is therefore elementwise -- no
transpose, no regrouping -- except for the one split of ``gate_up_proj`` into
its gate and up halves. ``tests/unit_tests/cpu/distributed/test_ep_swap.py``
pins both halves of that claim against the HF block being replaced.

Routing parity with the HF block: Qwen3Moe and Mixtral score with a softmax over
fp32 logits; DeepSeek-V3/GLM4 score with a sigmoid and apply
``routed_scaling_factor``; DeepSeek-V2 scores with a softmax but never
renormalizes, whatever its config declares (see ``_ignores_norm_topk_prob``).
``RouterGateLinear`` computes in fp32, so the same score function reproduces it,
and ``TokenChoiceTopKRouter`` takes node-limited routing as
``num_expert_groups``/``num_limited_groups``. The one deliberate addition is the
load-balance machinery HF never gets to run through this wrapper (the wrapper
never asks for router logits): the router carries a
``MicrobatchWiseLoadBalanceLoss`` with the config's ``router_aux_loss_coef``,
whose gradient is injected on backward (see ``models/common/aux_loss.py``).

Two families are deliberately *not* swapped, each because the HF block asks for
something the hpmesh stack does not express -- refusing beats approximating,
since the difference shows up only as different experts being chosen, which no
loss curve reveals:

* ``GPT-OSS``: per-expert bias vectors, a transposed ``(E, D, 2F)`` layout, and
  a hardcoded clamped sigmoid-GLU activation rather than a module
  (see ``_fused_experts_of``).
* ``DeepSeek-V2`` with ``topk_method="group_limited_greedy"``: scores a group by
  its single best expert where V3/GLM4 sum the top-2 (see
  ``_read_expert_groups``). Its default ``"greedy"`` is supported and exact.

``Qwen2Moe``'s ``shared_expert_gate`` multiplies where ``MoE.shared_experts``
only adds, and is refused at the point the shared expert is found.
"""

from __future__ import annotations

import torch.nn as nn

from hpmesh.errors import EnvironmentUnsupportedError

from ...models.common.token_dispatcher import EP_DISPATCHER_BACKENDS
from ...utils.logger_utils import get_logger
from .convert import _convert_block, _restore_fp32_state_buffers
from .probe import _is_hf_moe_block, _moe_block_of, _router_of

logger = get_logger(__name__)

__all__ = ["swap_hf_moe_blocks"]

# Re-exported for callers that reached the probe/convert helpers through this
# module before the three-way split (tensor_parallel/apply.py, the swap tests).
_is_hf_moe_block = _is_hf_moe_block
_restore_fp32_state_buffers = _restore_fp32_state_buffers


def swap_hf_moe_blocks(
    model: nn.Module,
    *,
    ep_group=None,
    router_aux_loss_coef: float | None = None,
    quantile_balancing: bool = False,
    token_dispatcher: str = "alltoall",
    torchao_pad_multiple: int = 16,
    tp_enabled: bool = False,
) -> int:
    """Replace every HF MoE block in ``model`` with hpmesh's MoE, in place.

    Args:
        model: a ``HFTransformerModel`` (anything exposing ``.layers``).
        ep_group: the EP process group. ``None`` (or a size-1 group is not
            special-cased -- pass ``None``) keeps all experts local and routes
            them with the reordering-only dispatcher; a multi-rank group
            shards the experts across it and routes with all-to-alls.
        router_aux_loss_coef: coefficient for the per-forward load-balance
            loss. ``None`` takes the HF config's ``router_aux_loss_coef``
            (Qwen3Moe has one; DeepSeek-V3's config has no such field and gets
            no loss). A float here overrides the config for every MoE layer.
        quantile_balancing: replace the sign-based load-balancing bias with
            quantile-balanced routing (``QuantileBalancedTopKRouter``): the
            bias is then re-solved from a required-bias histogram once per
            optimizer step instead of nudged by the sign rule, and
            ``load_balance_coeff`` is forced off. Requires sigmoid router
            scores and no group-limited routing.
        token_dispatcher: EP dispatch backend (``"alltoall"`` default,
            ``"torchao"`` optional-import adapter, ``"deepep"``/``"hybridep"``
            registered gaps refused here). See
            ``ParallelConfig.ep_token_dispatcher``.
        torchao_pad_multiple: padding multiple for the ``"torchao"`` backend.
        tp_enabled: the model is also tensor-parallelized (tp x ep). Used only
            for fail-fast validation of unverified combinations (currently a
            shared expert): the swap itself is layout-identical either way --
            the swapped block consumes and produces the T/tp sequence shard
            directly, which is the layout upstream's ep+sp MoE uses.

    Returns:
        The number of blocks swapped. Mixed sparse/dense models (e.g.
        Qwen3Moe's ``decoder_sparse_step``) swap only the sparse layers.

    Raises:
        TypeError: if no layer carries a recognizable HF MoE block. EP on a
            dense model is a config mistake, and silently swapping nothing
            would run it replicated.
        NotImplementedError: if a MoE block uses a layout this swap does not
            implement -- GPT-OSS's transposed, bias-bearing experts (see
            ``_fused_experts_of``) or DeepSeek-V2's ``group_limited_greedy``
            routing (see ``_read_expert_groups``).
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"swap_hf_moe_blocks expects a model with .layers; got "
            f"{type(model).__name__}."
        )
    # Backend gating, before any probing: ParallelConfig.__post_init__ is the
    # primary gate; this is the defensive copy for callers that reach the swap
    # directly.
    if token_dispatcher not in EP_DISPATCHER_BACKENDS:
        raise ValueError(
            f"unknown ep_token_dispatcher {token_dispatcher!r}; expected one "
            f"of {EP_DISPATCHER_BACKENDS}."
        )
    if token_dispatcher in ("deepep", "hybridep"):
        raise EnvironmentUnsupportedError(
            f"ep_token_dispatcher={token_dispatcher!r} is a registered gap: "
            "CUDA-only kernels plus torchtitan's distributed/deepep/ wrappers "
            "that hpmesh does not vendor. Use 'alltoall' meanwhile."
        )

    hf_config = getattr(getattr(model, "model", None), "config", None)
    aux_loss_coeff = (
        router_aux_loss_coef
        if router_aux_loss_coef is not None
        else getattr(hf_config, "router_aux_loss_coef", None)
    )
    # The load-balance coefficient belongs to a *native* hpmesh run; an HF
    # checkpoint has no equivalent field. It is therefore derived from the
    # probed bias: a block that carries ``e_score_correction_bias`` is one the
    # model was trained with the auxiliary-loss-free scheme on (DeepSeek-V3,
    # GLM4), and gets the hpmesh default. Everything else stays off, so a
    # frozen zero bias never lands in the checkpoint.
    default_coeff = getattr(hf_config, "load_balance_coeff", 1e-3)

    swapped = 0
    for layer in layers:
        attr, block = _moe_block_of(layer)
        # Dense layers of a mixed model (Qwen3Moe's decoder_sparse_step, and the
        # first_k_dense_replace of the DeepSeek/GLM4 families) are left alone.
        if block is None or not _is_hf_moe_block(block):
            continue
        router_gate = _router_of(block)
        load_balance_coeff = (
            float(default_coeff)
            if getattr(router_gate, "e_score_correction_bias", None) is not None
            else None
        )
        moe = _convert_block(
            block,
            ep_group=ep_group,
            aux_loss_coeff=aux_loss_coeff,
            load_balance_coeff=load_balance_coeff,
            quantile_balancing=quantile_balancing,
            token_dispatcher=token_dispatcher,
            torchao_pad_multiple=torchao_pad_multiple,
            tp_enabled=tp_enabled,
        )
        # Replace in the slot the block was actually found in. Writing to a
        # different attribute would leave the original block in place and route
        # around the swap entirely.
        setattr(layer, attr, moe)
        # FSDP's MoE branch (fully_shard/fsdp.py) keys off these two
        # attributes, as does upstream's swap (moe_replacement.py):
        # ``moe_enabled`` marks the block as sparse, and ``moe`` is where the
        # branch reads the expert weights from. Without them the experts are
        # sharded as dense parameters over the dense DP mesh, mixing ranks of
        # different EP coordinates into one FSDP group. ``object.__setattr__``
        # keeps ``moe`` out of the module registry: it is the same module as
        # the block attribute, and registering it would double every expert
        # weight in the state_dict.
        layer.moe_enabled = True
        object.__setattr__(layer, "moe", moe)
        swapped += 1

    if swapped == 0:
        raise TypeError(
            f"no HF MoE block found on {type(model).__name__} "
            f"({type(getattr(model, 'model', model)).__name__}): no layer's "
            "``mlp`` has the router-gate + fused gate_up_proj/down_proj shape "
            "this swap recognizes. Add the model family's layout to the probe "
            "in parallel/expert_parallel/swap.py."
        )
    logger.info("Swapped %d HF MoE blocks for hpmesh MoE blocks", swapped)
    return swapped
