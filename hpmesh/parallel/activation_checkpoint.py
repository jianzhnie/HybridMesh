"""Activation checkpointing: recompute each decoder layer during backward.

Vendored in shape from torchtitan's ``distributed/activation_checkpoint.py``.
Two of its policies are ported, and they are the two that need nothing
outside torch:

* ``"full"`` (upstream ``FullAC``) wraps each decoder layer in torch's
  non-reentrant ``checkpoint_wrapper``, so a forward keeps only the layer's
  inputs and recomputes its activations inside backward -- one extra forward
  per layer in exchange for the layer's activation memory.

* ``"selective"`` (upstream ``SelectiveAC``) is per-op: a ``context_fn`` policy
  is asked about every op inside the layer and answers ``MUST_SAVE`` for the
  ones in the save set (``_get_default_save_ops`` -- matmuls, SDPA and flex
  attention, and the collectives whose outputs are expensive to resend) and
  ``PREFER_RECOMPUTE`` for the rest. Matmuls in the save set are recomputed
  every second time instead of always, which is the memory/compute dial. One
  op is dropped from upstream's set; ``_get_default_save_ops`` says which and
  why, and it is the one place that behavioural difference lives.

Both use the same wrapper factory as upstream
(``torch.distributed.algorithms._checkpoint.checkpoint_wrapper``), with the
same non-default knob, ``early_stop=False``, so a checkpointed region inside
the layer cannot end the recompute early. ``"full"`` additionally keeps
``preserve_rng_state=True`` by default, so the recompute sees the RNG state
the original forward saw and the run stays bitwise-equal to the
uncheckpointed one.

Not ported, deliberately:

* ``RegionAC`` needs ``torch_remat``, which hpmesh does not depend on -- its
  model-declared regions are a ``Module``-protocol feature hpmesh has no
  equivalent of.
* ``MemoryBudgetAC`` sets two ``torch._functorch.config`` globals and lets the
  compile partitioner do the trading, so it only means anything once the model
  is compiled; it carries no policy code of its own to align with.
* ``_disable_dynamo_lru_cache``. It works around a SAC-with-pipeline-parallel
  recompilation interaction, and hpmesh refuses activation checkpointing on the
  ``pp > 1`` path outright (see ``parallelize_hf``), so the case it fixes is
  unreachable here. It also mutates a process-global dynamo knob, which is not
  something to do speculatively (see https://github.com/pytorch/pytorch/issues/166926).

The remaining deviation from upstream is the entry point: torchtitan selects a
policy by instantiating an ``ActivationCheckpointing`` subclass, hpmesh by
passing a mode string (the extension point its config already documents).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch._functorch.partitioners import get_default_op_list
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

if TYPE_CHECKING:
    from hpmesh.trainer.config import SelectiveACConfig

logger = logging.getLogger(__name__)

__all__ = ["VALID_AC_MODES", "apply_ac"]

VALID_AC_MODES = ("none", "full", "selective")


def _get_default_save_ops() -> set:
    """The ops whose activations ``"selective"`` saves rather than recomputes.

    Two sources, ported from upstream: torch's own list of compute-intensive
    ops (``get_default_op_list``), plus the explicit sets below. Each spec in
    those sets is either an op handle (always present) or a ``(root, dotted
    path)`` pair for an op that only exists in some builds -- resolved through
    ``getattr`` and skipped when it is not registered, so the same list runs
    against a CPU-only torch.

    The comm ops at the end are worth keeping pointed at even though hpmesh
    runs no DeepEP/HybridEP: they resolve-or-skip, so their absence is silent
    here while their presence (a future EP backend) is exactly when
    re-communication would be the thing to avoid.

    ``aten.topk`` is upstream's one deliberate omission. It saves topk because
    topk can be non-deterministic, so a recompute can pick different experts
    than the forward did. hpmesh cannot save it: HF's MoE routers normalize
    the values topk returns with an in-place divide (``router_top_value /=
    ...``), and torch's selective checkpoint raises "Tensor cached during
    selective activation checkpoint has been mutated" when a cached output is
    written afterwards -- on every MoE model, on any device. So a
    selective-AC run over HF MoE recomputes its topk and inherits the
    non-determinism: on a GPU whose topk kernel is not reproducible, the
    backward is then the gradient of a slightly different expert assignment.
    Expert assignment is a piecewise-constant partition of the scores, so a
    flip needs two experts to land on exactly equal scores; on CPU torches'
    topk is deterministic and the two runs agree bitwise (pinned by
    ``test_selective_ac_matches_uncheckpointed_bitwise_on_moe``).
    """
    # Outputs that are expensive to recompute (matmuls, attention, ...).
    compute_ops = [
        # SDPA variants
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        torch.ops.aten._scaled_dot_product_attention_math.default,
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        # Low-precision training always saves the absolute maximum used to
        # compute the quantization scaling factor.
        torch.ops.aten.max.default,
        # FlexInnerAttention (torch.ops.higher_order.flex_attention is the
        # same object).
        torch._higher_order_ops.flex_attention,
        torch.ops.aten.linear.default,
        torch.ops.aten.mm.dtype,
        # Inductor-compiled code (only present when torch.compile is used).
        (torch._higher_order_ops, "inductor_compiled_code"),
        # torch_attn custom backend.
        (torch.ops, "torch_attn._varlen_attn.default"),
    ]

    # Communication ops: saving their outputs avoids re-communicating.
    comm_ops = [
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops._c10d_functional.all_to_all_single.default,
        # DeepEP (only present when it is installed).
        (torch.ops, "deepep.dispatch.default"),
        (torch.ops, "deepep.combine.default"),
        # HybridEP (only present when it is installed).
        (torch.ops, "hybridep.dispatch.default"),
        (torch.ops, "hybridep.combine.default"),
    ]

    def _resolve(op_specs: list) -> set:
        # Upstream builds a dict here and then only ever uses its keys; a set
        # is the same value with the unused mapping dropped.
        ops = set()
        for spec in op_specs:
            if isinstance(spec, tuple):
                obj, path = spec
                try:
                    for part in path.split("."):
                        obj = getattr(obj, part)
                    ops.add(obj)
                except AttributeError:
                    pass
            else:
                ops.add(spec)
        return ops

    save_ops = {op.default for op in get_default_op_list().compute_intensive_ops}
    save_ops.update(_resolve(compute_ops))
    save_ops.update(_resolve(comm_ops))
    return save_ops


def _mm_recompute_shapes(
    module: nn.Module, base_fqn: str | None, fqns: list[str]
) -> set[tuple[int, int]]:
    """Collect the ``(in, out)`` weight shapes to force-recompute, by fqn.

    ``fqns`` are matched as substrings of each submodule's fully qualified
    name, exactly as upstream matches them -- so a pattern matches anywhere in
    the path, and the shape it yields applies to *any* matmul with that shape,
    not just the one whose module matched.
    """
    shapes: set[tuple[int, int]] = set()
    for module_fqn, submod in module.named_modules():
        fqn = f"{base_fqn}.{module_fqn}" if base_fqn else module_fqn
        if not any(f in fqn for f in fqns):
            continue
        if not isinstance(submod, nn.Linear):
            raise ValueError(
                "force_recompute_mm_shapes_by_fqns expected to match a "
                f"nn.Linear, but got: {submod}"
            )
        out_f, in_f = submod.weight.shape
        shapes.add((in_f, out_f))
    return shapes


# Some backends (e.g. PrivateUse1) register aten.linear as a leaf op instead of
# decomposing it into aten.mm, so both spellings have to be handled.
_MM_OPS = (
    torch.ops.aten.mm.default,
    torch.ops.aten.mm.dtype,
    torch.ops.aten.linear.default,
)


def _selective_policy(
    save_ops: set, mm_recompute_shapes: set[tuple[int, int]]
) -> Callable:
    """Build the per-op policy the selective context consults (upstream's
    ``_get_custom_policy``)."""
    meta = {"forward_mm_count": 0, "recompute_mm_count": 0}

    def wrapped_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
        # Always save CUDA -> CPU results rather than recomputing them (e.g. a
        # MoE D2H sync for all-to-all metadata).
        if (
            func == torch.ops.aten._to_copy.default
            and "cuda" in str(args[0].device)
            and "device" in kwargs
            and str(kwargs["device"]) == "cpu"
        ):
            return CheckpointPolicy.MUST_SAVE

        mode = "recompute" if ctx.is_recompute else "forward"
        mm_count_key = f"{mode}_mm_count"

        if func in _MM_OPS:
            weight_shape = args[1].shape
            # linear's weight is (out, in); normalize to mm's (in, out).
            if func == torch.ops.aten.linear.default:
                weight_shape = torch.Size((weight_shape[1], weight_shape[0]))
            if tuple(weight_shape) in mm_recompute_shapes:
                return CheckpointPolicy.PREFER_RECOMPUTE
            meta[mm_count_key] += 1

        # Save every compute/comm op in the set, except every second matmul.
        if func in save_ops:
            if func in _MM_OPS and meta[mm_count_key] % 2 == 0:
                return CheckpointPolicy.PREFER_RECOMPUTE
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return wrapped_policy


def _wrap_selective(
    module: nn.Module, cfg: SelectiveACConfig, *, base_fqn: str | None = None
) -> nn.Module:
    """Wrap one block with the selective policy (upstream's ``_wrap_block``)."""
    save_ops = _get_default_save_ops()
    mm_recompute_shapes = _mm_recompute_shapes(
        module, base_fqn, cfg.force_recompute_mm_shapes_by_fqns
    )
    policy = _selective_policy(save_ops, mm_recompute_shapes)
    return ptd_checkpoint_wrapper(
        module,
        context_fn=lambda: create_selective_checkpoint_contexts(policy),
        preserve_rng_state=cfg.preserve_rng_state,
        determinism_check=cfg.determinism_check,
        early_stop=False,
        debug=cfg.debug,
    )


def apply_ac(
    model: nn.Module,
    mode: str = "none",
    *,
    selective: SelectiveACConfig | None = None,
    preserve_rng_state: bool = True,
) -> nn.Module:
    """Wrap every decoder layer of ``model`` in a checkpoint wrapper.

    No-op when ``mode == "none"`` (the model is handed back untouched), so the
    caller needs no mode check of its own.

    ``mode == "full"`` checkpoints the whole layer: its forward activations are
    dropped and recomputed during backward. ``preserve_rng_state`` is this
    mode's knob; the default restores the RNG state for the recompute, which is
    what keeps a full-AC run bitwise-equal to an uncheckpointed one.

    ``mode == "selective"`` needs ``selective``, the ``SelectiveACConfig``
    carrying that policy's settings (it has its own ``preserve_rng_state``, so
    the two modes never share one). Any other mode is a loud error.

    Apply after TP/EP/CP and before compile/FSDP (torchtitan's order in
    ``parallelize_llama``): the wrapper must enclose the TP-sharded layer, and
    FSDP has to wrap the checkpointed block so the recompute runs with
    all-gathered parameters instead of re-triggering the gather.
    """
    if mode == "none":
        return model
    if mode not in VALID_AC_MODES:
        raise ValueError(
            f"Unknown activation checkpointing mode {mode!r}; expected one of "
            f"{VALID_AC_MODES}."
        )
    if mode == "selective" and selective is None:
        raise ValueError(
            "mode='selective' needs the SelectiveACConfig that carries its "
            "save set and rng/determinism settings; pass "
            "selective=cfg.training.selective_ac."
        )

    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"apply_ac expects a HFTransformerModel (with .layers); got "
            f"{type(model).__name__}."
        )

    for layer_id, transformer_block in layers.named_children():
        if mode == "selective":
            wrapped = _wrap_selective(
                transformer_block, selective, base_fqn=f"layers.{layer_id}"
            )
        else:
            wrapped = ptd_checkpoint_wrapper(
                transformer_block,
                preserve_rng_state=preserve_rng_state,
                early_stop=False,
            )
        layers.register_module(layer_id, wrapped)

    logger.info("Applied %s activation checkpointing to %d layers", mode, len(layers))
    return model
