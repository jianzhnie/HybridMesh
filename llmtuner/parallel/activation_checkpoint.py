"""Activation checkpointing: recompute each decoder layer during backward.

Vendored in shape from torchtitan's ``distributed/activation_checkpoint.py``.
All four of its policies are ported:

* ``"full"`` (upstream ``FullAC``) wraps each decoder layer in torch's
  non-reentrant ``checkpoint_wrapper``, so a forward keeps only the layer's
  inputs and recomputes its activations inside backward -- one extra forward
  per layer in exchange for the layer's activation memory. Upstream routes this
  through a selective checkpoint context carrying a constant
  ``PREFER_RECOMPUTE`` policy (``_full_ac_policy``), which is what lets torch
  still *save* an op whose output cannot be recomputed -- a registered effect --
  rather than replay it blindly; ``wrap_full``/``full_policy`` here are the same
  two pieces.

* ``"selective"`` (upstream ``SelectiveAC``) is per-op: a ``context_fn`` policy
  is asked about every op inside the layer and answers ``MUST_SAVE`` for the
  ones in the save set (``get_default_save_ops`` -- matmuls, SDPA and flex
  attention, and the collectives whose outputs are expensive to resend) and
  ``PREFER_RECOMPUTE`` for the rest. Matmuls in the save set are recomputed
  every second time instead of always, which is the memory/compute dial. One
  op is dropped from upstream's set; ``get_default_save_ops`` says which and
  why, and it is the one place that behavioural difference lives.

* ``"memory_budget"`` (upstream ``MemoryBudgetAC``) wraps nothing: it sets one
  process-global, ``torch._functorch.config.activation_memory_budget``, and
  the compile partitioner trades compute for memory inside each compiled
  region (0.0 is full-checkpointing memory, 1.0 the runtime-optimized
  default). It therefore only means anything when the model is compiled --
  selecting it with compile off is a config error, as upstream validates --
  and on a torch whose ``torch._functorch.config`` has no
  ``activation_memory_budget`` knob it refuses loudly rather than setting a
  global nothing reads. Upstream's ``visualize_memory_budget_pareto`` (an SVG
  dump into a trainer dump folder) is not ported; llmtuner's AC path has no
  dump folder.

* ``"region"`` (upstream ``RegionAC``) keeps the block's declared regions and
  recomputes the rest, using the optional ``torch_remat`` package (imported at
  apply time; it needs torch >= 2.10, so this mode is the one policy llmtuner
  cannot exercise on its development torch). Upstream gets its region names
  from ``torch_remat.region`` call sites in its own model code, wired by
  ``Module.configure_remat_regions``; HF models carry neither, so
  ``remat_regions`` derives the same vocabulary structurally -- the block's
  ``nn.Linear``s -- and ``wrap_region`` annotates them and then checkpoints the
  block's forward. ``wrap_region``'s docstring and ``remat_regions``' state the
  naming rules and what the vocabulary deliberately leaves out.

``"full"`` and ``"selective"`` use the same wrapper factory as upstream
(``torch.distributed.algorithms._checkpoint.checkpoint_wrapper``) and the same
``early_stop`` setting as upstream, which since upstream #4836 is the torch
default ``True``: the recompute stops as soon as every needed tensor is
produced instead of replaying the rest of the region. The old ``False`` was a
workaround for an upstream llama4 memory leak that no longer applies, and
carried a 1-4% step-time cost. ``"full"`` additionally keeps
``preserve_rng_state=True`` by default, so the recompute sees the RNG state
the original forward saw and the run stays bitwise-equal to the
uncheckpointed one. (``"region"`` wraps with ``torch_remat``'s own checkpoint
instead, which has no ``early_stop`` knob to set.)

Not ported, deliberately: upstream's RegionAC has no other moving parts. The
region *declaration* channel is the one piece llmtuner had to reinvent rather
than copy, because ``Module.configure_remat_regions`` presumes model code
llmtuner does not own; see ``remat_regions``.

Upstream's ``disable_dynamo_lru_cache`` IS ported, because the case it fixes
is reachable here: activation checkpointing applies on the ``pp > 1`` path too
(per stage chunk, upstream's ``ac_config``-per-model-part order). It is the
only place that touches that process-global knob, it runs only when a mode is
actually applied, and it is capability-guarded -- the knob is absent from the
torch 2.2.2 build this repo develops against, where the function logs and
returns instead of failing the run.

The remaining deviation from upstream is the entry point: torchtitan selects a
policy by instantiating an ``ActivationCheckpointing`` subclass, llmtuner by
passing a mode string (the extension point its config already documents).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch._functorch.config
import torch.nn as nn
from torch._functorch.partitioners import get_default_op_list
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from llmtuner.config import (
    VALID_AC_MODES,
    MemoryBudgetACConfig,
    RegionACConfig,
    SelectiveACConfig,
)

from ..accelerator.capabilities import has
from ..utils.logger_utils import get_logger
from .remat_regions import (
    region_names,
    region_policy,
    unmatched_save_patterns,
)

logger = get_logger(__name__)

# ``VALID_AC_MODES`` is declared next to the field it constrains
# (``config/training.py``) and imported here. It stays in ``__all__`` as a
# re-export so ``from llmtuner.parallel.activation_checkpoint import
# VALID_AC_MODES`` keeps working.
__all__ = [
    "VALID_AC_MODES",
    "apply_ac",
    "full_policy",
    "require_torch_remat",
    "wrap_full",
    "wrap_region",
]


def get_default_save_ops() -> set:
    """The ops whose activations ``"selective"`` saves rather than recomputes.

    Two sources, ported from upstream: torch's own list of compute-intensive
    ops (``get_default_op_list``), plus the explicit sets below. Each spec in
    those sets is either an op handle (always present) or a ``(root, dotted
    path)`` pair for an op that only exists in some builds -- resolved through
    ``getattr`` and skipped when it is not registered, so the same list runs
    against a CPU-only torch.

    The comm ops at the end are worth keeping pointed at even though llmtuner
    runs no DeepEP/HybridEP: they resolve-or-skip, so their absence is silent
    here while their presence (a future EP backend) is exactly when
    re-communication would be the thing to avoid.

    ``aten.topk`` is upstream's one deliberate omission. It saves topk because
    topk can be non-deterministic, so a recompute can pick different experts
    than the forward did. llmtuner cannot save it: HF's MoE routers normalize
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

    def resolve(op_specs: list) -> set:
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
    save_ops.update(resolve(compute_ops))
    save_ops.update(resolve(comm_ops))
    return save_ops


def mm_recompute_shapes(
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
        # ``shape[-1]`` / ``numel() // in_f`` instead of unpacking a 2D shape,
        # so a stacked weight (extra leading projection dim) resolves to the
        # same (in, out) pair its per-projection GEMMs run at.
        in_f = submod.weight.shape[-1]
        out_f = submod.weight.numel() // in_f
        shapes.add((in_f, out_f))
    return shapes


# Some backends (e.g. PrivateUse1) register aten.linear as a leaf op instead of
# decomposing it into aten.mm, so both spellings have to be handled.
_MM_OPS = (
    torch.ops.aten.mm.default,
    torch.ops.aten.mm.dtype,
    torch.ops.aten.linear.default,
)


def selective_policy(
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


def wrap_selective(
    module: nn.Module, cfg: SelectiveACConfig, *, base_fqn: str | None = None
) -> nn.Module:
    """Wrap one block with the selective policy (upstream's ``_wrap_block``)."""
    save_ops = get_default_save_ops()
    mm_shapes = mm_recompute_shapes(
        module, base_fqn, cfg.force_recompute_mm_shapes_by_fqns
    )
    policy = selective_policy(save_ops, mm_shapes)
    return ptd_checkpoint_wrapper(
        module,
        context_fn=lambda: create_selective_checkpoint_contexts(policy),
        preserve_rng_state=cfg.preserve_rng_state,
        determinism_check=cfg.determinism_check,
        early_stop=True,
        debug=cfg.debug,
    )


def full_policy(_ctx, _op, *_args, **_kwargs) -> CheckpointPolicy:
    """Prefer recompute for every op, letting torch keep registered effects.

    Upstream's ``_full_ac_policy``, verbatim: the policy that module-level full
    AC hands to the *selective* checkpoint context. ``PREFER_RECOMPUTE`` still
    lets torch fall back to saving an op whose output cannot be recomputed (a
    registered effect), which is what this constant policy buys over running the
    wrapper with no policy at all -- the two arguments are the op's context and
    handle and are unused, as upstream's are.
    """
    return CheckpointPolicy.PREFER_RECOMPUTE


def wrap_full(module: nn.Module, *, preserve_rng_state: bool = True) -> nn.Module:
    """Wrap one block with the full policy (upstream's ``FullAC._wrap_block``).

    Upstream does not hand the wrapper a bare block: it passes ``full_policy``
    through a selective checkpoint context, so full AC runs on the same
    machinery as ``"selective"`` and torch preserves the outputs of ops with
    registered effects instead of recomputing them blindly.
    ``determinism_check`` and ``debug`` keep torch's defaults -- ``"default"``
    and ``False``, which are upstream's own defaults for this policy -- and
    llmtuner does not expose them for this mode (registered in the symbol
    guide). ``early_stop`` matches upstream #4836.
    """
    return ptd_checkpoint_wrapper(
        module,
        context_fn=lambda: create_selective_checkpoint_contexts(full_policy),
        preserve_rng_state=preserve_rng_state,
        early_stop=True,
    )


def disable_dynamo_lru_cache() -> None:
    """Select dynamo graphs in insertion order (upstream's SAC+PP workaround).

    With activation checkpointing and pipeline parallelism together, a second
    microbatch's forward recompiles with dynamic shapes enabled, so two valid
    compiled graphs exist for the same region. Dynamo's default latest-wins
    (LRU) selection can then hand back the one whose runtime wrapper expects
    an extra symint output, which SAC's cached inductor-HOP output does not
    carry, and the assertion fails; insertion order avoids it. See
    https://github.com/pytorch/pytorch/issues/166926.

    Upstream calls this at the top of every policy's ``apply``, not only on
    the PP path, and so does this module -- but only when a mode is actually
    being applied, and only where the knob exists. It is a process-global,
    private torch setting: this is the one place that touches it.
    """
    if not has("dynamo_lru_cache"):
        logger.info(
            "torch %s has no torch._C._dynamo.eval_frame._set_lru_cache; "
            "activation checkpointing runs without upstream's SAC + "
            "pipeline-parallel cache workaround.",
            torch.__version__,
        )
        return
    # pyrefly: ignore [missing-attribute]
    torch._C._dynamo.eval_frame._set_lru_cache(False)


def apply_memory_budget(cfg: MemoryBudgetACConfig) -> None:
    """Set the one global ``"memory_budget"`` consists of (upstream's
    ``MemoryBudgetAC.apply``).

    The partitioner reads ``activation_memory_budget`` at compile time, so on
    a torch without that knob setting it would be a silent no-op -- refuse
    instead. Upstream never restores the global; it is a per-run setting.
    Upstream's ``apply`` sets the dynamo cache selection before touching the
    budget; the same order here, but after the check, so a config this torch
    cannot honour mutates nothing.
    """
    if not has("functorch_activation_memory_budget"):
        raise NotImplementedError(
            "mode='memory_budget' needs "
            "torch._functorch.config.activation_memory_budget, which this "
            f"torch ({torch.__version__}) does not have; the budget would be "
            "a global nothing reads."
        )
    disable_dynamo_lru_cache()
    torch._functorch.config.activation_memory_budget = cfg.memory_budget
    logger.info("Selected %s memory budget option", cfg.memory_budget)


def require_torch_remat():
    """Import the optional ``torch_remat`` package, or fail with the unlock.

    Imported here rather than at module scope for the same reason the
    checkpointer's backend imports are deferred: the package is optional and
    needs ``torch >= 2.10``, far ahead of what llmtuner otherwise runs on, so
    every caller of this module -- ``parallelize``, the trainer, the tests --
    stays importable without it. Reaching the region policy without the package
    is a loud ``ImportError`` naming the install command, at the point the
    dependency is actually needed.
    """
    try:
        import torch_remat
    except ImportError as error:
        raise ImportError(
            "activation_checkpoint_mode='region' needs the optional "
            "`torch_remat` package, which is not installed. It also requires "
            "torch >= 2.10. Install it with: pip install \"torch_remat @ "
            'git+https://github.com/meta-pytorch/remat.git" -- or use '
            "activation_checkpoint_mode='full'/'selective'."
        ) from error
    return torch_remat


def wrap_region(
    block: nn.Module,
    cfg: RegionACConfig,
    *,
    base_fqn: str,
) -> nn.Module:
    """Wrap one block with the region policy (upstream's ``RegionAC``).

    Upstream mutates the block in place: its ``apply`` sets ``module.forward``
    to a ``torch_remat.checkpoint``-wrapped forward, after
    ``Module.configure_remat_regions`` has pushed the save patterns into the
    module tree that the model's own ``remat.region`` call sites read. HF models
    carry no such call sites, so this stands in for both halves at once: it
    annotates the block's ``nn.Linear``s as the regions -- see
    ``remat_regions`` for that vocabulary and why it is upstream's policy in
    HF's spelling -- and then checkpoints the block's forward.

    Two naming rules, both deliberate:

    * ``cfg.save_regions`` patterns are matched against region names *relative
      to the block* (``self_attn.q_proj``). That is upstream's rule, and it is
      what lets one policy cover every block.
    * the label handed to ``torch_remat`` is block-qualified
      (``layers.0.self_attn.q_proj``), so a trace or a memory report can tell
      two blocks' regions apart. torch_remat treats the name as an opaque label
      -- the matching happens here -- and block-qualified names are unique
      within a forward, which is the only property its uniqueness rule asks for.

    The package is reached through ``require_torch_remat``, so a test can stand
    the real thing in for a ``sys.modules`` entry -- the same way the torchao
    adapter is tested.
    """
    remat = require_torch_remat()
    regions = region_names(block)
    policy = region_policy(regions, cfg.save_regions)
    for name, module in block.named_modules():
        if name in policy:
            module.forward = remat.region(
                module.forward, f"{base_fqn}.{name}", recompute=policy[name]
            )
    retained = sum(1 for recompute in policy.values() if not recompute)
    logger.info(
        "RegionAC on %s: %d regions, %d retained%s",
        base_fqn,
        len(regions),
        retained,
        f" ({', '.join(sorted(regions))})" if regions else "",
    )
    unmatched = unmatched_save_patterns(regions, cfg.save_regions)
    if unmatched:
        logger.warning(
            "RegionAC save_regions matched nothing in %s: %s (available: %s)",
            base_fqn,
            ", ".join(unmatched),
            ", ".join(sorted(regions)) or "none",
        )
    block.forward = remat.checkpoint(
        region_name=base_fqn,
        determinism_check=cfg.determinism_check,
        preserve_rng_state=cfg.preserve_rng_state,
    )(block.forward)
    return block


def apply_ac(
    model: nn.Module,
    mode: str = "none",
    *,
    selective: SelectiveACConfig | None = None,
    memory_budget: MemoryBudgetACConfig | None = None,
    region: RegionACConfig | None = None,
    compile_enabled: bool = False,
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
    the two modes never share one).

    ``mode == "memory_budget"`` wraps nothing: it needs ``memory_budget`` (the
    ``MemoryBudgetACConfig``) and ``compile_enabled=True``, and sets the
    process-global budget the compile partitioner later reads.

    ``mode == "region"`` needs ``region`` (the ``RegionACConfig``) and the
    optional ``torch_remat`` package, imported at apply time. It has no flat
    ``preserve_rng_state`` argument: torch_remat refuses ``True``, so the mode
    carries its own field, which can only stay ``False`` (see
    ``RegionACConfig``). Any other mode is a loud error.

    Apply after TP/EP/CP and before compile/FSDP (torchtitan's order in
    ``parallelize_llama``): the wrapper must enclose the TP-sharded layer, and
    FSDP has to wrap the checkpointed block so the recompute runs with
    all-gathered parameters instead of re-triggering the gather. For
    ``"memory_budget"`` the same call site is what guarantees the global is
    set before the compile it governs.
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
    if mode == "region" and region is None:
        raise ValueError(
            "mode='region' needs the RegionACConfig that carries its save "
            "patterns; pass region=cfg.training.region_ac."
        )
    if mode == "memory_budget":
        if memory_budget is None:
            raise ValueError(
                "mode='memory_budget' needs the MemoryBudgetACConfig that "
                "carries the budget; pass "
                "memory_budget=cfg.training.memory_budget_ac."
            )
        if not compile_enabled:
            raise ValueError(
                "mode='memory_budget' requires compile: the budget is "
                "consumed by the compile partitioner, so without "
                "torch.compile it would silently do nothing."
            )
        apply_memory_budget(memory_budget)
        return model

    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"apply_ac expects a HFTransformerModel (with .layers); got "
            f"{type(model).__name__}."
        )

    # Upstream sets this before wrapping, for every policy; so does the
    # memory_budget branch above (inside ``apply_memory_budget``). Placed
    # after the checks so a rejected config never mutates a global.
    disable_dynamo_lru_cache()

    for layer_id, transformer_block in layers.named_children():
        if mode == "selective":
            wrapped = wrap_selective(
                transformer_block, selective, base_fqn=f"layers.{layer_id}"
            )
        elif mode == "region":
            wrapped = wrap_region(
                transformer_block, region, base_fqn=f"layers.{layer_id}"
            )
        else:
            wrapped = wrap_full(
                transformer_block, preserve_rng_state=preserve_rng_state
            )
        layers.register_module(layer_id, wrapped)

    logger.info("Applied %s activation checkpointing to %d layers", mode, len(layers))
    return model
