"""The TP entry point: apply a sharding plan to a model.

Split out of ``tp.py`` (which keeps the declaration/realizer machinery) so
every parallelism family exposes the same ``apply.py`` entry-point shape --
``parallelize_hf`` can then import ``apply_tp`` the same way it imports
``apply_ep`` / ``apply_cp`` / ``apply_fsdp``.
"""

from __future__ import annotations

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from hpmesh.config import ParallelConfig

from .. import matrix
from .tp import (
    _MOE_PLAN_SPECS,
    ColumnParallelLinear,
    ColwiseLinearNoGather,
    ShardingConfig,
    _enable_symm_mem,
    _GatherSequenceFirst,
    _looks_like_attention,
    _match,
    _resolve_plan,
    _shard_experts_for_tp,
    _supports_symm_mem,
    _TPMoeSequenceBoundary,
)

__all__ = ["apply_tp"]


def apply_tp(
    model: nn.Module,
    mesh: DeviceMesh | None,
    cfg: ParallelConfig,
    plan=None,
) -> nn.Module:
    """Tensor-parallelize ``model`` in place. No-op when ``tp == 1``.

    ``plan`` defaults to the model's HF ``tp_plan`` (HF ships one for Qwen3,
    Llama, ...). Pass an explicit ``{path_pattern: ShardingConfig}`` to override
    it -- e.g. to leave a projection replicated or to use a different realizer.
    """
    if mesh is None or cfg.tp <= 1:
        return model

    # Validate the plan before touching the mesh: a plan that resolves to
    # nothing, or that matches no module, used to leave the model fully
    # replicated while the run reported a healthy TP setup. Both are loud
    # errors now, and both fire before any process-group access.
    sharding_plan = _resolve_plan(model, plan)
    if not sharding_plan:
        raise ValueError(
            f"apply_tp with tp={cfg.tp}: {type(model).__name__} provides no TP "
            "plan (neither a `tp_plan`/`_tp_plan` declaration nor an explicit "
            "`plan` argument). Refusing to run TP as a silently replicated "
            "model; shard the projections declaratively or set tp=1."
        )

    targets: list[tuple[str, nn.Linear, ShardingConfig]] = []
    for module_path, module in model.named_modules():
        if isinstance(module, nn.Linear):
            spec = _match(sharding_plan, module_path)
            if spec is not None:
                targets.append((module_path, module, spec))

    # MoE-under-TP is declared by spec strings that name no nn.Linear (the
    # expert weights are stacked parameters), so it has to be detected on the
    # RAW plan, before _resolve_plan maps those specs to None.
    raw_plan = plan
    if raw_plan is None:
        raw_plan = (
            getattr(model, "tp_plan", None) or getattr(model, "_tp_plan", None) or {}
        )
    plan_declares_moe = any(
        isinstance(spec, str) and spec in _MOE_PLAN_SPECS for spec in raw_plan.values()
    )
    # tp x ep: TP shards only the dense parts and EP owns the routed experts
    # (the upstream alignment). The HF blocks are left whole here and swapped
    # for the native MoE stack by apply_ep, which runs next; the swapped block
    # consumes and produces the T/tp sequence shard directly (all-to-all
    # dispatch), so no sequence-boundary collectives are installed either.
    moe_deferred_to_ep = plan_declares_moe and cfg.ep > 1
    moe_blocks: list[tuple[str, nn.Module]] = []
    already_bracketed = False
    if plan_declares_moe and not moe_deferred_to_ep:
        from ..expert_parallel.swap import _is_hf_moe_block

        for module_path, module in model.named_modules():
            if getattr(module, "_tp_moe_boundary", False):
                # Already bracketed by an earlier apply_tp pass (the engine is
                # idempotent per module): still counts as realized MoE TP.
                already_bracketed = True
                continue
            # Blocks the probe refuses (GPT-OSS's transposed, bias-bearing
            # experts) raise out of it here, before any weight is touched.
            if _is_hf_moe_block(module):
                moe_blocks.append((module_path, module))
        if not moe_blocks and not already_bracketed:
            matrix.tp_moe_specs_without_block(cfg.tp, model)
    nothing_sharded = not targets and not moe_blocks and not already_bracketed
    if nothing_sharded and not moe_deferred_to_ep:
        raise ValueError(
            f"apply_tp with tp={cfg.tp}: the plan patterns "
            f"{sorted(sharding_plan)} matched no nn.Linear on "
            f"{type(model).__name__}. The patterns are spelled relative to the "
            "module tree being parallelized; check the prefix (e.g. a wrapper's "
            "`model.` prefix) rather than training a replicated model by "
            "mistake."
        )

    group = mesh["tp"].get_group()
    tp_size = mesh["tp"].size()
    tp_rank = mesh["tp"].get_local_rank()
    use_symm_mem = _supports_symm_mem(mesh["tp"])
    if use_symm_mem:
        _enable_symm_mem(group)

    for module_path, block in moe_blocks:
        if getattr(block, "shared_expert", None) is not None or (
            getattr(block, "shared_experts", None) is not None
        ):
            matrix.shared_expert_tp(module_path, block)
        # The sharded expert parameters are excluded from the trainer's
        # replicated-gradient all-reduce through this id set: each rank's
        # F-shard gradient is complete, and summing it with a different
        # shard's gradient would corrupt it.
        block._tp_sharded_param_ids = _shard_experts_for_tp(
            block, tp_size=tp_size, tp_rank=tp_rank
        )
        block._tp_seq_group = group
        block._tp_moe_boundary = True
        block.__class__ = type(
            f"TPMoe{type(block).__name__}",
            (_TPMoeSequenceBoundary, type(block)),
            {},
        )

    # Deepest paths first, so replacing a module never hides an inner target.
    attention_parents: dict[str, nn.Module] = {}
    for module_path, inner, spec in sorted(targets, key=lambda t: -t[0].count(".")):
        if inner.bias is not None:
            raise ValueError(
                f"TP over {module_path} has a bias, which this minimal engine does "
                "not shard; HF decoder projections are bias-free."
            )
        parent_path, _, attr = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model

        implementation = spec.implementation
        if spec.kind == "colwise" and _looks_like_attention(parent):
            # HF attention reshapes q/k/v by the input's shape, so the fused
            # in-GEMM sequence gather would silently mis-shape them. Take the
            # same gather at the module boundary instead (below) and give the
            # projection a plain feature-sharded GEMM. Only the default
            # realizer is swapped out; an explicitly provided one is the
            # caller's responsibility.
            if implementation is ColumnParallelLinear:
                implementation = ColwiseLinearNoGather
            attention_parents[parent_path] = parent

        wrapped = implementation(
            inner.weight,
            tp_size=tp_size,
            tp_rank=tp_rank,
            group=group,
            use_symm_mem=use_symm_mem,
        )
        setattr(parent, attr, wrapped)

    for parent in attention_parents.values():
        if getattr(parent, "_tp_seq_group", None) is not None:
            continue  # already gathered (apply_tp is idempotent per module)
        parent._tp_seq_group = group
        parent.__class__ = type(
            f"TPGather{type(parent).__name__}",
            (_GatherSequenceFirst, type(parent)),
            {},
        )

    return model
