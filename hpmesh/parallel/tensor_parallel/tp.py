"""Step 2: tensor parallelism -- declarative sharding realized by fused TP GEMMs.

A tensor-parallel projection is two separate things: WHERE its weight is cut, and
HOW its activations move around the cut. This file keeps those apart.

* ``ShardingConfig`` (built by the ``colwise()`` / ``rowwise()`` factories) is the
  DECLARATION -- plain data, no tensors, no collectives.
* ``ColwiseLinear`` / ``RowwiseLinear`` are the modules that REALIZE a
  declaration, built on the fused collective+GEMM primitives in ``linear.py``.
* ``apply_tp`` is the ENGINE -- it reads a plan (the model's HF ``tp_plan`` by
  default, or an explicit ``{pattern: ShardingConfig}`` map), cuts each target
  weight and swaps the projection out for its sharded module.

Weight layout, in the ``nn.Linear`` convention ``weight: [out_features,
in_features]``:

* ``colwise`` -- output features are split. HF's stored ``[out, in]`` weight is
  transposed once and sharded over its now-last dim, because
  ``AllGatherLinear`` consumes ``[in_features, out_features]``.
* ``rowwise`` -- input features are split, i.e. the stored weight is cut on dim 1,
  which is the layout ``LinearReduceScatter`` consumes as-is.

Activations stay sharded across the two: a column-parallel projection produces a
feature-sharded activation, which is exactly what the following row-parallel
projection consumes; the collectives are the sequence-parallel pair (all-gather
in, reduce-scatter out) fused into the GEMMs. This is the async-TP formulation,
not the older replicated-activation one -- the arithmetic is identical, but the
collective never materializes a full-sized activation and can overlap the matmul.

Scope: this is the mechanism. No meta-init (weights come from the HF model as
usual), no fused QKV (HF keeps q/k/v as separate projections), no FP8.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from ...trainer.config import HybridMeshConfig
from .linear import AllGatherLinear, LinearReduceScatter

ShardKind = Literal["colwise", "rowwise"]


# -- declaration --------------------------------------------------------------


def _shard_weight(
    weight: torch.Tensor, dim: int, *, tp_size: int, tp_rank: int
) -> torch.Tensor:
    """Cut ``weight`` into ``tp_size`` pieces along ``dim``; keep this rank's.

    A plain slice with no collective: every rank starts from the same full weight
    (the model is built identically everywhere), so each just drops the rest.
    """
    if weight.shape[dim] % tp_size != 0:
        raise ValueError(
            f"weight dim {dim} (size {weight.shape[dim]}) is not divisible by "
            f"tp_size={tp_size}"
        )
    return torch.chunk(weight.detach(), tp_size, dim=dim)[tp_rank].contiguous()


class ColwiseLinear(nn.Module):
    """Column-parallel projection: output features split across TP ranks.

    Stores the weight as ``[in_features, out_features / tp]`` -- the layout
    ``AllGatherLinear`` expects -- so HF's ``[out, in]`` weight is transposed
    once at parallelize time, then cut on its last dim.

    forward all-gathers the sequence shard and leaves the activation feature-
    sharded; backward is the dual (reduce-scatter of the input gradient, local
    weight gradient).
    """

    def __init__(
        self, weight: torch.Tensor, *, tp_size: int, tp_rank: int, group
    ) -> None:
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        w_in_out = weight.detach().t().contiguous()
        self.weight = nn.Parameter(
            _shard_weight(w_in_out, 1, tp_size=tp_size, tp_rank=tp_rank)
        )
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return AllGatherLinear.apply(
            x, self.weight, None, self.group, self.group.group_name
        )


class RowwiseLinear(nn.Module):
    """Row-parallel projection: input features split across TP ranks.

    Stores the weight as ``[out_features, in_features / tp]`` -- HF's layout cut
    on dim 1, which is what ``LinearReduceScatter`` expects.

    forward multiplies the feature-sharded activation and reduce-scatters the
    partial sums back to a sequence shard; backward is the dual (all-gather of the
    output gradient, local weight gradient).
    """

    def __init__(
        self, weight: torch.Tensor, *, tp_size: int, tp_rank: int, group
    ) -> None:
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = nn.Parameter(
            _shard_weight(weight, 1, tp_size=tp_size, tp_rank=tp_rank)
        )
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LinearReduceScatter.apply(
            x, self.weight, None, self.group, self.group.group_name
        )


@dataclass(frozen=True)
class ShardingConfig:
    """How one projection is sharded on the TP axis.

    Attributes:
        kind: ``"colwise"`` or ``"rowwise"`` -- the direction the weight is cut.
        implementation: the module class that realizes this declaration. Paired
            here rather than looked up from ``kind`` so a caller can swap in a
            different realizer (e.g. a fused-QKV variant) without touching the
            engine.
    """

    kind: ShardKind
    implementation: type[nn.Module]

    def __post_init__(self) -> None:
        if self.kind not in ("colwise", "rowwise"):
            raise ValueError(f"Unknown shard kind {self.kind!r}")


def colwise() -> ShardingConfig:
    """Output features split; activations stay feature-sharded after the GEMM."""
    return ShardingConfig(kind="colwise", implementation=ColwiseLinear)


def rowwise() -> ShardingConfig:
    """Input features split; activations reduce-scatter back to a sequence shard."""
    return ShardingConfig(kind="rowwise", implementation=RowwiseLinear)


# -- engine -------------------------------------------------------------------


def _resolve_plan(model: nn.Module, plan) -> dict[str, ShardingConfig]:
    """Normalize a plan into ``{module_path_pattern: ShardingConfig}``.

    ``plan`` may be ``None`` (use the model's declared plan), a map of patterns
    to ``ShardingConfig``, or a map of patterns to ``"colwise"`` / ``"rowwise"``
    strings (the form HF ships).

    When ``plan`` is omitted the model's own declaration is used, preferring the
    ``tp_plan`` property over the raw ``_tp_plan`` attribute: a wrapper that
    re-parents the HF model has to rewrite the patterns to its own module paths
    (see ``HFTransformerModel.tp_plan``), and reading the raw attribute on such a
    wrapper yields either nothing or patterns that match no module.
    """
    if plan is None:
        plan = getattr(model, "tp_plan", None) or getattr(model, "_tp_plan", None) or {}
    resolved: dict[str, ShardingConfig] = {}
    for pattern, spec in plan.items():
        if isinstance(spec, ShardingConfig):
            resolved[pattern] = spec
        elif spec == "colwise":
            resolved[pattern] = colwise()
        elif spec == "rowwise":
            resolved[pattern] = rowwise()
        else:
            raise ValueError(f"Unsupported TP plan entry for {pattern!r}: {spec!r}")
    return resolved


def _match(plan: dict[str, ShardingConfig], module_path: str) -> ShardingConfig | None:
    for pattern, spec in plan.items():
        if fnmatch.fnmatch(module_path, pattern):
            return spec
    return None


def _enable_symm_mem(group) -> None:
    """Register ``group`` for symmetric-memory collectives.

    ``torch.ops.symm_mem.fused_all_gather_matmul`` (and its reduce-scatter dual)
    only work on a group registered here; PyTorch does not yet do this
    automatically for the TP group. CUDA-only, so -- like the fused ops
    themselves -- this only runs on a machine where TP can run at all.
    """
    import warnings

    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        enable_symm_mem_for_group(group.group_name)


def apply_tp(
    model: nn.Module,
    mesh: DeviceMesh | None,
    cfg: HybridMeshConfig,
    plan=None,
) -> nn.Module:
    """Tensor-parallelize ``model`` in place. No-op when ``tp == 1``.

    ``plan`` defaults to the model's HF ``tp_plan`` (HF ships one for Qwen3,
    Llama, ...). Pass an explicit ``{path_pattern: ShardingConfig}`` to override
    it -- e.g. to leave a projection replicated or to use a different realizer.
    """
    if mesh is None or cfg.tp <= 1:
        return model

    group = mesh["tp"].get_group()
    tp_size = mesh["tp"].size()
    tp_rank = mesh["tp"].get_local_rank()
    sharding_plan = _resolve_plan(model, plan)
    _enable_symm_mem(group)

    targets: list[tuple[str, nn.Linear, ShardingConfig]] = []
    for module_path, module in model.named_modules():
        if isinstance(module, nn.Linear):
            spec = _match(sharding_plan, module_path)
            if spec is not None:
                targets.append((module_path, module, spec))

    # Deepest paths first, so replacing a module never hides an inner target.
    for module_path, inner, spec in sorted(targets, key=lambda t: -t[0].count(".")):
        if inner.bias is not None:
            raise ValueError(
                f"TP over {module_path} has a bias, which this minimal engine does "
                "not shard; HF decoder projections are bias-free."
            )
        wrapped = spec.implementation(
            inner.weight, tp_size=tp_size, tp_rank=tp_rank, group=group
        )
        parent_path, _, attr = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, wrapped)

    return model
