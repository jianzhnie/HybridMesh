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

* ``colwise`` -- output features are split, i.e. the stored weight is cut on
  dim 0 into ``[out / tp, in]``. That is the ``w_shard_n = [N / R, K]`` contract
  ``AllGatherLinear`` documents: the op consumes the native layout and
  self-transposes inside the GEMM.
* ``rowwise`` -- input features are split, i.e. the stored weight is cut on dim 1,
  which is the layout ``LinearReduceScatter`` consumes as-is.

Activations stay sharded across the two: a column-parallel projection produces a
feature-sharded activation, which is exactly what the following row-parallel
projection consumes; the collectives are the sequence-parallel pair (all-gather
in, reduce-scatter out) fused into the GEMMs. This is the async-TP formulation,
not the older replicated-activation one -- the arithmetic is identical, but the
collective never materializes a full-sized activation and can overlap the matmul.

One site cannot host the fused gather: HF attention derives its q/k/v view
shapes from ``hidden_states.shape``, which a projection that physically
lengthens the sequence would silently mis-shape. Attention therefore takes the
same all-gather at the module boundary instead (``_GatherSequenceFirst`` --
numerically identical, just unfused), its q/k/v projections become plain
feature-sharded GEMMs (``ColwiseLinearNoGather``), and its o_proj keeps the
fused reduce-scatter, which returns the activation to the sequence shard. The
MLP has no such shape derivation, so gate/up/down keep the fused realizers.

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

from hpmesh.trainer.config import ParallelConfig

from .linear import (
    AllGatherLinear,
    LinearReduceScatter,
    all_gather_along,
    all_gather_linear,
    linear_reduce_scatter,
)

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

    Stores the weight as ``[out_features / tp, in_features]`` -- HF's own
    ``[out, in]`` layout cut on dim 0, which is the ``w_shard_n = [N / R, K]``
    layout ``AllGatherLinear`` contracts for (the op self-transposes).

    forward all-gathers the sequence shard and leaves the activation feature-
    sharded; backward is the dual (reduce-scatter of the input gradient, local
    weight gradient).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tp_size: int,
        tp_rank: int,
        group,
        use_symm_mem: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = nn.Parameter(
            _shard_weight(weight, 0, tp_size=tp_size, tp_rank=tp_rank)
        )
        self.group = group
        self.tp_size = tp_size
        self.use_symm_mem = use_symm_mem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The collective primitives are strictly 2D (sequence-major rows), but
        # the HF decoder feeds [B, T, K] hidden states: fold the leading dims
        # and restore them after, with the row count multiplied by tp_size --
        # that is the all-gathered sequence length.
        lead = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if self.use_symm_mem:
            y_2d = AllGatherLinear.apply(
                x_2d, self.weight, None, self.group, self.group.group_name
            )
        else:
            y_2d = all_gather_linear(x_2d, self.weight, self.group)
        return y_2d.reshape(*lead[:-1], lead[-1] * self.tp_size, -1)


class RowwiseLinear(nn.Module):
    """Row-parallel projection: input features split across TP ranks.

    Stores the weight as ``[out_features, in_features / tp]`` -- HF's layout cut
    on dim 1, which is what ``LinearReduceScatter`` expects.

    forward multiplies the feature-sharded activation and reduce-scatters the
    partial sums back to a sequence shard; backward is the dual (all-gather of the
    output gradient, local weight gradient).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tp_size: int,
        tp_rank: int,
        group,
        use_symm_mem: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = nn.Parameter(
            _shard_weight(weight, 1, tp_size=tp_size, tp_rank=tp_rank)
        )
        self.group = group
        self.tp_size = tp_size
        self.use_symm_mem = use_symm_mem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror of ColwiseLinear.forward: fold to 2D for the collective, then
        # restore the leading dims with the row count divided by tp_size --
        # that is the reduce-scattered sequence shard.
        lead = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if self.use_symm_mem:
            y_2d = LinearReduceScatter.apply(
                x_2d, self.weight, None, self.group, self.group.group_name
            )
        else:
            y_2d = linear_reduce_scatter(x_2d, self.weight, self.group)
        return y_2d.reshape(*lead[:-1], lead[-1] // self.tp_size, -1)


class ColwiseLinearNoGather(nn.Module):
    """Column-parallel projection without the fused sequence all-gather.

    Same weight shard as :class:`ColwiseLinear` (``[out / tp, in]``) but a plain
    local GEMM, for sites whose input is already full-sequence: the attention
    boundary gather (``_GatherSequenceFirst``) runs upstream, because HF
    attention derives q/k/v shapes from ``hidden_states`` and cannot absorb a
    projection whose output is physically longer than its input.

    The backward is exact without any collective here: the rowwise o_proj's
    backward all-gather reassembles the full-sequence, total-loss output
    gradient before attention's backward runs, so the local ``dy.T @ x`` is the
    complete weight gradient for this rank's feature shard.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tp_size: int,
        tp_rank: int,
        group,
        use_symm_mem: bool = True,
    ) -> None:
        super().__init__()
        del group, use_symm_mem  # no collective of its own; kept for the engine
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = nn.Parameter(
            _shard_weight(weight, 0, tp_size=tp_size, tp_rank=tp_rank)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class _GatherSequenceFirst:
    """Mixin that all-gathers the TP sequence shard before HF attention runs.

    Installed by ``apply_tp`` via a ``__class__`` swap (not a module wrapper),
    so module paths, ``state_dict`` keys and later attach points (``apply_cp``'s
    ``_titan_flex_kernel``) are all untouched. ``hidden_states`` arrives as this
    rank's ``[B, T / tp, K]`` sequence shard and is gathered along the sequence
    dim to the length the inner forward expects -- the full sequence, or the CP
    shard when CP is on (the gather spans the TP group only, and a TP group
    collectively holds exactly one CP shard).

    The gather's autograd dual is a reduce-scatter, which sums each rank's
    input-gradient contribution back to its own token shard -- the exact dual
    the sequence-parallel layout needs.
    """

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        gathered = all_gather_along(hidden_states, -2, self._tp_seq_group)
        return super().forward(gathered, *args, **kwargs)


def _looks_like_attention(module: nn.Module) -> bool:
    """HF attention modules hold q/k/v projections as direct attributes.

    This is the site test for the boundary gather: such a module reshapes its
    projections' outputs by the input's shape, so the gather must happen before
    it, not inside the projections. Probed structurally rather than by class
    name because HF spells the class differently per family (``LlamaAttention``,
    ``Qwen3Attention``, ...).
    """
    return all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj"))


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


def _resolve_plan(model: nn.Module, plan) -> dict[str, ShardingConfig | None]:
    """Normalize a plan into ``{module_path_pattern: ShardingConfig}``.

    ``plan`` may be ``None`` (use the model's declared plan), a map of patterns
    to ``ShardingConfig``, or a map of patterns to strings (the form HF ships).

    Three of HF's string specs map onto a realizer: ``colwise`` and ``rowwise``
    as below, plus ``replicated_with_grad_allreduce`` -- a projection left whole
    on every rank whose gradient the trainer's own
    ``_allreduce_replicated_tp_grads`` already sums. That last one is not
    decoration: Qwen3's plan marks ``q_norm`` / ``k_norm`` with it, and without
    this branch every Qwen3 TP run dies here before touching a weight.

    HF's MoE specs (``packed_colwise``, ``moe_tp_experts``) still raise. hpmesh
    does not shard MoE experts over the TP axis -- it has no fused-expert
    realizer -- so honouring them silently would be worse than refusing.

    When ``plan`` is omitted the model's own declaration is used, preferring the
    ``tp_plan`` property over the raw ``_tp_plan`` attribute: a wrapper that
    re-parents the HF model has to rewrite the patterns to its own module paths
    (see ``HFTransformerModel.tp_plan``), and reading the raw attribute on such a
    wrapper yields either nothing or patterns that match no module.
    """
    if plan is None:
        plan = getattr(model, "tp_plan", None) or getattr(model, "_tp_plan", None) or {}
    resolved: dict[str, ShardingConfig | None] = {}
    for pattern, spec in plan.items():
        if isinstance(spec, ShardingConfig):
            resolved[pattern] = spec
        elif spec == "colwise":
            resolved[pattern] = colwise()
        elif spec == "rowwise":
            resolved[pattern] = rowwise()
        elif spec == "replicated_with_grad_allreduce":
            # Nothing for apply_tp to do: the projection stays whole on every
            # rank. The ``_with_grad_allreduce`` half is already implemented --
            # _allreduce_replicated_tp_grads sums exactly these parameters'
            # gradients -- so this entry only has to be understood, not acted
            # on. Recorded as None rather than dropped so _match still stops
            # here instead of falling through to a broader later pattern.
            resolved[pattern] = None
        else:
            raise ValueError(f"Unsupported TP plan entry for {pattern!r}: {spec!r}")
    return resolved


def _match(
    plan: dict[str, ShardingConfig | None], module_path: str
) -> ShardingConfig | None:
    """The first pattern in ``plan`` that matches ``module_path``.

    The None entries are plans that deliberately declare a projection *not*
    sharded; for those first-match-wins is load-bearing, because ``_match``
    walks the plan in insertion order and stops at the first hit. Returning
    None from them is indistinguishable from "no pattern matched" to the
    caller, which is correct here -- both mean "leave this module alone" -- but
    it does mean a sharded pattern sitting *after* a replicated one in the plan
    can never win for the same path.
    """
    for pattern, spec in plan.items():
        if fnmatch.fnmatch(module_path, pattern):
            return spec
    return None


def _supports_symm_mem(tp_mesh: DeviceMesh) -> bool:
    """Whether the fused symmetric-memory TP collectives can run on this mesh.

    They are CUDA-only; anywhere else the modules fall back to the functional-
    collective realization of the same math (``all_gather_linear`` /
    ``linear_reduce_scatter``), which is what makes TP runnable -- and testable
    -- on CPU/gloo.
    """
    if tp_mesh.device_type != "cuda":
        return False
    try:
        import torch.distributed._symmetric_memory  # noqa: F401
    except ImportError:
        return False
    return True


def _enable_symm_mem(group) -> None:
    """Register ``group`` for symmetric-memory collectives.

    ``torch.ops.symm_mem.fused_all_gather_matmul`` (and its reduce-scatter dual)
    only work on a group registered here; PyTorch does not yet do this
    automatically for the TP group. CUDA-only -- call only when
    ``_supports_symm_mem`` held for the mesh.
    """
    import warnings

    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        enable_symm_mem_for_group(group.group_name)


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
    if not targets:
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
            if implementation is ColwiseLinear:
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
