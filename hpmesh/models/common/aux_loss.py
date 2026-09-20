"""Auxiliary-loss gradient injection and distributed metric collection.

Vendored from torchtitan ``models/common/aux_loss.py``. What changed:

* The ``Module`` protocol and its ``Config`` carrier are gone; ``AuxLoss``
  takes its coefficients as keyword args, and ``_init_self_buffers`` (a
  meta-device build hook) is dropped.
* ``torch_remat`` is gone. Upstream wraps the accumulation in a
  ``remat.region(..., recompute=False)`` so ``torch_remat``'s activation
  checkpointing retains the region instead of replaying the side effect. hpmesh
  does not depend on ``torch_remat``, so the accumulation is simply inline. The
  injected gradient is unaffected; what changes is that under *any* activation
  checkpointing the forward is replayed and the logged metric counts the
  microbatch once per replay. Treat the logged value as a relative signal under
  checkpointing, not an absolute one.
* The ``spmd_types`` blocks are gone. ``spmd_assert_type`` on the injection's
  output described a placement the framework already inferred.
* ``OptimizersContainer`` becomes a plain ``torch.optim.Optimizer`` (hpmesh's
  trainer owns one directly) and ``ParallelDims`` is hpmesh's.

Design, unchanged from upstream:

Normalization: every auxiliary loss is scaled by the step's global valid-token
count (``set_step_denominator``), the same denominator the main loss uses, so
contributions stay comparable across parallelism degrees. The per-step metric
is the mean over loss instances (layers) of that scaled value, summed over
data-parallel ranks and pipeline stages.

The metric accumulates during the model forward. A step pre-hook rolls the
per-instance accumulators into ``group_acc`` registers, and
``collect_aux_loss_metrics`` reduces those for logging.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn as nn
from torch.distributed._functional_collectives import all_reduce

from hpmesh.utils.device import device_type

if TYPE_CHECKING:
    # Annotation-only: ``ParallelDims`` appears in two signatures below and
    # nowhere at runtime. Importing it eagerly would form a cycle --
    # ``parallel/__init__`` -> ``expert_parallel.apply`` -> ``models.common.moe``
    # -> here -- which is entered whenever a model module is imported before
    # ``hpmesh.parallel``.
    from hpmesh.parallel.parallel_dims import ParallelDims

__all__ = [
    "AuxLoss",
    "collect_aux_loss_metrics",
    "register_aux_loss_zero_hook",
]


class _AuxLossInjection(torch.autograd.Function):
    """Identity forward that injects an aux-loss gradient on the backward pass.

    The carrier passes through untouched; its only job is to give the aux
    loss a differentiable path into the graph. On the way back, the aux loss
    contributes exactly 1.0 to its own gradient -- the ``coeff`` and the
    ``1/denominator`` scaling were already folded into the loss value, so the
    injection itself stays a constant.
    """

    @staticmethod
    def forward(ctx, carrier, aux_loss):  # pyrefly: ignore[bad-override]
        ctx.save_for_backward(aux_loss)
        return carrier

    @staticmethod
    def backward(ctx, grad_carrier):  # pyrefly: ignore[bad-override]
        (aux_loss,) = ctx.saved_tensors
        return grad_carrier, torch.ones_like(aux_loss)


class AuxLoss(nn.Module):
    """Base class: subclasses call ``inject()`` each forward.

    Each instance accumulates its scaled value in the ``instance_acc`` buffer; a
    step pre-hook rolls those into the ``group_acc`` registers, which
    ``collect_aux_loss_metrics`` reduces for logging.

    Args:
        coeff: scales the loss's gradient contribution.
        reduce_mesh: mesh the per-step metric is summed over -- ``"batch"``
            (dp) for cp-identical losses like the microbatch-wise load-balance
            loss, ``"loss"`` (dp+cp) for per-token-additive losses whose
            rank-local values add up across coordinates.

    Normalization: ``denominator`` is the step's global valid-token count, set
    by the trainer via ``set_step_denominator`` before the first forward.
    """

    # Metric groups are populated during model build, before PP splitting, so
    # every pipeline stage participates with its own (zero) accumulators and
    # every rank holds the same count. That count is also the divisor in
    # ``collect_aux_loss_metrics``: the per-rank sums are summed over the
    # reduce mesh and the pipeline stages, then divided by it, giving the mean
    # over all layers of the model.
    _group_counts: ClassVar[dict[tuple[str, str], int]] = defaultdict(int)

    # Global valid-token count of the current step, set by the trainer before
    # the first forward. Shared by all instances: the framework normalizes
    # every auxiliary loss by the same per-step count, matching the main loss,
    # so contributions are comparable across parallelism degrees.
    _step_denominator: ClassVar[torch.Tensor | None] = None

    # Per metric group (``(reduce_mesh, metric_name)``): this rank's total value
    # for the current step, rolled up from the per-instance ``instance_acc``
    # buffers by ``_zero_aux_losses`` at each optimizer step pre-hook and
    # reduced by ``collect_aux_loss_metrics`` at log time. Cleared at the next
    # pre-hook.
    group_acc: ClassVar[dict[tuple[str, str], torch.Tensor]] = {}

    def __init__(self, *, coeff: float, reduce_mesh: str = "batch") -> None:
        super().__init__()
        self.coeff = coeff
        self.reduce_mesh = reduce_mesh
        # Per-instance accumulator: the sum of this loss instance's scaled
        # per-microbatch values over the current training step. Filled in the
        # forward; rolled into ``group_acc`` and zeroed by ``_zero_aux_losses``
        # at each optimizer step pre-hook. Non-persistent: it is scratch, and
        # a checkpoint resuming mid-step should not restore a partial sum.
        self.register_buffer(
            "instance_acc", torch.zeros((), dtype=torch.float32), persistent=False
        )
        AuxLoss._group_counts[(self.reduce_mesh, self.metric_name)] += 1

    @property
    def metric_name(self) -> str:
        """The class name in snake_case, e.g. ``MicrobatchWiseLoadBalanceLoss``
        -> ``microbatch_wise_load_balance_loss``."""
        return re.sub(
            r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            "_",
            type(self).__name__,
        ).lower()

    @classmethod
    def set_step_denominator(cls, denominator: torch.Tensor) -> None:
        """Set the current step's global valid-token count.

        The trainer calls this once per step with the same dp-summed token count
        the main loss normalizes by, so auxiliary losses stay on the same scale
        as the main loss and independent of parallelism degrees.
        """
        cls._step_denominator = denominator

    def inject(self, raw_sum: torch.Tensor, *, carrier: torch.Tensor) -> torch.Tensor:
        """Inject the aux-loss gradient on ``carrier`` and accumulate the metric.

        Args:
            raw_sum: unnormalized per-microbatch loss value (differentiable).
            carrier: the tensor whose backward path carries the gradient.

        Returns:
            ``carrier`` unchanged.

        Raises:
            ValueError: if no step denominator has been set, which would make
                the loss's scale undefined.
        """
        if AuxLoss._step_denominator is None:
            raise ValueError(
                "AuxLoss.set_step_denominator() must be called with the "
                "step's global valid-token count before the first forward."
            )
        denominator = AuxLoss._step_denominator
        scale = 1.0 / denominator
        injected = raw_sum * (self.coeff * scale)
        # Accumulate the metric in the forward. The no_grad mask keeps the
        # buffer out of the autograd graph.
        with torch.no_grad():
            self.instance_acc.add_(raw_sum * scale)
        return _AuxLossInjection.apply(carrier, injected)


def _zero_aux_losses(model_parts: Sequence[nn.Module]) -> None:
    """Roll per-instance ``instance_acc`` into ``group_acc`` and clear them.

    Optimizer step pre-hook. Each metric group is summed once per model part
    that holds an instance, so a group spread across pipeline stages ends up
    with one register holding the step's total.
    """
    AuxLoss.group_acc.clear()
    for part in model_parts:
        for module in part.modules():
            if isinstance(module, AuxLoss):
                key = (module.reduce_mesh, module.metric_name)
                if key not in AuxLoss.group_acc:
                    AuxLoss.group_acc[key] = torch.zeros_like(module.instance_acc)
                AuxLoss.group_acc[key] += module.instance_acc
                module.instance_acc.zero_()


def collect_aux_loss_metrics(parallel_dims: ParallelDims) -> dict[str, float]:
    """Reduce the current step's ``group_acc`` registers for logging.

    Returns ``{metric_name}/mean`` per group, or ``{}`` when no auxiliary loss
    is configured. All ranks call this at log time.
    """
    if not AuxLoss._group_counts:
        return {}

    pp_mesh = parallel_dims.get_optional_mesh("pp")

    def _group_acc_or_zero(key: tuple[str, str]) -> torch.Tensor:
        value = AuxLoss.group_acc.get(key)
        if value is not None:
            return value
        # Ranks that own no instance of this group still join the collectives
        # below with a zero contribution.
        return torch.zeros((), dtype=torch.float32, device=device_type)

    group_accs = {key: _group_acc_or_zero(key) for key in AuxLoss._group_counts}
    metrics = {}
    for key, total in sorted(group_accs.items()):
        mesh_name, tag = key
        reduce_mesh = parallel_dims.get_optional_mesh(mesh_name)
        for mesh in (reduce_mesh, pp_mesh):
            if mesh is None:
                continue
            # Sum: each coordinate contributes its own data, and every layer
            # lives on exactly one pipeline stage, so summing over the reduce
            # mesh and the stages counts every layer once. Dividing by the
            # build-time instance count below (identical on every rank) then
            # gives the mean over all layers.
            total = all_reduce(total, reduceOp="sum", group=mesh)
        metrics[f"{tag}/mean"] = float(total.item()) / AuxLoss._group_counts[key]
    return metrics


def register_aux_loss_zero_hook(
    optimizer: torch.optim.Optimizer,
    model_parts: Sequence[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Register the step pre-hook that rolls ``instance_acc`` into ``group_acc``.

    ``parallel_dims`` is accepted for symmetry with the metric collection it
    feeds; the roll-up itself is local.

    TODO: drop the unused ``parallel_dims`` argument once a second caller
    exists -- it is kept so the call site reads the same as
    ``register_moe_load_balancing_hook``.
    """
    del parallel_dims
    optimizer.register_step_pre_hook(
        lambda *args, **kwargs: _zero_aux_losses(model_parts)
    )
