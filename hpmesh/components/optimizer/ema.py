"""Online EMA of model weights, driven as a gradient-free pseudo-optimizer.

Vendored from torchtitan's ``components/optimizer/ema.py``. ``EMA`` subclasses
``OptimizersContainer`` -- it is never a real training optimizer -- to reuse its
flat, FQN-keyed, resharding-safe ``state_dict()``/``load_state_dict()``: the EMA
checkpoint key (``ema``) therefore has exactly the layout optimizer state
already follows (``state.{fqn}.ema_params``), which is what lets DCP reshard it
across world sizes with no bespoke format.

Semantics, kept from upstream:

* **Update.** On every firing, ``ema = decay * ema + (1 - decay) * param``,
  implemented as ``torch._foreach_lerp_(ema_params, params, 1 - decay)``.
* **Decay.** Either fixed (``decay``), or dynamic (the default): with
  ``num_updates`` the count of firings so far,
  ``decay = 2 ** (-1 / (half_life_fraction * num_updates))``, which keeps
  roughly the most recent ``half_life_fraction`` share of updates dominant.
  The count is *derived from the trainer step* rather than stored, so the
  schedule survives a checkpoint resume: only ``ema_params`` are checkpointed,
  and a stored counter would restart at 0, collapsing the decay to a full
  overwrite of the restored average on the first firing after a resume.
* **Tracked set.** Parameters with ``requires_grad`` at construction time, plus
  the floating-point buffers matched by ``buffer_patterns`` (e.g. an MoE's
  ``expert_bias_E``, updated by a non-gradient heuristic). The set is keyed by
  tensor identity and fixed at construction: a parameter or buffer that appears
  or is replaced afterwards raises rather than silently going unaveraged.
* **Cold start.** ``load_state_dict({})`` reseeds the average from the live
  model weights. The checkpointer calls this when a load restored the model
  but not the EMA (an ``exclude_from_loading=["ema"]`` load, or a model-only
  load), which is also the path for turning EMA on mid-run against a checkpoint
  that predates it.

Departures from upstream, all subtractions:

* **No ``Configurable``.** The knobs are explicit constructor arguments; their
  validation lives in ``hpmesh.trainer.config.EMAConfig``, where every hpmesh
  config lives, and the trainer passes the fields across.
* **No ``offload_to_cpu``.** Upstream's offload path is a CUDA side-stream
  H2D/D2H pipeline over pinned memory, tuned for GH200's NVLink-C2C. hpmesh
  runs CPU and NPU, has no use for it, and cannot exercise it in this
  environment, so it is not ported; the EMA tensors simply live where the
  parameters live. With it go the DTensor unwrap/rewrap dance at save/load --
  the inherited container state dict hands DCP the live ``ema_params`` tensors,
  which under FSDP2 are DTensors already.
"""

import re
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ...utils.logger_utils import get_logger
from ..checkpointer.utils import canonical_fqn
from .optimizer import OptimizersContainer

__all__ = ["EMA"]

logger = get_logger(__name__)


class _EMAParamOptimizer(Optimizer):
    """Holds ``state[t]["ema_params"]`` per tensor (parameter or buffer) for
    one model part.

    Never step()-ed; reuses ``Optimizer``'s per-tensor state dict plus the
    FQN-flattening DCP machinery in ``optimizer/utils.py`` instead of a
    bespoke state-dict format. Also used for buffer EMA (e.g. MoE's
    ``expert_bias_E``), which is why tensors need not be ``nn.Parameter``s.
    """

    def __init__(self, named_tensors: Sequence[tuple[str, torch.Tensor]]) -> None:
        tensors = [t for _, t in named_tensors]
        names = [canonical_fqn(name) for name, _ in named_tensors]
        super().__init__([{"params": tensors, "param_names": names}], {})
        for t in tensors:
            self.state[t]["ema_params"] = t.detach().clone()

    def step(self, closure=None) -> None:
        raise RuntimeError(
            "_EMAParamOptimizer must not be step()-ed; call "
            "EMA.step(current_step) instead."
        )


class EMA(OptimizersContainer):
    """Pseudo-optimizer maintaining an online EMA of model weights.

    Subclasses ``OptimizersContainer`` to reuse its FQN-flattened,
    resharding-safe ``state_dict()``/``load_state_dict()`` while overriding
    ``__init__``/``step()``/``zero_grad()`` -- this is never a real training
    optimizer. Never merged into ``Trainer.optimizer`` or the LR scheduler
    container -- it is a sibling object, only built when the run configures
    one (``training.ema_config``), and stepped explicitly from
    ``train_step()``.

    Args:
        model_parts: the model chunks to track, one per pipeline stage.
        decay: fixed decay per firing, or None (default) to compute it
            dynamically from ``half_life_fraction``.
        half_life_fraction: used when ``decay`` is None:
            ``decay = 2 ** (-1 / (half_life_fraction * num_updates))``.
        start_step: last trainer step before EMA tracking begins, so the first
            update fires at ``start_step + update_every_n_steps``.
        step_bias: offset added to the firing count when computing
            ``num_updates``, for renumbering a new training phase (e.g. a
            restart that resets the trainer step) without resetting EMA aging.
            Measured in EMA firings, not raw steps. A normal resume needs no
            bias -- the firing count already continues correctly on its own.
        update_every_n_steps: only fire the EMA update every N optimizer steps.
        buffer_patterns: regex patterns (``re.search``, matched against buffer
            FQNs from ``model.named_buffers()``) selecting which buffers also
            get an EMA tracked alongside trainable parameters. Empty (default):
            no buffers tracked.
    """

    def __init__(
        self,
        *,
        model_parts: list[nn.Module],
        decay: float | None = None,
        half_life_fraction: float = 0.05,
        start_step: int = 0,
        step_bias: int = 0,
        update_every_n_steps: int = 1,
        buffer_patterns: Sequence[str] = (),
    ) -> None:
        self.decay = decay
        self.half_life_fraction = half_life_fraction
        self.start_step = start_step
        self.step_bias = step_bias
        self.update_every_n_steps = update_every_n_steps
        self.model_parts = model_parts

        self._param_optimizers: list[_EMAParamOptimizer] = []
        all_params: list[nn.Parameter] = []
        for model in model_parts:
            named_params = [
                (name, p) for name, p in model.named_parameters() if p.requires_grad
            ]
            self._param_optimizers.append(_EMAParamOptimizer(named_params))
            all_params.extend(p for _, p in named_params)
        self._validate_params(all_params)

        self._buffer_patterns = [re.compile(p) for p in buffer_patterns]
        self._buffer_optimizers: list[_EMAParamOptimizer] = []
        if self._buffer_patterns:
            total_matched = 0
            for model in model_parts:
                named_buffers = [
                    (name, b)
                    for name, b in model.named_buffers()
                    if any(p.search(name) for p in self._buffer_patterns)
                ]
                for name, b in named_buffers:
                    if not (torch.is_floating_point(b) or torch.is_complex(b)):
                        raise ValueError(
                            f"EMA buffer_patterns matched buffer {name!r} of "
                            f"dtype {b.dtype}. Only floating-point and complex "
                            "buffers can be averaged: an integer or boolean "
                            "average has to be rounded back into the buffer's "
                            "dtype, which freezes it once the per-step increment "
                            "falls below one."
                        )
                total_matched += len(named_buffers)
                self._buffer_optimizers.append(_EMAParamOptimizer(named_buffers))
            if total_matched == 0:
                logger.warning(
                    "EMA buffer_patterns=%s matched no buffers across any "
                    "model part -- buffer EMA is configured but silently "
                    "tracking nothing. Check the patterns for typos.",
                    list(buffer_patterns),
                )

        # OptimizersContainer.state_dict()/load_state_dict() (reused as-is)
        # iterate self.optimizers and merge each one's FQN-keyed flat dict, so
        # folding _buffer_optimizers in here is what gives buffer EMA the same
        # "ema" checkpoint key as parameters.
        self.optimizers: list[_EMAParamOptimizer] = (
            self._param_optimizers + self._buffer_optimizers
        )
        low_precision = sorted(
            {
                str(param_state["ema_params"].dtype)
                for ema_opt in self.optimizers
                for param_state in ema_opt.state.values()
                if param_state["ema_params"].dtype
                not in (torch.float32, torch.float64, torch.complex64, torch.complex128)
            }
        )
        if low_precision:
            logger.warning(
                "EMA is tracking %s tensors. Once the average and the live "
                "value are close, the per-firing increment rounds away at that "
                "precision and the EMA stops tracking altogether. Keep the "
                "tracked tensors in float32 (FSDP2's mixed_precision_param "
                "already does).",
                ", ".join(low_precision),
            )

        self._post_init(all_params)

    def zero_grad(self, *args, **kwargs) -> None:
        pass  # never called by the training loop; no-op for safety

    # Takes the step rather than an optimizer closure: the firing count has to
    # be derived from it, and this is never merged into Trainer.optimizer.
    @torch.no_grad()
    def step(self, current_step: int) -> None:
        """Call directly with the trainer's global step -- never merged into
        the optimizer container, so there is no closure/zero-arg step() to
        honor."""
        elapsed = current_step - self.start_step
        if elapsed <= 0 or elapsed % self.update_every_n_steps != 0:
            return
        # num_updates is the firing count (1, 2, 3, ...), derived from
        # current_step rather than kept as a counter so that it survives a
        # checkpoint resume: only ema_params are checkpointed, so a stored
        # counter would restart at 0 and the decay would collapse to a full
        # overwrite of the restored EMA. step_bias is added after the division
        # so its value is never truncated by update_every_n_steps.
        num_updates = elapsed // self.update_every_n_steps + self.step_bias
        self._update(num_updates)

    def _decay_at(self, num_updates: int) -> float:
        if self.decay is not None:
            return self.decay
        return 2.0 ** (-1.0 / (self.half_life_fraction * num_updates))

    # TODO: params/buffers are re-derived from the model on every firing
    # (model.parameters()/model.named_buffers() + regex matching for
    # buffers), even though the identical lists are already sitting in
    # ema_opt.param_groups[0]["params"] from construction -- EMA state is
    # already looked up by tensor identity, so identity (and thus this list)
    # is already assumed stable across the run. Reusing param_groups[0]
    # instead of recomputing would remove this per-step traversal/regex cost
    # with no behavior change.
    def _update(self, num_updates: int) -> None:
        decay = self._decay_at(num_updates)
        for ema_opt, model in zip(
            self._param_optimizers, self.model_parts, strict=True
        ):
            params: list[torch.Tensor] = [
                p for p in model.parameters() if p.requires_grad
            ]
            self._update_group(ema_opt, params, decay)
        for ema_opt, model in zip(
            self._buffer_optimizers, self.model_parts, strict=True
        ):
            buffers: list[torch.Tensor] = [
                b
                for name, b in model.named_buffers()
                if any(p.search(name) for p in self._buffer_patterns)
            ]
            self._update_group(ema_opt, buffers, decay)

    def _update_group(
        self,
        ema_opt: "_EMAParamOptimizer",
        tensors: list[torch.Tensor],
        decay: float,
    ) -> None:
        """Shared lerp/decay body for one model part's params or buffers."""
        if not tensors:
            return
        # State is keyed by tensor identity and the tracked set is fixed at
        # construction (see the TODO on _update), so a tensor that appeared or
        # was replaced since then has no entry. state is a defaultdict, so
        # indexing it here would insert an empty entry and fail later with a
        # bare KeyError("ema_params").
        untracked = [t for t in tensors if "ema_params" not in ema_opt.state.get(t, {})]
        if untracked:
            raise RuntimeError(
                f"EMA has no state for {len(untracked)} of {len(tensors)} "
                f"tensors (first: shape {tuple(untracked[0].shape)}, dtype "
                f"{untracked[0].dtype}). EMA tracks the tensors that existed "
                "when it was built, by identity, so unfreezing a parameter, "
                "replacing a parameter or buffer, or rebuilding part of the "
                "model after the EMA is constructed is not supported. Build "
                "the EMA after the model is final."
            )
        ema_params = [ema_opt.state[t]["ema_params"] for t in tensors]
        torch._foreach_lerp_(ema_params, tensors, 1.0 - decay)

    # -- checkpointing ----------------------------------------------------------

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict:
            # ``Optimizer.load_state_dict`` replaces each param group with the
            # loaded one, which drops ``param_names`` -- the extra key the flat
            # save path (``get_flat_optim_state_dict``) reads the FQNs back
            # from. Snapshot and reattach them so a load leaves the container
            # saveable again. The DCP resume path never calls this (it writes
            # into the tensors ``state_dict()`` reports), but the direct path
            # must not corrupt the group.
            param_names = [
                [group["param_names"] for group in ema_opt.param_groups]
                for ema_opt in self.optimizers
            ]
            super().load_state_dict(state_dict)
            for ema_opt, groups_names in zip(self.optimizers, param_names, strict=True):
                for group, names in zip(
                    ema_opt.param_groups, groups_names, strict=True
                ):
                    group["param_names"] = names
            return
        # The checkpoint had no EMA data (excluded via exclude_from_loading,
        # or predates this feature) -- cold-start from the just-loaded model
        # weights (and buffers, if buffer_patterns is set).
        for ema_opt, model in zip(
            self._param_optimizers, self.model_parts, strict=True
        ):
            for p in (p for p in model.parameters() if p.requires_grad):
                ema_opt.state[p]["ema_params"].copy_(p.detach())
        for ema_opt, model in zip(
            self._buffer_optimizers, self.model_parts, strict=True
        ):
            for name, b in model.named_buffers():
                if any(p.search(name) for p in self._buffer_patterns):
                    ema_opt.state[b]["ema_params"].copy_(b.detach())
        logger.warning(
            "EMA state was not restored; cold-starting it from the loaded "
            "model weights, which discards all EMA history. Expected the "
            "first time EMA is enabled against an older checkpoint. If "
            'checkpoint.exclude_from_loading still lists "ema", remove it, '
            "or every later resume will discard the EMA again."
        )
