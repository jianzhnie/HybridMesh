"""The optimizer container a run trains with, and the ``Stateful`` view over it.

Moved here from ``components/checkpointer/base.py``: neither the materialization
nor ``init_optim_state`` is checkpoint machinery. The materialization is an
optimizer operation the checkpointer happens to need, and the FQN re-keying is
the on-disk *format* of optimizer state, which is a property of the optimizer,
not of the thing writing it out.

Vendored from torchtitan's ``components/optimizer/optimizer.py``. The unit that
came across is ``OptimizersContainer``, driven by the training loop and handed
to the checkpointer.

* It owns one ``torch.optim.Optimizer`` per (model part, optimizer name) pair, so
  a run with pipeline parallelism and two optimizer types holds four inner
  optimizers. ``step`` / ``zero_grad`` fan out over them; ``state_dict`` /
  ``load_state_dict`` flatten them into one FQN-keyed dict, which is what makes
  a PP checkpoint unambiguous (see ``utils.get_flat_optim_state_dict``).
* One of its behaviors is load-bearing beyond its name: ``state_dict``
  **materializes** optimizer state first (``init_optim_state``, a zero-gradient,
  zero-lr step) before reporting it. PyTorch's ``Optimizer`` already satisfies
  ``Stateful``, so that is what makes this object usable as one -- a resumed run
  builds a fresh optimizer whose Adam moments do not exist until its first
  ``step()``, and DCP, handed an unmaterialized state dict, would find no
  ``exp_avg`` to write into and silently restart from a cold optimizer under
  warm weights.

Departures from upstream, all subtractive:

* **No ``Configurable``.** torchtitan configs build themselves. hpmesh keeps
  every config in ``hpmesh.trainer.config``, so the container takes an
  ``OptimizerConfig`` and the trainer constructs it.
* **No ``DistMuon``.** torchtitan's factory table also offers ``DistMuon``,
  which is built on ``torchtitan.distributed.flex_shard``; hpmesh has no
  equivalent, so the table is ``Adam`` / ``AdamW``.
* **No bf16 optimizer states.** ``fused_opt_states_bf16`` and the
  materialize-in-bf16 pre-hook it needs are not ported; the implementation
  setting stops at ``fused`` / ``foreach`` / ``for-loop``.
* **No ``optimizer_factory_kwargs_by_name``.** That hook exists for per-parameter
  compute metadata and communication bucket specs; nothing in hpmesh passes it.
* **``_validate_params`` raises ``ValueError``, not ``AssertionError``, and
  names the offending parameters.** An unclaimed trainable parameter is
  reachable from user config -- list explicit ``param_groups`` and forget the
  catch-all -- and the contract is that user-facing errors are ``ValueError``.
  A parameter left out and a parameter assigned twice are different mismatches
  with different fixes, which a pair of counts cannot tell apart.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from ...utils.logger_utils import get_logger
from ..checkpointer.utils import canonical_fqn
from .utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)

if TYPE_CHECKING:
    # Type-only. ``trainer.config`` does not import this package at runtime (it
    # describes the object; the trainer builds it), so there is no cycle to
    # break -- this matches ``lr_scheduler.py``'s import of ``LRSchedulerConfig``.
    from ...trainer.config import OptimizerConfig, ParamGroupConfig

logger = get_logger(__name__)

__all__ = ["OptimizersContainer"]


class OptimizersContainer(Optimizer, Stateful):
    """One optimizer per (model part, optimizer name), driven as a single one.

    The training loop should not know how many optimizers exist, so this is
    itself an ``Optimizer``: ``step`` and ``zero_grad`` (the two methods the
    loop calls) fan out to the inner ones, and everything else -- gradient
    clipping, checkpointing -- reaches the parameters through ``param_groups``,
    which ``Optimizer.__init__`` merges across the inner optimizers.

    That merge is why ``_post_init`` exists rather than a plain assignment: the
    loop's ``clip_grad_norm_`` needs ``param_groups`` populated, and the step
    pre-hooks (MoE load balancing, the aux-loss roll-up) need the hook machinery
    that only ``Optimizer.__init__`` sets up. Calling it with an empty options
    dict gives both without registering any hyperparameters of its own.

    The number of inner optimizers follows from two independent splits:

    * one per **model part**, because each pipeline stage holds its own
      parameters. A run with pp=2 holds at least two, even when every parameter
      matches the same group -- which matters, because a step pre-hook fires once
      per ``step()`` call, i.e. once per container, not once per inner optimizer.
    * one per **optimizer name** within a part, because different parameter
      groups may name different optimizer classes.

    Parameters are matched to groups by regex against their FQN, first pattern
    wins; that is what lets norm and bias parameters take a different weight
    decay from the rest. A ``ValueError`` is raised if a pattern matches nothing
    (a typo'd pattern is otherwise a silent no-op) or if a trainable parameter
    ends up in no group at all.

    Args:
        config: the run's optimizer configuration. ``param_groups`` is the list
            of ``ParamGroupConfig`` patterns; ``implementation`` selects
            ``fused`` / ``foreach`` / ``for-loop`` for every inner optimizer.
        model_parts: the model chunks to optimize, one per pipeline stage.
    """

    optimizers: list[Optimizer]
    model_parts: list[nn.Module]

    def __init__(
        self, config: OptimizerConfig, *, model_parts: list[nn.Module]
    ) -> None:
        impl_kwargs = self._build_impl_kwargs(config)
        all_params: list[nn.Parameter] = []
        self.optimizers = []
        self.model_parts = model_parts

        for part_idx, model in enumerate(self.model_parts):
            groups_by_opt_name, patterns_by_opt_name = self._build_param_groups(
                model, config.param_groups, impl_kwargs
            )
            for opt_name, opt_param_groups in groups_by_opt_name.items():
                optimizer = self._resolve_optimizer_factory(opt_name)(opt_param_groups)
                self.optimizers.append(optimizer)
                self._log_optimizer(optimizer, part_idx, patterns_by_opt_name[opt_name])
                for group in opt_param_groups:
                    all_params.extend(group["params"])

        self._validate_params(all_params)
        self._post_init(all_params)

    @staticmethod
    def _resolve_optimizer_factory(name: str) -> Callable[..., Optimizer]:
        optimizer_factories: dict[str, Callable[..., Optimizer]] = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
        }
        if name not in optimizer_factories:
            raise NotImplementedError(f"Optimizer {name} not added.")
        return optimizer_factories[name]

    @staticmethod
    def _build_impl_kwargs(config: OptimizerConfig) -> dict[str, Any]:
        """The implementation kwargs (``fused`` / ``foreach``) applied to all groups.

        An ``optimizer_kwargs`` entry on a ``ParamGroupConfig`` overrides these --
        the update below is per group, so a group can opt out (``fused=False``).
        """
        implementation = config.implementation
        if implementation not in ("fused", "foreach", "for-loop"):
            raise ValueError(
                f"Unknown optimizer implementation {implementation!r}; expected "
                "one of 'fused', 'foreach', 'for-loop'."
            )
        return {
            "fused": implementation == "fused",
            "foreach": implementation == "foreach",
        }

    @staticmethod
    def _build_param_groups(
        model: nn.Module,
        param_group_configs: list[ParamGroupConfig],
        impl_kwargs: dict[str, Any],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
        """Partition a model's parameters into per-optimizer param groups.

        Each parameter is claimed by the first ``ParamGroupConfig`` whose pattern
        it matches, so order matters and a catch-all belongs last.

        Returns two dicts keyed by optimizer name and aligned by index: the group
        dicts to hand the optimizer constructor, and the pattern of each group.
        The patterns stay out of the group dict on purpose -- they are for the
        log line only, and a saved optimizer state dict would carry them forever.

        Each group dict also carries ``param_names`` (canonical FQNs aligned with
        ``params``). PyTorch records those on the group, and the checkpoint
        helpers read them back to key optimizer state by FQN.
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        patterns: dict[str, list[str]] = defaultdict(list)
        claimed: set[str] = set()

        for param_group_config in param_group_configs:
            pattern = re.compile(param_group_config.pattern)
            params: list[nn.Parameter] = []
            param_names: list[str] = []
            for name, param in model.named_parameters():
                if param.requires_grad and name not in claimed and pattern.search(name):
                    params.append(param)
                    param_names.append(canonical_fqn(name))
                    claimed.add(name)

            if not params:
                raise ValueError(
                    f"optimizer.param_groups pattern "
                    f"{param_group_config.pattern!r} matched no parameters"
                )

            groups[param_group_config.optimizer_name].append(
                {
                    "params": params,
                    "param_names": param_names,
                    **impl_kwargs,
                    **param_group_config.optimizer_kwargs,
                }
            )
            patterns[param_group_config.optimizer_name].append(
                param_group_config.pattern
            )

        return groups, patterns

    def _log_optimizer(
        self, optimizer: Optimizer, part_idx: int, patterns: list[str]
    ) -> None:
        """Log one inner optimizer's group assignments.

        The patterns are logged here and nowhere else -- they are deliberately
        not stored on the param groups, so this line is the only record of which
        pattern produced which group.
        """
        key_kwargs = {
            "lr",
            "weight_decay",
            "betas",
            "eps",
            "momentum",
            "nesterov",
            "fused",
            "foreach",
        }
        optimizer_name = type(optimizer).__name__
        for group, pattern in zip(optimizer.param_groups, patterns, strict=True):
            kwargs = {key: group[key] for key in key_kwargs if key in group}
            logger.info(
                "Optimizer %s (model_part=%d): %d params [%s] %s",
                optimizer_name,
                part_idx,
                len(group["params"]),
                pattern,
                kwargs,
            )

    def _validate_params(self, all_params: list[nn.Parameter]) -> None:
        """Every trainable parameter must land in exactly one group.

        Upstream asserts equality of the id sets; this names *which* invariant
        broke, because the two directions have different fixes and upstream's
        count-only message leaves them indistinguishable. User-reachable both
        ways, which is why this raises ``ValueError`` rather than asserting:

        * missing -- explicit ``param_groups`` without a catch-all, so a
          parameter would silently stay frozen at its initial value.
        * assigned twice -- two overlapping patterns (first-match-wins makes
          this unreachable through the public path today, but the equality
          upstream tests for would miss it, and silently double-stepping a
          parameter is the failure it was there to catch).
        """
        registered = {id(param) for param in all_params}
        missing = [
            (name, param)
            for model in self.model_parts
            for name, param in model.named_parameters()
            if param.requires_grad and id(param) not in registered
        ]
        if missing:
            names = ", ".join(
                f"{name} ({tuple(param.shape)})" for name, param in missing
            )
            raise ValueError(
                f"optimizer.param_groups left {len(missing)} trainable parameter(s) "
                f"unassigned, so they would keep their initial values: {names}. "
                "Add a catch-all ParamGroupConfig(pattern='.*') last."
            )

        duplicates = len(all_params) - len(registered)
        if duplicates:
            raise ValueError(
                f"optimizer.param_groups assigned {duplicates} parameter(s) to more "
                f"than one group; let the first matching pattern win. Overlapping "
                f"patterns are the usual cause."
            )

    def __iter__(self) -> Iterator[Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Advance every inner optimizer.

        ``closure`` is rejected rather than ignored: this container cannot
        support one, and silently dropping it would skip the user's loss
        recomputation. Returning ``None`` matches ``Optimizer.step``.
        """
        if closure is not None:
            raise ValueError("OptimizersContainer does not support closures")
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """A flat, FQN-keyed state dict covering every inner optimizer.

        Side effect: if an inner optimizer has not stepped yet,
        ``init_optim_state`` materializes its state first (a zero-gradient,
        zero-lr step) so DCP has tensors to read, and to write into on load. It
        leaves parameters alone, and is a no-op once state exists.
        """
        result: dict[str, Any] = {}
        for optimizer in self.optimizers:
            init_optim_state(optimizer)
            result.update(get_flat_optim_state_dict(optimizer))
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # init_optim_state must run first: the unflattening step reads each
        # optimizer's live state to learn which state tensors to expect, so a
        # fresh optimizer would find nothing to write into.
        for optimizer in self.optimizers:
            init_optim_state(optimizer)
            load_flat_optim_state_dict(optimizer, state_dict)

    def init_cache_state_dict(self) -> None:
        """Initialize cached state dict for TorchFT. No-op for base class.

        Present because upstream's subclasses override it and the training loop
        would call it unconditionally; keeping the no-op means such a caller
        does not have to know which container it holds.
        """
        pass

    def _post_init(self, all_params: list[nn.Parameter]) -> None:
        # ``Optimizer.__init__`` is what populates ``param_groups`` and sets up
        # the hook machinery that ``register_step_pre_hook`` needs. The empty
        # options dict is deliberate: the container's own ``param_groups`` is
        # only a view over the inner optimizers' parameters, and each inner
        # optimizer already holds the hyperparameters for its groups.
        Optimizer.__init__(self, all_params, {})
