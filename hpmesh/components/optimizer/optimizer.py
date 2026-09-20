# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The optimizers a run trains with, and the ``Stateful`` view over them.

Moved here from ``components/checkpointer/base.py``: neither ``OptimizerWrapper``
nor ``init_optim_state`` is checkpoint machinery. The materialization is an
optimizer operation the checkpointer happens to need, and the FQN re-keying is
the on-disk *format* of optimizer state, which is a property of the optimizer,
not of the thing writing it out.

Vendored from torchtitan's ``components/optimizer/optimizer.py``: the same
``OptimizersContainer``, split across the same two roles.

* **``OptimizersContainer``** is what the training loop drives. It owns one
  ``torch.optim.Optimizer`` per (model part, optimizer name) pair, so a run with
  pipeline parallelism and two optimizer types holds four inner optimizers.
  ``step`` / ``zero_grad`` fan out over them; ``state_dict`` /
  ``load_state_dict`` flatten them into one FQN-keyed dict, which is what makes
  a PP checkpoint unambiguous (see ``utils.get_flat_optim_state_dict``).
* **``OptimizerWrapper``** is the older, narrower spelling of the same idea: one
  inner optimizer, no parameter grouping. The trainer still builds a bare
  ``AdamW``, so this remains the object the checkpointer is handed until that
  changes. ``OptimizersContainer`` supersedes it -- a container with a single
  catch-all group carries the same state.

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
* **``_validate_params`` raises ``ValueError``, not ``AssertionError``.** An
  unclaimed trainable parameter is reachable from user config -- list explicit
  ``param_groups`` and forget the catch-all -- and the contract is that
  user-facing errors are ``ValueError``.
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
    # Type-only: ``trainer.config`` imports this package, so a runtime import
    # here would close the cycle config -> optimizer -> config.
    from ...trainer.config import ParamGroupConfig

logger = get_logger(__name__)

__all__ = ["OptimizersContainer", "OptimizerWrapper"]


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

    def __init__(self, config: Any, *, model_parts: list[nn.Module]) -> None:
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
    def _build_impl_kwargs(config: Any) -> dict[str, Any]:
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

        Upstream asserts this; it is user-reachable here -- supplying explicit
        ``param_groups`` without a catch-all silently leaves parameters frozen at
        their initial values -- so it raises ``ValueError`` with the count.
        """
        expected = {
            id(param)
            for model in self.model_parts
            for param in model.parameters()
            if param.requires_grad
        }
        actual = {id(param) for param in all_params}
        if expected != actual:
            raise ValueError(
                "optimizer.param_groups left trainable parameters unassigned: "
                f"{len(expected)} trainable params in the model, "
                f"{len(actual)} assigned. Add a catch-all "
                "ParamGroupConfig(pattern='.*') last."
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

    def _post_init(self, all_params: list[nn.Parameter]) -> None:
        # ``Optimizer.__init__`` is what populates ``param_groups`` and sets up
        # the hook machinery that ``register_step_pre_hook`` needs. The empty
        # options dict is deliberate: the container's own ``param_groups`` is
        # only a view over the inner optimizers' parameters, and each inner
        # optimizer already holds the hyperparameters for its groups.
        Optimizer.__init__(self, all_params, {})


class OptimizerWrapper(Stateful):
    """A ``Stateful`` view over one optimizer that survives a fresh load.

    ``torch.optim.Optimizer`` already satisfies ``Stateful``, and DCP writes
    straight into the tensors a ``Stateful`` reports -- which is why a *plain*
    optimizer works for saving and for loading into an optimizer that has
    already taken a step. It does not work for the case that matters: a resumed
    run builds a fresh optimizer, whose Adam moments do not exist until its
    first ``step()``, so DCP finds no ``exp_avg`` to write into and the run
    silently restarts from a cold optimizer under warm weights.

    The fix is to give DCP the tensors to load into before it plans the load.
    ``load_state_dict`` therefore materializes the state first (via
    ``init_optim_state``, a zero-gradient, zero-lr step) and only then hands the
    state dict to the optimizer, which -- with the state present -- restores in
    place.

    PyTorch expects this of a ``Stateful`` wrapper: it calls ``load_state_dict``
    on the *object it was given*, and that object is responsible for pushing the
    values into whatever it manages. hpmesh hands DCP this wrapper instead of
    the bare optimizer, so the contract is met.

    Args:
        optimizer: the optimizer to wrap.
        fqn_keying: when on, optimizer state is keyed by parameter FQN rather
            than by positional index, and the ``param_groups`` entry is reduced
            to the values that are identical across stages. Pipeline parallelism
            needs this: every stage's optimizer numbers its own parameters from
            0, so the positional keys of two stages collide in one shared
            checkpoint (torchtitan solves the same collision with its
            ``OptimizersContainer``'s FQN flattening). The FQNs are read off the
            wrapper's own ``fqns``, so the caller must have built the optimizer
            over exactly those parameters in that order -- the trainer does
            (``chain(*(part.parameters() ...))``). Off keeps the positional
            format, which every non-PP checkpoint already on disk uses.
        fqns: the parameter FQNs, in optimizer order. Only read when
            ``fqn_keying`` is on.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        fqn_keying: bool = False,
        fqns: list[str] | None = None,
    ) -> None:
        if fqn_keying and fqns is None:
            raise ValueError(
                "OptimizerWrapper(fqn_keying=True) needs the parameter FQNs in "
                "optimizer order; pass fqns=[name for part in parts "
                "for name, _ in part.named_parameters()]."
            )
        self.optimizer = optimizer
        self._fqns = fqns if fqn_keying else None

    def state_dict(self) -> dict[str, Any]:
        # Materialize first, on both directions. On a save, DCP reads whatever
        # tensors this reports -- an optimizer that has never stepped reports
        # none, so the checkpoint would quietly carry weights and no optimizer
        # state. On a load, DCP calls ``state_dict()`` to learn where the values
        # are going and only afterwards calls ``load_state_dict()``, so a
        # version that materialized lazily would be a step too late for the
        # planner to have anywhere to put ``exp_avg``.
        init_optim_state(self.optimizer)
        state_dict = self.optimizer.state_dict()
        if self._fqns is None:
            return state_dict
        return {
            "state": {
                # ``state_dict`` packs state positionally; the FQN order is the
                # same parameter order, so the re-keying is a rename only.
                self._fqns[index]: state
                for index, state in state_dict["state"].items()
            },
            # The positional ``params`` list must not be saved: under PP each
            # stage's optimizer numbers its own parameters from 0, so every
            # rank would write the same ``optimizer.param_groups`` key with a
            # list of its own length (stages differ in parameter count), and
            # one shared checkpoint key cannot hold them all. What remains --
            # lr, betas, and friends -- is config-level and identical across
            # stages, so a single shared copy is correct. ``load_state_dict``
            # rebuilds the list from the live optimizer.
            "param_groups": [
                {key: value for key, value in group.items() if key != "params"}
                for group in state_dict["param_groups"]
            ],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Already materialized by the ``state_dict()`` call that precedes this
        # one on the DCP path; idempotent here, and load-bearing for any caller
        # that restores without going through DCP.
        init_optim_state(self.optimizer)
        if self._fqns is not None:
            fqn_to_index = {fqn: i for i, fqn in enumerate(self._fqns)}
            state_dict = {
                "state": {
                    fqn_to_index[fqn]: state
                    for fqn, state in state_dict["state"].items()
                },
                "param_groups": [
                    # Re-inject the positional ``params`` list ``state_dict``
                    # dropped: index i is the i-th parameter of the live group.
                    {**group, "params": list(range(len(live["params"])))}
                    for group, live in zip(
                        state_dict["param_groups"],
                        self.optimizer.param_groups,
                        strict=True,
                    )
                ],
            }
        self.optimizer.load_state_dict(state_dict)
