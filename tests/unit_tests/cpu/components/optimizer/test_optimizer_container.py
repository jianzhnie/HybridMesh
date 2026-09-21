"""``OptimizersContainer``: grouping, validation, and the FQN state round-trip.

The trainer exercises the container end-to-end in ``test_trainer.py``, but only
through a checkpoint round-trip. The pieces worth pinning here are the ones a
round-trip cannot see: which parameters land in which group, that a pattern
matching nothing fails loudly instead of silently optimizing nothing, and that
a ``step()`` fires a registered hook exactly once -- the contract the aux-loss
roll-up depends on.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hpmesh.components.optimizer import OptimizersContainer
from hpmesh.trainer.config import OptimizerConfig, ParamGroupConfig


def _model() -> nn.Module:
    """A model with both a norm (no decay) and projections (decay)."""
    return nn.Sequential(
        nn.Linear(8, 8),
        nn.LayerNorm(8),
        nn.Linear(8, 4),
    )


def _cfg(
    *groups: ParamGroupConfig, implementation: str = "for-loop"
) -> OptimizerConfig:
    return OptimizerConfig(implementation=implementation, param_groups=list(groups))


def _catch_all(lr: float = 0.1, **kwargs) -> ParamGroupConfig:
    return ParamGroupConfig(
        pattern=".*", optimizer_name="AdamW", optimizer_kwargs={"lr": lr, **kwargs}
    )


def test_the_default_config_becomes_a_single_catch_all_group() -> None:
    """An empty ``param_groups`` must not mean "optimize nothing"."""
    cfg = OptimizerConfig(learning_rate=0.25)
    assert len(cfg.param_groups) == 1
    assert cfg.param_groups[0].optimizer_kwargs["lr"] == 0.25

    model = _model()
    container = OptimizersContainer(cfg, model_parts=[model])
    assert len(container) == 1
    assert len(container.optimizers[0].param_groups[0]["params"]) == len(
        list(model.parameters())
    )


def test_patterns_partition_by_first_match() -> None:
    """The norm takes its own weight decay; everything else takes the fallback."""
    model = _model()
    container = OptimizersContainer(
        _cfg(
            ParamGroupConfig(
                pattern=r"^1\.weight$",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0},
            ),
            _catch_all(weight_decay=0.01),
        ),
        model_parts=[model],
    )

    norm, rest = container.optimizers[0].param_groups
    assert norm["weight_decay"] == 0.0
    assert rest["weight_decay"] == 0.01
    # First match wins, so the norm is not also in the catch-all.
    assert len(norm["params"]) == 1
    assert len(rest["params"]) == len(list(model.parameters())) - 1


def test_a_pattern_matching_nothing_is_rejected() -> None:
    """A typo'd regex is otherwise a silent no-op that optimizes nothing."""
    unmatched = ParamGroupConfig(
        pattern=r"no_such_parameter",
        optimizer_name="AdamW",
        optimizer_kwargs={"lr": 0.1},
    )
    # A catch-all is present too, so the only thing wrong is the dead pattern.
    with pytest.raises(ValueError, match="matched no parameters"):
        OptimizersContainer(_cfg(unmatched, _catch_all()), model_parts=[_model()])


def test_a_trainable_parameter_left_out_of_every_group_is_rejected() -> None:
    """Explicit groups without a catch-all would freeze the leftovers."""
    model = _model()
    only_norm = ParamGroupConfig(
        pattern=r"1\.weight$",
        optimizer_name="AdamW",
        optimizer_kwargs={"lr": 0.1},
    )
    with pytest.raises(ValueError, match="unassigned"):
        OptimizersContainer(_cfg(only_norm), model_parts=[model])


def test_the_unassigned_error_names_the_parameters() -> None:
    """The counts alone cannot say which way the id sets differ, so name them.

    A parameter left out and a parameter double-assigned are both mismatches,
    but only the first is reachable through the public path, and the fix (add a
    catch-all, or a pattern for the named parameters) needs the names.
    """
    model = _model()
    only_norm = ParamGroupConfig(
        pattern=r"1\.weight$",
        optimizer_name="AdamW",
        optimizer_kwargs={"lr": 0.1},
    )
    with pytest.raises(ValueError, match=r"0\.weight \(\(8, 8\)\)"):
        OptimizersContainer(_cfg(only_norm), model_parts=[model])


def test_frozen_parameters_are_not_required_in_a_group() -> None:
    """``requires_grad=False`` parameters are skipped, not counted as missing."""
    model = _model()
    for param in model[0].parameters():
        param.requires_grad = False
    # Only the remaining parameters need a group, and the catch-all takes them.
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=[model])
    assert (
        len(container.optimizers[0].param_groups[0]["params"])
        == len(list(model.parameters())) - 2
    )


def test_each_model_part_gets_its_own_inner_optimizer() -> None:
    """PP puts each stage's parameters in its own optimizer, hook count aside."""
    parts = [_model(), _model()]
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=parts)
    assert len(container) == 2
    assert list(container) == container.optimizers


@pytest.mark.parametrize("implementation", ["for-loop", "foreach", "fused"])
def test_the_implementation_flag_reaches_the_inner_optimizer(
    implementation: str,
) -> None:
    container = OptimizersContainer(
        _cfg(_catch_all(), implementation=implementation), model_parts=[_model()]
    )
    group = container.optimizers[0].param_groups[0]
    assert group["fused"] is (implementation == "fused")
    assert group["foreach"] is (implementation == "foreach")


def test_an_unknown_optimizer_name_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="not added"):
        OptimizersContainer(
            _cfg(
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="SGD",
                    optimizer_kwargs={"lr": 0.1},
                )
            ),
            model_parts=[_model()],
        )


def test_an_unknown_implementation_is_rejected() -> None:
    """``implementation`` is a closed enum; a typo must not silently pick a kernel."""
    with pytest.raises(ValueError, match="Unknown optimizer implementation"):
        OptimizersContainer(
            _cfg(_catch_all(), implementation="quantum"), model_parts=[_model()]
        )


def test_a_closure_is_rejected_rather_than_dropped() -> None:
    """``step(closure)`` cannot be honoured, and ignoring it skips the caller's work.

    Returning ``None`` like ``Optimizer.step`` does would look like success while
    the closure -- typically a loss recomputation -- never ran.
    """
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=[_model()])
    with pytest.raises(ValueError, match="does not support closures"):
        container.step(lambda: 1.0)


def test_zero_grad_and_step_reach_every_inner_optimizer() -> None:
    model = _model()
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=[model])
    loss = model(torch.ones(2, 8)).pow(2).mean()
    container.zero_grad(set_to_none=True)
    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]
    container.step()
    assert any(
        not torch.equal(a, p) for a, p in zip(before, model.parameters(), strict=False)
    )


def test_a_step_hook_fires_once_per_container_step_not_per_optimizer() -> None:
    """The aux-loss roll-up is registered on the container, so PP must not multiply it.

    Two model parts means two inner optimizers. If the hook were registered on
    each of them it would fire twice per training step and double-count the
    rolled-up aux losses.
    """
    container = OptimizersContainer(
        _cfg(_catch_all()), model_parts=[_model(), _model()]
    )
    assert len(container) == 2

    calls: list[int] = []
    container.register_step_pre_hook(lambda *a, **k: calls.append(1))

    model = _model()
    loss = model(torch.ones(2, 8)).pow(2).mean()
    container.zero_grad(set_to_none=True)
    loss.backward()
    container.step()

    assert len(calls) == 1


def test_state_dict_is_flat_and_fqn_keyed() -> None:
    """The format DCP needs: one key per (parameter, state tensor), no indices."""
    model = _model()
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=[model])
    container.zero_grad(set_to_none=True)
    model(torch.ones(2, 8)).pow(2).mean().backward()
    container.step()

    state = container.state_dict()
    assert "state.0.weight.exp_avg" in state
    assert "state.0.weight.exp_avg_sq" in state
    # No integer-indexed nesting survives the flattening.
    assert all(isinstance(key, str) for key in state)
    assert not any(key.startswith("state.0.") and key.count(".") == 2 for key in state)


def test_state_round_trips_into_a_fresh_container() -> None:
    """A resumed run's cold optimizer must come back warm, bit for bit."""
    model = _model()
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=[model])
    for _ in range(2):
        container.zero_grad(set_to_none=True)
        model(torch.ones(2, 8)).pow(2).mean().backward()
        container.step()
    saved = {
        key: value.clone()
        for key, value in container.state_dict().items()
        if key.endswith(("exp_avg", "exp_avg_sq"))
    }
    assert saved, "sanity: the run must have produced optimizer state"

    fresh_model = _model()
    fresh = OptimizersContainer(_cfg(_catch_all()), model_parts=[fresh_model])
    # A cold container reports no state until something materializes it, which
    # is exactly what load_state_dict does first.
    fresh.load_state_dict(container.state_dict())

    restored = {
        key: value
        for key, value in fresh.state_dict().items()
        if key.endswith(("exp_avg", "exp_avg_sq"))
    }
    assert restored.keys() == saved.keys()
    for key, value in saved.items():
        assert torch.equal(restored[key], value)
