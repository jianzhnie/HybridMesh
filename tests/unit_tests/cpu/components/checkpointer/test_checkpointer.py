"""The checkpointer's own surface: FQN handling, config validation, the wrapper.

The manager's save/load behaviour is covered in ``test_trainer.py``, where it is
exercised against real DCP round-trips. This file covers what sits underneath:
the pieces that are wrong in ways a round-trip would not reveal -- a config
that accepts a combination it cannot honour, an FQN helper that strips too much
or too little, an optimizer wrapper that reports success without restoring.
"""

from __future__ import annotations

from tests.caps import require_env

require_env('dcp', 'dtensor')


import torch
import torch.nn as nn

from hpmesh.components.checkpointer import (
    CheckpointManager,
    CheckpointStorage,
    ModelWrapper,
    canonical_fqn,
)
from hpmesh.components.checkpointer.dcp import FilesystemCheckpointStorage
from hpmesh.components.optimizer import init_optim_state
from hpmesh.components.optimizer.lr_scheduler import build_lr_scheduler
from hpmesh.config import CheckpointConfig as Config
from hpmesh.config import LRSchedulerConfig

# -- canonical_fqn ------------------------------------------------------------


def test_canonical_fqn_strips_the_wrapper_at_any_depth() -> None:
    assert (
        canonical_fqn(
            "model.layers.0._checkpoint_wrapped_module.self_attn.q_proj.weight"
        )
        == "model.layers.0.self_attn.q_proj.weight"
    )
    # The wrapper can also wrap the whole module, putting the segment first.
    assert (
        canonical_fqn("_checkpoint_wrapped_module.model.layers.0.weight")
        == "model.layers.0.weight"
    )


def test_canonical_fqn_leaves_a_real_name_alone() -> None:
    name = "model.layers.0.self_attn.q_proj.weight"
    assert canonical_fqn(name) == name


# -- config validation --------------------------------------------------------


def test_config_defaults_do_not_validate_at_import() -> None:
    """Constructing the defaults is how every run starts; it must not raise."""
    config = Config()
    assert config.enable is False
    assert config.keep_latest_k == 10


def test_config_rejects_keep_latest_k_of_one() -> None:
    """One retained slot is not a policy -- it is the slot a live save occupies."""
    try:
        Config(keep_latest_k=1)
    except ValueError as error:
        assert "at least 2 checkpoint replicas" in str(error)
        return
    raise AssertionError("keep_latest_k=1 should be rejected")


def test_config_rejects_a_zero_interval() -> None:
    try:
        Config(interval=0)
    except ValueError as error:
        assert "at least 1 step" in str(error)
        return
    raise AssertionError("interval=0 should be rejected")


def test_config_requires_optimizer_exclusion_to_imply_lr_scheduler() -> None:
    try:
        Config(exclude_from_loading=["optimizer"])
    except ValueError as error:
        assert "lr_scheduler" in str(error)
        return
    raise AssertionError("excluding the optimizer alone should be rejected")


def test_config_rejects_a_relative_initial_load_path() -> None:
    try:
        Config(initial_load_path="weights/step-1")
    except ValueError as error:
        assert "absolute path" in str(error)
        return
    raise AssertionError("a relative initial_load_path should be rejected")


def test_config_rejects_hf_quantized_without_hf() -> None:
    try:
        Config(initial_load_in_hf_quantized=True)
    except ValueError as error:
        assert "initial_load_in_hf" in str(error)
        return
    raise AssertionError("quantized-without-hf should be rejected")


def test_config_rejects_hf_safetensors_against_a_remote_folder() -> None:
    """Remote IO supports only the native DCP format."""
    try:
        Config(last_save_in_hf=True, folder="gs://bucket/checkpoints")
    except ValueError as error:
        assert "remote" in str(error)
        return
    raise AssertionError("last_save_in_hf over a remote URI should be rejected")


def _schedule(optimizer):
    """The lr schedule the manager now requires alongside the optimizer.

    A bare scheduler over the bare ``AdamW`` these cases build: the manager
    checkpoints ``last_epoch`` and nothing else, so the lambda is irrelevant.
    """
    return build_lr_scheduler(
        LRSchedulerConfig(), optimizers=[optimizer], training_steps=8
    )


# -- the manager's unwired-option guard ---------------------------------------


def test_hf_options_are_rejected_without_a_state_dict_adapter(tmp_path) -> None:
    """hpmesh ships no adapter, so the HF paths must refuse, not silently no-op."""
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    try:
        CheckpointManager(
            Config(enable=True, last_save_in_hf=True),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=_schedule(optimizer),
            states={},
            folder=str(tmp_path),
        )
    except ValueError as error:
        assert "last_save_in_hf" in str(error)
        return
    raise AssertionError("last_save_in_hf without an sd_adapter should raise")


def test_a_disabled_manager_returns_early_without_building_anything(tmp_path) -> None:
    """``enable=False`` is the default; it must not allocate a purge thread."""
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    manager = CheckpointManager(
        Config(enable=False),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer),
        states={},
        folder=str(tmp_path),
    )
    assert not hasattr(manager, "purge_thread")
    assert manager.save(1) is False
    assert manager.load(-1) is False
    manager.close()  # must be a no-op, not an AttributeError


def test_close_and_the_public_methods_survive_a_failed_constructor(tmp_path) -> None:
    """A manager whose ``__init__`` raised is still called by ``__del__``.

    The HF-options guard raises after ``enable`` and ``_storage`` are assigned
    but before the async futures and retention policy are, so this is the one
    reachable path where a half-built manager is left for the garbage collector.
    """
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    manager = None
    try:
        CheckpointManager(
            Config(enable=True, async_mode="async", last_save_in_hf=True),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=_schedule(optimizer),
            states={},
            folder=str(tmp_path),
        )
    except ValueError as error:
        assert "last_save_in_hf" in str(error)
    assert manager is None

    # Rebuild the same half-constructed state and drive the entry points that
    # ``__del__`` and a partially-failed setup would reach.
    half_built = CheckpointManager.__new__(CheckpointManager)
    half_built.enable = True  # assigned first, so it reads as enabled
    assert half_built.save(1) is False
    assert half_built.load(-1) is False
    half_built.maybe_wait_for_staging()
    half_built.close()


# -- storage protocol ---------------------------------------------------------


def test_filesystem_storage_satisfies_the_protocol(tmp_path) -> None:
    """``runtime_checkable`` catches a rename that would surface mid-save."""
    assert isinstance(FilesystemCheckpointStorage(), CheckpointStorage)


def test_filesystem_storage_reports_paths(tmp_path) -> None:
    storage = FilesystemCheckpointStorage()
    (tmp_path / "step-1").mkdir()
    (tmp_path / "step-1" / ".metadata").write_text("{}")

    assert storage.isdir(str(tmp_path))
    assert storage.isdir(str(tmp_path / "step-1"))
    assert storage.isfile(str(tmp_path / "step-1" / ".metadata"))
    assert not storage.isfile(str(tmp_path / "step-1"))
    assert storage.listdir(str(tmp_path)) == ["step-1"]

    storage.remove(str(tmp_path / "step-1"))
    assert not storage.isdir(str(tmp_path / "step-1"))
    # Deleting again is a no-op rather than a FileNotFoundError in the purge
    # thread, which would otherwise kill retention for the rest of the run.
    storage.remove(str(tmp_path / "step-1"))


# -- init_optim_state ---------------------------------------------------------


def test_init_optim_state_materializes_without_training() -> None:
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in model.parameters()]

    assert optimizer.state_dict()["state"] == {}
    init_optim_state(optimizer)

    assert len(optimizer.state_dict()["state"]) == len(before)
    for original, current in zip(before, model.parameters(), strict=True):
        assert torch.equal(original, current.detach())
    assert [p.grad for p in model.parameters()] == [None, None]


def test_init_optim_state_leaves_the_first_real_update_as_step_one() -> None:
    """Adam's counter must not be advanced by the materializing step."""
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    init_optim_state(optimizer)

    for state in optimizer.state_dict()["state"].values():
        assert int(state["step"]) == 0
        assert torch.count_nonzero(state["exp_avg"]) == 0


def test_init_optim_state_is_a_no_op_once_state_exists() -> None:
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    saved = optimizer.state_dict()["state"]

    init_optim_state(optimizer)

    for param_id, state in optimizer.state_dict()["state"].items():
        assert torch.equal(state["exp_avg"], saved[param_id]["exp_avg"])
        assert int(state["step"]) == int(saved[param_id]["step"])


def test_init_optim_state_preserves_existing_gradients() -> None:
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.ones(2, 4)).sum().backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]

    init_optim_state(optimizer)

    for original, param in zip(grads, model.parameters(), strict=True):
        assert torch.equal(original, param.grad)


def test_init_optim_state_completes_a_partially_initialized_optimizer() -> None:
    """A parameter that has not received a gradient still needs checkpoint state.

    Adam initializes state lazily per parameter. A conditional branch can
    therefore leave one parameter cold while another has already stepped; an
    early return based on ``optim.state`` being merely non-empty produces an
    incomplete checkpoint that a fresh optimizer cannot restore.
    """
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    # Initialize only the weight. The bias deliberately receives no gradient.
    model.weight.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert model.weight in optimizer.state
    assert model.bias not in optimizer.state

    weight_state = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[model.weight].items()
    }
    init_optim_state(optimizer)

    assert model.bias in optimizer.state
    for key, expected in weight_state.items():
        actual = optimizer.state[model.weight][key]
        if torch.is_tensor(expected):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected
    # The newly materialized state must still make the next real update step 1.
    assert int(optimizer.state[model.bias]["step"]) == 0


def test_model_wrapper_keeps_tensor_storage_stable_across_calls() -> None:
    """Stable storage is what lets async DCP reuse its pinned host buffers."""
    model = nn.Linear(4, 4)
    wrapper = ModelWrapper(model)

    first = wrapper.state_dict()
    storages = {k: v.untyped_storage().data_ptr() for k, v in first.items()}
    with torch.no_grad():
        model.weight.add_(1)

    second = wrapper.state_dict()

    assert {k: v.untyped_storage().data_ptr() for k, v in second.items()} == storages
    assert torch.equal(second["weight"], model.weight)


# -- TorchCheckpointingManager ------------------------------------------------
#
# The backend (``torch_checkpointing``) is not a dependency and is not installed
# in the development environment, so nothing here can round-trip a checkpoint.
# What *is* testable is the import guard and the state registration, both of
# which are the parts a missing backend would otherwise let rot silently.


def test_a_disabled_torch_checkpointing_manager_never_touches_the_backend() -> None:
    """``enable=False`` must return before the import.

    This is the property that keeps a disabled manager constructible on a
    machine without the backend; if the import moved above the guard, every
    caller would need the package installed just to not use it.
    """
    from hpmesh.components.checkpointer.torch_checkpointing import (
        TorchCheckpointingManager,
    )

    manager = TorchCheckpointingManager(
        Config(enable=False),
        model_parts=[],
        optimizer=None,
        lr_scheduler=None,
        states={},
        folder="/tmp/unused",
    )
    assert manager.enable is False


def test_an_enabled_torch_checkpointing_manager_without_the_backend_raises() -> None:
    """The failure must be an actionable ImportError, not a bare ModuleNotFoundError.

    ``torch_checkpointing`` is optional, so a missing install is an expected
    state rather than a bug: the message has to name the alternative (the DCP
    manager) instead of surfacing whichever submodule happened to import first.
    """
    import importlib.util

    import pytest

    from hpmesh.components.checkpointer.torch_checkpointing import (
        TorchCheckpointingManager,
    )

    if importlib.util.find_spec("torch_checkpointing") is not None:
        pytest.skip("backend is installed; the guard cannot fire")

    with pytest.raises(ImportError, match="not installed"):
        TorchCheckpointingManager(
            Config(enable=True),
            model_parts=[],
            optimizer=None,
            lr_scheduler=None,
            states={},
            folder="/tmp/unused",
        )


def test_an_enabled_manager_registers_the_lr_scheduler(monkeypatch) -> None:
    """The scheduler must ride along under ``LR_SCHEDULER``, like the DCP manager.

    Without it a resumed run's fresh scheduler restarts ``last_epoch`` at 0, so
    the curve restarts on the step after a resume -- invisible while the lr is
    constant, wrong for the rest of the run once warmup or decay is set.

    The backend is absent, so the import is stubbed to reach the registration.
    Everything after it needs real backend objects and is cut short, but the
    states dict is populated first, which is what this pins.
    """
    import pytest

    from hpmesh.components.checkpointer import LR_SCHEDULER
    from hpmesh.components.checkpointer import torch_checkpointing as tc

    class _StubBackend:
        def __getattr__(self, name):
            return None

    class _ReachedBackendUse(Exception):
        pass

    def _stop(_backend):
        raise _ReachedBackendUse

    monkeypatch.setattr(tc, "require_torch_checkpointing", _StubBackend)
    monkeypatch.setattr(tc, "async_save_config", _stop)

    sentinel = object()
    states: dict[str, object] = {}
    with pytest.raises(_ReachedBackendUse):
        tc.TorchCheckpointingManager(
            Config(enable=True),
            model_parts=[],
            optimizer=None,
            lr_scheduler=sentinel,
            states=states,
            folder="/tmp/unused",
        )
    assert states[LR_SCHEDULER] is sentinel
