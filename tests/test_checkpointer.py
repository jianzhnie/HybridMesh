# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The checkpointer's own surface: FQN handling, config validation, the wrapper.

The manager's save/load behaviour is covered in ``test_trainer.py``, where it is
exercised against real DCP round-trips. This file covers what sits underneath:
the pieces that are wrong in ways a round-trip would not reveal -- a config
that accepts a combination it cannot honour, an FQN helper that strips too much
or too little, an optimizer wrapper that reports success without restoring.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hpmesh.components.checkpointer import (
    CheckpointManager,
    CheckpointStorage,
    ModelWrapper,
    OptimizerWrapper,
    canonical_fqn,
    init_optim_state,
)
from hpmesh.components.checkpointer.dcp import _FilesystemCheckpointStorage

Config = CheckpointManager.Config


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
    assert isinstance(_FilesystemCheckpointStorage(), CheckpointStorage)


def test_filesystem_storage_reports_paths(tmp_path) -> None:
    storage = _FilesystemCheckpointStorage()
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


# -- OptimizerWrapper ---------------------------------------------------------


def test_optimizer_wrapper_restores_into_a_cold_optimizer() -> None:
    """The regression this wrapper exists for.

    DCP writes into the tensors a state dict reports; a fresh Adam reports none,
    so without the materializing ``state_dict`` the load reports success and
    restores nothing.
    """
    torch.manual_seed(0)
    source_model = nn.Linear(4, 4)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.1)
    for _ in range(2):
        source_optimizer.zero_grad()
        source_model(torch.ones(2, 4)).sum().backward()
        source_optimizer.step()
    source_state = source_optimizer.state_dict()

    target_model = nn.Linear(4, 4)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.1)
    assert target_optimizer.state_dict()["state"] == {}

    wrapper = OptimizerWrapper(target_optimizer)
    wrapped_state = wrapper.state_dict()
    # state_dict() is the hook DCP calls to plan the load, so materializing here
    # is what gives the planner somewhere to put exp_avg.
    assert wrapped_state["state"] != {}

    wrapper.load_state_dict(source_state)

    restored = target_optimizer.state_dict()["state"]
    assert restored.keys() == source_state["state"].keys()
    for param_id, state in source_state["state"].items():
        for key, value in state.items():
            assert torch.equal(restored[param_id][key], value)


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
