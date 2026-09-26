"""The capability registry: probing, caching, and the require() contract."""

import pytest

from hpmesh.accelerator import capabilities
from hpmesh.accelerator.capabilities import CAPABILITIES, has, require
from hpmesh.errors import EnvironmentUnsupportedError

ALL_NAMES = [
    "dynamo_capture_scalar_outputs",
    "inductor_micro_pipeline_tp",
    "fx_regional_inductor",
    "symm_mem",
    "functorch_activation_memory_budget",
    "torch_grouped_mm",
]


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    capabilities.probe.cache_clear()
    yield
    capabilities.probe.cache_clear()


def test_registry_covers_the_expected_names() -> None:
    assert sorted(CAPABILITIES) == sorted(ALL_NAMES)


def test_every_entry_probes_a_bool() -> None:
    for name in ALL_NAMES:
        assert isinstance(has(name), bool), name


def test_unknown_name_raises_immediately() -> None:
    with pytest.raises(KeyError, match="unknown capability"):
        has("dynamo_captrue_scalar_outputs")  # the typo must not read as False
    with pytest.raises(KeyError, match="unknown capability"):
        require("nope", feature="anything")


def test_probe_result_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def counting_probe() -> bool:
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setitem(CAPABILITIES, "counting", CAPABILITIES["symm_mem"])
    monkeypatch.setattr(CAPABILITIES["counting"], "probe", counting_probe)
    assert has("counting") is True
    assert has("counting") is True
    assert calls == 1


def test_require_passes_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CAPABILITIES["symm_mem"], "probe", lambda: True)
    require("symm_mem", feature="async TP")  # no raise


def test_require_raises_with_the_unlock_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CAPABILITIES["symm_mem"], "probe", lambda: False)
    with pytest.raises(EnvironmentUnsupportedError) as excinfo:
        require("symm_mem", feature="compile_config.enable_async_tensor_parallel")
    message = str(excinfo.value)
    # The feature, the missing thing, the torch version, and the unlock hint.
    assert "enable_async_tensor_parallel" in message
    assert "enable_symm_mem_for_group" in message
    assert "Upgrade torch" in message
    assert "symm_mem" in message


def test_require_is_a_not_implemented_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CAPABILITIES["symm_mem"], "probe", lambda: False)
    with pytest.raises(NotImplementedError):
        require("symm_mem", feature="async TP")


def test_has_reflects_a_flipped_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        CAPABILITIES["functorch_activation_memory_budget"], "probe", lambda: False
    )
    assert has("functorch_activation_memory_budget") is False
    monkeypatch.setattr(
        CAPABILITIES["functorch_activation_memory_budget"], "probe", lambda: True
    )
    capabilities.probe.cache_clear()
    assert has("functorch_activation_memory_budget") is True
