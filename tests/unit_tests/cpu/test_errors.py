"""The hpmesh.errors hierarchy: three fail-fast semantics, told apart by type.

Each leaf also IS the builtin its guard points raised before the hierarchy
existed, which is what keeps legacy ``pytest.raises(ValueError /
NotImplementedError)`` assertions catching the same failures.
"""

import pytest

from hpmesh.errors import (
    ConfigError,
    EnvironmentUnsupportedError,
    HpmeshError,
    UnsupportedCombinationError,
)


def test_all_leaves_are_hpmesh_errors() -> None:
    for exc in (ConfigError, UnsupportedCombinationError, EnvironmentUnsupportedError):
        assert issubclass(exc, HpmeshError)
        assert issubclass(exc, Exception)


def test_config_error_stays_a_value_error() -> None:
    with pytest.raises(ValueError):
        raise ConfigError("bad field")
    with pytest.raises(HpmeshError):
        raise ConfigError("bad field")


def test_combination_error_stays_a_not_implemented_error() -> None:
    with pytest.raises(NotImplementedError):
        raise UnsupportedCombinationError("tp x ep x cp")
    with pytest.raises(HpmeshError):
        raise UnsupportedCombinationError("tp x ep x cp")


def test_environment_error_stays_a_not_implemented_error() -> None:
    with pytest.raises(NotImplementedError):
        raise EnvironmentUnsupportedError("needs torch >= 2.12")
    with pytest.raises(HpmeshError):
        raise EnvironmentUnsupportedError("needs torch >= 2.12")


def test_the_two_not_implemented_leaves_are_distinct() -> None:
    # Catching one must not catch the other: "fix the combination" and
    # "unlock the dependency" are different operator actions.
    with pytest.raises(UnsupportedCombinationError):
        try:
            raise UnsupportedCombinationError("combo")
        except EnvironmentUnsupportedError:
            pytest.fail("combination caught as environment")
    with pytest.raises(EnvironmentUnsupportedError):
        try:
            raise EnvironmentUnsupportedError("env")
        except UnsupportedCombinationError:
            pytest.fail("environment caught as combination")


def test_config_error_is_not_a_not_implemented_error() -> None:
    assert not issubclass(ConfigError, NotImplementedError)
    assert not issubclass(UnsupportedCombinationError, ValueError)
