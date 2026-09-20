"""The ambient SPMD context: what it makes answerable, and what it must not.

``spmd_context`` is the single place hpmesh enters the ambient mesh state. Two
of its properties are worth pinning on CPU, without a process group:

* a single-process run (``parallel_dims is None``) must be a clean no-op, since
  that is the path ``python -m hpmesh`` takes -- every lookup has to answer
  "off" rather than raise;
* the state must be *restored* on exit, so a nested region cannot leak a mesh to
  the code after it.

The multi-rank behavior (that a live mesh actually resolves a group, and that it
is the *same* process group the parallel layer would shard on) needs two ranks
and lives in ``tests/cp_equivalence.py``.
"""

from __future__ import annotations

import pytest

from hpmesh.utils.spmd_context import (
    current_spmd_mesh,
    spmd_context,
    spmd_mesh_group,
    spmd_mesh_size,
)


@pytest.mark.parametrize("axis", ["dp", "cp", "tp", "ep"])
def test_no_context_means_every_axis_is_off(axis: str) -> None:
    """Without a context there is no group and the size reads as 1.

    This is the degradation every model component relies on: ``None`` is "axis
    off", so a component skips its collective instead of running a size-1 one
    that would be a silent no-op with the wrong answer attached.
    """
    assert spmd_mesh_group(axis) is None
    assert spmd_mesh_size(axis) == 1
    assert current_spmd_mesh() is None


def test_single_process_run_enters_a_harmless_context() -> None:
    """``parallel_dims is None`` must not raise -- it is the step-0 path."""
    with spmd_context(None):
        assert spmd_mesh_group("tp") is None
        assert spmd_mesh_size("tp") == 1

    # ...and it leaves no state behind.
    assert current_spmd_mesh() is None


def test_context_restores_the_previous_state_on_exit() -> None:
    """A nested region must not leak its mesh to whatever runs after it."""
    assert current_spmd_mesh() is None

    with spmd_context(None):
        assert current_spmd_mesh() is None
        with spmd_context(None):
            assert current_spmd_mesh() is None
        assert current_spmd_mesh() is None

    assert current_spmd_mesh() is None
