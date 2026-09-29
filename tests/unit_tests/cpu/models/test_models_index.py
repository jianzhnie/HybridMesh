"""The ``models.common`` index: lazy, complete, and honest about a miss.

The index exists so a caller can say ``from llmtuner.models.common import MoE``
without knowing the node layout. What it must *not* do is make that convenience
cost an import: an eager index drags the MoE stack (and the fused QKV
projection's ``DTensor``) into every consumer, including the ones that only
wanted a mask helper -- which on a torch build without those is the difference
between working and not importing at all. The lightness check therefore runs in
a subprocess (module state is process-global) while the rest is plain attribute
inspection.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import llmtuner
from llmtuner.models import common


def test_every_exported_name_has_a_leaf() -> None:
    """``__all__`` is written out by hand (so linters see the re-exports), which
    means it can drift from the lazy table. This is the pin."""
    assert set(common.__all__) == set(common._EXPORTS)
    # Case-insensitive, matching how the linter orders the re-exports it reads.
    assert common.__all__ == sorted(common.__all__, key=str.casefold)


def test_a_name_resolves_through_the_index_and_is_the_leaf_object() -> None:
    """The index hands back the leaf's object, not a copy."""
    from llmtuner.models.common.activation import SwiGLU

    assert common.SwiGLU is SwiGLU


def test_an_unknown_name_raises_an_attribute_error() -> None:
    """A miss must look like a miss, not like an empty value."""
    with pytest.raises(AttributeError, match="has no attribute 'Nope'"):
        _ = common.Nope


def test_importing_a_leaf_does_not_import_its_siblings() -> None:
    """The documented promise of the index: naming a node must not import it.

    Checked in a subprocess because ``sys.modules`` is process-global and pytest
    shares one process across every test.
    """
    code = (
        "import sys;"
        "import llmtuner.models.common.rope;"
        "leaked = [m for m in sys.modules if m.startswith("
        "'llmtuner.models.common.moe')];"
        "assert not leaked, leaked;"
        "print('ok')"
    )
    # The package root, so the subprocess resolves ``llmtuner`` the same way
    # this one did (editable install or repo checkout, either is fine).
    repo_root = Path(llmtuner.__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
