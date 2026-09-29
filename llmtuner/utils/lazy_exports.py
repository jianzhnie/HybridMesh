"""One implementation of the lazy package index (PEP 562) used across llmtuner.

Five ``__init__`` files here are *indexes*: each names a slice of its package's
public surface so that ``from llmtuner.accelerator import all_reduce`` works,
while the submodule behind a name is imported only when that name is touched.
The laziness is load-bearing, not a style choice:

* ``llmtuner.accelerator`` has to stay importable where ``dist.py`` is not;
* ``llmtuner.parallel`` is read by ``llmtuner.config``, which cannot pay for the
  engine layer (its context-parallel half pulls in the model stack);
* ``llmtuner.components.checkpointer`` guards a torch.distributed surface that
  older torch builds do not have;
* ``llmtuner.models.common`` must not drag the MoE stack into every consumer;
* ``llmtuner`` itself keeps ``Trainer`` -- and the whole ``torch.distributed``
  stack behind it -- out of ``import llmtuner``.

Each index keeps its own name -> submodule table; that table is the discoverable
thing. What is shared is the lookup, the error message and the ``__dir__`` sort,
which used to be copy-pasted into all five.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

__all__ = ["export_names", "resolve_export"]


def export_names(sources: Mapping[str, str]) -> list[str]:
    """The sorted names an index exports, for ``__all__`` and ``__dir__``."""
    return sorted(sources)


def resolve_export(module_name: str, sources: Mapping[str, str], name: str) -> Any:
    """Import ``name`` from the submodule ``sources`` maps it to.

    Args:
        module_name: the index's own ``__name__`` (the import is absolute
            against it, so callers pass the package they are).
        sources: name -> submodule, the index's table.
        name: the attribute being looked up.

    Raises:
        AttributeError: when the index does not export ``name``. A miss has to
            read as a miss -- a quiet ``None`` would turn a typo into a silently
            absent feature -- and the message names the module that was asked.
    """
    source = sources.get(name)
    if source is None:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    return getattr(import_module(f"{module_name}.{source}"), name)
