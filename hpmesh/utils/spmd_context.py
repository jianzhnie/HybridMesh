"""Ambient SPMD mesh context: the bottom layer of hpmesh's SPMD helpers.

This module holds the only shared mutable SPMD state: a thread-local stack of
active meshes plus the registered dense/sparse meshes, and the by-name axis
queries that read them (``spmd_mesh_group`` / ``spmd_mesh_size``). It depends
on torch and the PyPI ``spmd_types`` package only, so ``models/common`` (and
the trainer) can ask "which process group is the TP axis?" without a reverse
dependency on the parallel layer. The trainer enters ``spmd_context`` once
per run; every other consumer reads the ambient state through the helpers
here.

The split also removes a name collision: the old home of this code,
``hpmesh/parallel/spmd_types.py``, shadowed the PyPI ``spmd_types`` package
inside the package's own namespace.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from threading import local
from typing import Any

import spmd_types as spmd
import torch
from torch.distributed.device_mesh import DeviceMesh

# TODO: Remove after spmd_types fixes deepcopy for its variadic tuple subclass.
# PartitionSpec is immutable, so sharing it across a model deepcopy is safe.
setattr(spmd.PartitionSpec, "__deepcopy__", lambda self, memo: self)  # noqa: B010

__all__ = [
    "current_spmd_mesh",
    "set_current_spmd_mesh",
    "set_spmd_meshes",
    "spmd_context",
    "spmd_dense_mesh",
    "spmd_mesh_group",
    "spmd_mesh_size",
    "spmd_sparse_mesh",
]


_MESH_TLS = local()


def set_spmd_meshes(
    *,
    dense_mesh: DeviceMesh,
    sparse_mesh: DeviceMesh | None,
) -> None:
    """Register the SPMD meshes for dense and sparse runtime regions."""
    _MESH_TLS.dense_mesh = dense_mesh
    _MESH_TLS.sparse_mesh = sparse_mesh


def spmd_dense_mesh() -> DeviceMesh:
    """Return the registered dense SPMD mesh."""
    mesh = getattr(_MESH_TLS, "dense_mesh", None)
    assert mesh is not None, "SPMD dense mesh has not been registered"
    return mesh


def spmd_sparse_mesh() -> DeviceMesh | None:
    """Return the registered sparse SPMD mesh, if EP is enabled."""
    return getattr(_MESH_TLS, "sparse_mesh", None)


def _spmd_mesh_stack() -> list[DeviceMesh | None]:
    stack = getattr(_MESH_TLS, "mesh_stack", None)
    if stack is None:
        stack = []
        _MESH_TLS.mesh_stack = stack
    return stack


def current_spmd_mesh() -> DeviceMesh | None:
    """Return the current runtime mesh, or ``None`` if unset."""
    stack = _spmd_mesh_stack()
    if not stack:
        return None
    return stack[-1]


def spmd_mesh_size(axis_name: str) -> int:
    """Return the size of a mesh axis, or 1 if not active."""
    mesh = current_spmd_mesh()
    if mesh is None:
        return 1
    names = mesh.mesh_dim_names or ()
    if axis_name not in names:
        return 1
    return mesh.size(names.index(axis_name))


def spmd_mesh_group(axis_name: str) -> torch.distributed.ProcessGroup | None:
    """Return a non-singleton process group from the current SPMD mesh."""
    mesh = current_spmd_mesh()
    if mesh is None:
        return None
    names = mesh.mesh_dim_names or ()
    if axis_name not in names:
        return None
    group = mesh.get_group(axis_name)
    return group if group.size() > 1 else None


@contextlib.contextmanager
def set_current_spmd_mesh(mesh: DeviceMesh | None) -> Iterator[None]:
    """Set TorchTitan and spmd_types current mesh state for one runtime region."""
    stack = _spmd_mesh_stack()
    stack.append(mesh)
    if mesh is None:
        try:
            yield
        finally:
            popped = stack.pop()
            assert popped is mesh
        return

    with spmd.set_current_mesh(mesh):
        try:
            yield
        finally:
            popped = stack.pop()
            assert popped is mesh


@contextlib.contextmanager
def spmd_context(parallel_dims: Any) -> Iterator[None]:
    """Make the run's meshes answerable by name for the duration of the block.

    This is the one place the ambient SPMD state is entered. Everything
    downstream that asks "which process group is the TP axis?" -- the MoE token
    reduction, the vocab-parallel embedding, the fused dist-GEMMs -- reads it
    from here rather than receiving a mesh through its arguments, which is what
    keeps a ``DeviceMesh`` from having to thread through every model component.

    ``parallel_dims`` is a ``hpmesh.parallel.parallel_dims.ParallelDims``,
    duck-typed here as ``Any`` so this bottom layer never imports the parallel
    layer it serves.

    Two pieces of state, and both are required:

    * ``set_spmd_meshes`` registers the dense and sparse meshes so
      ``spmd_dense_mesh`` / ``spmd_sparse_mesh`` can answer. The sparse mesh is
      ``None`` unless EP is on, and ``None`` there reads as "no EP axis".
    * ``set_current_spmd_mesh`` pushes onto the mesh stack, which is what the
      by-name lookups (``spmd_mesh_group`` / ``current_spmd_mesh``) read.
      It also enters ``spmd_types.set_current_mesh``, so a model that wants
      *static* SPMD type checking gets a live mesh too.

    Without the second one the first is inert: the registry would hold a mesh
    that no lookup consults, and every caller would keep taking its "no mesh"
    branch -- the exact silent degradation this exists to remove.

    ``parallel_dims is None`` is the single-process case: there is no process
    group and no axis, so every lookup below correctly answers ``None``/``1``.
    """
    if parallel_dims is None:
        yield
        return

    set_spmd_meshes(
        dense_mesh=parallel_dims.spmd_dense_mesh(),
        sparse_mesh=parallel_dims.spmd_sparse_mesh(),
    )
    with set_current_spmd_mesh(spmd_dense_mesh()):
        yield
