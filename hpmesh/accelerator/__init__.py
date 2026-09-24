"""Accelerator layer: device discovery, communication primitives, and their
configuration functions.

Members:

* ``device.py`` -- hpmesh's backend-neutral device module (NPU/CUDA/MLU/MUSA
  discovery, distributed-backend selection, per-vendor predicates).
* ``mesh.py`` -- ``build_parallel_dims`` / ``build_mesh``: topology
  construction (the trainer bootstraps its PG via
  ``dist_utils._init_dist_pytorch``).
* ``collectives.py`` -- PG timeouts (``set_pg_timeouts``) and EP-aware
  ``clip_grad_norm_``.
* ``monitoring.py`` -- device memory monitors/snapshots and ``get_peak_flops``.
* ``spmd_context.py`` -- the ambient SPMD mesh context (TLS mesh stack and
  by-name process-group queries) that trainer and ``models/common`` read.
* ``dist.py`` + ``dist_utils.py`` -- vendored from OpenMMLab's ``mmengine.dist``,
  de-mmengine'd to depend only on torch and ``.device``; a standalone toolbox
  (multi-launcher ``init_dist``, object collectives, ``cast_data_device``).

The vendored toolbox is re-exported here lazily (PEP 562): importing this
package or a sibling submodule (``hpmesh.accelerator.mesh`` ...) does not pay
for ``dist.py`` unless a toolbox name is actually touched. ``mesh`` /
``collectives`` / ``monitoring`` / ``spmd_context`` are imported as
submodules -- re-exporting them would make ``import hpmesh.accelerator``
pull in the parallel and trainer layers and close an import cycle.
"""

_EXPORT_SOURCES = {
    # dist.py
    "all_gather": "dist",
    "all_gather_object": "dist",
    "all_reduce": "dist",
    "all_reduce_dict": "dist",
    "all_reduce_params": "dist",
    "broadcast": "dist",
    "broadcast_object_list": "dist",
    "collect_results": "dist",
    "collect_results_cpu": "dist",
    "collect_results_gpu": "dist",
    "gather": "dist",
    "gather_object": "dist",
    "sync_random_seed": "dist",
    # dist_utils.py
    "barrier": "dist_utils",
    "cast_data_device": "dist_utils",
    "get_backend": "dist_utils",
    "get_comm_device": "dist_utils",
    "get_data_device": "dist_utils",
    "get_dist_info": "dist_utils",
    "get_rank": "dist_utils",
    "get_world_size": "dist_utils",
    "infer_launcher": "dist_utils",
    "init_dist": "dist_utils",
    "is_distributed": "dist_utils",
    "is_main_process": "dist_utils",
    "master_only": "dist_utils",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    """Resolve toolbox names on first touch (PEP 562 lazy re-export)."""
    source = _EXPORT_SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{source}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
