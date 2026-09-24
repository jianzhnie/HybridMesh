"""Accelerator layer: device discovery, communication primitives, and their
configuration functions.

Members:

* ``device.py`` -- hpmesh's backend-neutral device module (NPU/CUDA/MLU/MUSA
  discovery, distributed-backend selection, per-vendor predicates).
* ``mesh.py`` -- ``init_distributed`` / ``build_parallel_dims`` /
  ``build_mesh``: process-group bootstrap and topology construction.
* ``collectives.py`` -- reductions (``dist_sum``/``dist_max``), PG timeouts,
  and EP-aware ``clip_grad_norm_``.
* ``monitoring.py`` -- device memory monitors/snapshots and ``get_peak_flops``.
* ``spmd_context.py`` -- the ambient SPMD mesh context (TLS mesh stack and
  by-name process-group queries) that trainer and ``models/common`` read.
* ``dist.py`` + ``dist_utils.py`` -- vendored from OpenMMLab's ``mmengine.dist``,
  de-mmengine'd to depend only on torch and ``.device``; a standalone toolbox
  (multi-launcher ``init_dist``, object collectives, ``cast_data_device``).

Only the vendored toolbox is re-exported here. ``mesh`` / ``collectives`` /
``monitoring`` / ``spmd_context`` are imported as submodules
(``hpmesh.accelerator.mesh`` ...): re-exporting them would make
``import hpmesh.accelerator`` pull in the parallel and trainer layers and
close an import cycle.
"""

from .dist import (
    all_gather,
    all_gather_object,
    all_reduce,
    all_reduce_dict,
    all_reduce_params,
    broadcast,
    broadcast_object_list,
    collect_results,
    collect_results_cpu,
    collect_results_gpu,
    gather,
    gather_object,
    sync_random_seed,
)
from .dist_utils import (
    barrier,
    cast_data_device,
    get_backend,
    get_comm_device,
    get_data_device,
    get_dist_info,
    get_local_rank,
    get_local_size,
    get_rank,
    get_world_size,
    infer_launcher,
    init_dist,
    init_local_group,
    is_distributed,
    is_main_process,
    master_only,
)

__all__ = [
    "all_gather",
    "all_gather_object",
    "all_reduce",
    "all_reduce_dict",
    "all_reduce_params",
    "barrier",
    "broadcast",
    "broadcast_object_list",
    "cast_data_device",
    "collect_results",
    "collect_results_cpu",
    "collect_results_gpu",
    "gather",
    "gather_object",
    "get_backend",
    "get_comm_device",
    "get_data_device",
    "get_dist_info",
    "get_local_rank",
    "get_local_size",
    "get_rank",
    "get_world_size",
    "infer_launcher",
    "init_dist",
    "init_local_group",
    "is_distributed",
    "is_main_process",
    "master_only",
    "sync_random_seed",
]
