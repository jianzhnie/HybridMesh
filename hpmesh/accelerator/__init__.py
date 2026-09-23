"""Accelerator layer: device discovery plus vendored distributed collectives.

``device.py`` is hpmesh's backend-neutral device module (NPU/CUDA/MLU/MUSA
discovery, distributed-backend selection, per-vendor predicates). ``dist.py``
and ``utils.py`` are vendored from OpenMMLab's ``mmengine.dist``, de-mmengine'd
to depend only on torch and ``.device`` -- a standalone toolbox (multi-launcher
``init_dist``, object collectives, ``cast_data_device``) that is not part of
the trainer's assembly path.
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
from .utils import (
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
