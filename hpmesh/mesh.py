"""Device mesh construction -- the foundation of every parallelism dimension.

Learning note (FRAMEWORK_DESIGN.md section 5.1): ParallelDims / DeviceMesh is the
FIRST core concept. A DeviceMesh is an n-d array of ranks; each axis is one
parallelism dimension. All collectives (all-gather, reduce-scatter, all-to-all, P2P)
run along one mesh axis.

The degrees come from ``ParallelismConfig`` (the torchtitan-shaped config hpmesh
adopts) and are resolved/validated by ``ParallelDims.from_config`` -- the same
class torchtitan uses -- so the ``world_size = dp * cp * tp * pp`` constraint is
enforced in exactly one place. We then build the named (dp, cp, tp, pp) mesh the
trainer indexes uniformly.

TODO: adopt ``ParallelDims.build_mesh()``'s richer axis set (``dp_shard``,
``efsdp``, ``loss``) once FSDP/EP need to distinguish replicate from shard axes.
"""

from __future__ import annotations

import math

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .parallel.parallel_dims import ParallelDims
from .trainer.config import HybridMeshConfig
from .utils.device import device_type

# Mesh axis names. `axis` names a specific DeviceMesh axis; `dim` is for shapes.
MESH_AXES = ("dp", "cp", "tp", "pp")


def build_parallel_dims(cfg: HybridMeshConfig, world_size: int) -> ParallelDims | None:
    """Resolve the parallelism degrees against ``world_size`` (torchtitan class).

    Single-process (step 0, no torchrun) -> ``None``: no process group, no
    parallelism, so downstream code guards on ``parallel_dims is None``.
    """
    if world_size == 1:
        return None
    return ParallelDims.from_config(cfg.parallel, world_size)


def build_mesh(parallel_dims: ParallelDims | None) -> DeviceMesh | None:
    """Build a named 4-d device mesh (dp, cp, tp, pp) covering all ranks.

    Takes the *already-resolved* ``ParallelDims`` (see ``build_parallel_dims``)
    rather than a config, so a run has exactly one degree-resolution object --
    and therefore one set of process groups. ``None`` in, ``None`` out: there is
    no process group and no parallelism to describe.
    """
    if parallel_dims is None:
        return None
    # ``dp`` here is the full data-parallel group (dp_replicate * dp_shard);
    # torchtitan splits the two so FSDP can shard on one and replicate on the
    # other, which is more than this learning mesh needs to expose.
    dp = parallel_dims.dp_replicate * parallel_dims.dp_shard
    mesh_shape = (dp, parallel_dims.cp, parallel_dims.tp, parallel_dims.pp)
    assert math.prod(mesh_shape) == parallel_dims.world_size, (
        f"mesh shape {mesh_shape} (dp*cp*tp*pp) != "
        f"world_size={parallel_dims.world_size}"
    )
    return init_device_mesh(
        device_type,
        mesh_shape,
        mesh_dim_names=list(MESH_AXES),
    )


def init_distributed() -> tuple[int, int, int]:
    """Init the process group if launched under torchrun; return
    (rank, local_rank, world).

    Single-process (no torchrun) -> (0, 0, 1) and no process group, so the
    prototype also runs as a plain CPU/GPU script for step 0.
    """
    import os

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        import torch

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1
