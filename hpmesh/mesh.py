"""Device mesh construction -- the foundation of every parallelism dimension.

Learning note (FRAMEWORK_DESIGN.md section 5.1): ParallelDims / DeviceMesh is the
FIRST core concept. A DeviceMesh is an n-d array of ranks; each axis is one
parallelism dimension. All collectives (all-gather, reduce-scatter, all-to-all, P2P)
run along one mesh axis.

Constraint enforced here: world_size = dp * cp * tp * pp  (ep is carved out of the
dp*cp*tp domain; we keep ep == 1 until the MoE step).
"""

from __future__ import annotations

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .trainer.config import HybridMeshConfig

# Mesh axis names. `axis` names a specific DeviceMesh axis; `dim` is for shapes.
MESH_AXES = ("dp", "cp", "tp", "pp")


def build_mesh(cfg: HybridMeshConfig, world_size: int) -> DeviceMesh | None:
    """Build a 4-d device mesh (dp, cp, tp, pp) covering all ranks.

    Single-process (step 0, no torchrun): return None -- there is no process group
    and no parallelism, so downstream code guards on world_size / mesh is None.
    Multi-process: init_device_mesh builds the (possibly degenerate) axes so
    downstream code can index mesh["dp"] uniformly.
    """
    if world_size == 1:
        return None
    dp = cfg.derive_dp(world_size)
    mesh_shape = (dp, cfg.cp, cfg.tp, cfg.pp)
    assert _prod(mesh_shape) == world_size, (
        f"mesh shape {mesh_shape} (dp*cp*tp*pp) != world_size={world_size}"
    )
    mesh = init_device_mesh(
        "cuda" if _cuda_available() else "cpu",
        mesh_shape,
        mesh_dim_names=list(MESH_AXES),
    )
    return mesh


def _prod(shape: tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= s
    return out


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def init_distributed() -> tuple[int, int, int]:
    """Init the process group if launched under torchrun; return (rank, local_rank, world).

    Single-process (no torchrun) -> (0, 0, 1) and no process group, so the
    prototype also runs as a plain CPU/GPU script for step 0.
    """
    import os

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if _cuda_available() else "gloo"
        dist.init_process_group(backend=backend)
        if _cuda_available():
            import torch

            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1
