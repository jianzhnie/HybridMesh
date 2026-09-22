"""Device mesh construction -- the foundation of every parallelism dimension.

Learning note (FRAMEWORK_DESIGN.md section 5.1): ParallelDims / DeviceMesh is the
FIRST core concept. A DeviceMesh is an n-d array of ranks; each axis is one
parallelism dimension. All collectives (all-gather, reduce-scatter, all-to-all, P2P)
run along one mesh axis.

The degrees come from ``ParallelConfig`` (the torchtitan-shaped config hpmesh
adopts) and are resolved/validated by ``ParallelDims.from_config`` -- the same
class torchtitan uses -- so the ``world_size = dp * cp * tp * pp`` constraint is
enforced in exactly one place.

``build_mesh`` deliberately does NOT call ``init_device_mesh`` itself. The
parallelism layer needs more than one view of the same ranks -- FSDP wants
``(dp_replicate, dp_shard, cp, tp)``, SPMD type checking wants ``(dp, cp, tp)``
with the two DP axes folded and singletons dropped, EP wants a separate sparse
mesh -- and those views have to come from ONE unflatten of the world mesh or they
end up with disjoint process groups covering the same ranks. ``ParallelDims``
owns that unflatten; this function just hands back the dense view the parallel
``apply_*`` functions index. Building a second mesh here would work right up
until something addressed a rank through one group and collected on another.
"""

from __future__ import annotations

import torch.distributed as dist

from .parallel.parallel_dims import ParallelDims
from .trainer.config import HybridMeshConfig
from .utils.device import (
    device_type,
    get_current_device,
    get_distributed_backend,
    set_device,
)

# Mesh axis names. `axis` names a specific DeviceMesh axis; `dim` is for shapes.
# These are the axes of the dense mesh the parallel layer is handed; ``pp`` is
# not among them because pipeline stages live on disjoint rank sets -- the PP
# path resolves its own views off ParallelDims instead (see
# parallel/pipeline_parallel/pp.py).
MESH_AXES = ("dp", "cp", "tp")


def build_parallel_dims(cfg: HybridMeshConfig, world_size: int) -> ParallelDims | None:
    """Resolve the parallelism degrees against ``world_size`` (torchtitan class).

    Single-process (step 0, no torchrun) -> ``None``: no process group, no
    parallelism, so downstream code guards on ``parallel_dims is None``.
    """
    if world_size == 1:
        return None
    return ParallelDims.from_config(cfg.parallel, world_size)


def build_mesh(parallel_dims: ParallelDims | None):
    """The dense ``(dp, cp, tp)`` mesh the parallel ``apply_*`` functions index.

    Takes the *already-resolved* ``ParallelDims`` (see ``build_parallel_dims``)
    rather than a config, so a run has exactly one degree-resolution object --
    and therefore one set of process groups. ``None`` in, ``None`` out: there is
    no process group and no parallelism to describe.

    Aliases ``ParallelDims.spmd_dense_mesh()``, which is the same object the
    SPMD context registers, so ``apply_tp``'s ``mesh["tp"]`` and a component's
    ``spmd_mesh_group("tp")`` resolve to the very same process group.
    """
    if parallel_dims is None:
        return None
    mesh = parallel_dims.spmd_dense_mesh()
    if mesh.mesh_dim_names != MESH_AXES:
        raise ValueError(
            f"dense mesh axes {mesh.mesh_dim_names} != expected {MESH_AXES}; the "
            "parallel layer indexes these names directly"
        )
    # The dense mesh spans dp * cp * tp ranks. PP is not an axis of it, so a
    # ``pp > 1`` run must not be handed this mesh as if it covered the world:
    # the trainer takes the per-stage dense view off ``parallel_dims`` instead.
    covered = (
        parallel_dims.dp_replicate
        * parallel_dims.dp_shard
        * parallel_dims.cp
        * parallel_dims.tp
    )
    if covered != parallel_dims.world_size:
        raise ValueError(
            f"dense mesh (dp*cp*tp = {covered}) does not cover the world "
            f"({parallel_dims.world_size} ranks); is pp > 1?"
        )
    return mesh


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
        if device_type != "cpu":
            set_device(get_current_device())
        backend = get_distributed_backend()
        dist.init_process_group(backend=backend)
        return rank, local_rank, world_size
    return 0, 0, 1
