"""EP-aware gradient norm and clipping must match the global reference.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/ep_grad_norm_equivalence.py

Dense gradients are replicated over EP and must be counted once. Expert
gradients are physically partitioned and their norm contributions must be
summed across the EP group. The test chooses different expert gradients on the
two ranks so an implementation that omits the EP reduction cannot pass.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hpmesh.accelerator.collectives import clip_grad_norm_


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check requires 2 ranks, got {world}"
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("ep",))

    dense = torch.nn.Parameter(torch.tensor([1.0]))
    expert = torch.nn.Parameter(torch.tensor([1.0]))
    dense.grad = torch.tensor([3.0])
    expert.grad = torch.tensor([4.0 if rank == 0 else 12.0])

    total = clip_grad_norm_(
        [dense, expert],
        max_norm=6.5,
        foreach=False,
        ep_mesh=mesh,
        expert_parameters=[expert],
    )

    # sqrt(3^2 + 4^2 + 12^2) = 13, so the clipping coefficient is 1/2.
    torch.testing.assert_close(total, torch.tensor(13.0))
    torch.testing.assert_close(dense.grad, torch.tensor([1.5]))
    expected_expert = torch.tensor([2.0 if rank == 0 else 6.0])
    torch.testing.assert_close(expert.grad, expected_expert)

    # Reporting-only mode performs the same global reduction without clipping.
    dense.grad = torch.tensor([3.0])
    expert.grad = torch.tensor([4.0 if rank == 0 else 12.0])
    reported = clip_grad_norm_(
        [dense, expert],
        max_norm=0.0,
        foreach=False,
        ep_mesh=mesh,
        expert_parameters=[expert],
    )
    torch.testing.assert_close(reported, torch.tensor(13.0))
    torch.testing.assert_close(dense.grad, torch.tensor([3.0]))

    if rank == 0:
        print("all checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun supplies these; keeping the access explicit makes accidental
    # direct invocation fail with a useful process-group error.
    assert "RANK" in os.environ
    main()
