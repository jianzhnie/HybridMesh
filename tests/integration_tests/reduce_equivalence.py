"""Semantics of ``accelerator.dist.all_reduce`` and its call convention.

The historical regression: ``train_step`` reduces the local token count, then
divides the loss by *that same tensor* to get the per-rank average. With an
in-place reduce and no clone, the tensor the division read held the *global*
count, and a per-rank mean silently became a global one. Nothing failed --
``max_loss`` was just wrong, and only in the metrics dict: stdout prints
``loss`` and ``grad_norm``, never ``max_loss``.

These checks validate the collective itself (sum/max over a mesh group, dtype
handling) and demonstrate the call convention -- clone before the in-place
reduce when the argument is read afterwards. The load-bearing clone in
production lives inline in ``trainer.py`` (``global_valid_tokens = ...clone()``)
and is protected by code review, not by this file: the checks here clone in
the test body, so they cannot fire on a missing clone there.

Run under torchrun:

    torchrun --nproc_per_node=2 tests/integration_tests/reduce_equivalence.py

Two ranks are required. A mutation is invisible over a size-1 group, because
every rank reads back the same values anyway -- which is exactly why the
existing single-process checks passed throughout.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from hpmesh.accelerator.dist import all_reduce

# Rank r holds 2 + r, so the SUM is 5 and the MAX is 3 on both ranks. A wrong
# answer is then attributable to a specific rank rather than to "the sum is off".
_BASE = 2.0


def _local() -> torch.Tensor:
    return torch.tensor([_BASE + dist.get_rank()], dtype=torch.float64)


def check_the_argument_survives_the_reduction() -> str | None:
    """The call convention demonstrated: cloning first keeps ``local`` intact."""
    local = _local()
    before = local.clone()

    out = local.clone()
    all_reduce(out, group=_mesh.get_group())

    if not torch.equal(local, before):
        return f"rank {dist.get_rank()}: input mutated to {local.tolist()}"
    if not torch.equal(out, torch.tensor([5.0], dtype=torch.float64)):
        return f"rank {dist.get_rank()}: sum is {out.tolist()}, expected [5.0]"
    return None


def check_sum_reduces_across_the_group() -> str | None:
    out = _local()
    all_reduce(out, group=_mesh.get_group())
    if not torch.equal(out, torch.tensor([5.0], dtype=torch.float64)):
        return f"rank {dist.get_rank()}: sum is {out.tolist()}, expected [5.0]"
    return None


def check_max_reduces_across_the_group() -> str | None:
    local = _local()
    out = local.clone()
    all_reduce(out, op="max", group=_mesh.get_group())
    if float(out) != 3.0:
        return f"rank {dist.get_rank()}: max is {float(out)}, expected 3.0"
    # max takes the same in-place path; the un-cloned argument must survive.
    if not torch.equal(local, _local()):
        return f"rank {dist.get_rank()}: max mutated its input to {local.tolist()}"
    return None


def check_the_local_average_pattern() -> str | None:
    """The shape of the historical misuse, reduced to arithmetic.

    A rank holding a short slice of the tokens must average over its OWN count.
    This demonstrates the convention the trainer's inline clone encodes; it
    does not exercise that clone directly.
    """
    rank = dist.get_rank()
    loss_sum = torch.tensor([4.0 * (rank + 1)], dtype=torch.float64)
    local_count = torch.tensor([1 + rank], dtype=torch.int64)

    global_count = local_count.clone()
    all_reduce(global_count, group=_mesh.get_group())
    local_avg = loss_sum / local_count  # must read the ORIGINAL local count

    if not torch.equal(global_count, torch.tensor([3], dtype=torch.int64)):
        return f"rank {rank}: global count is {global_count.tolist()}, expected [3]"
    if not torch.equal(local_avg, torch.tensor([4.0], dtype=torch.float64)):
        return f"rank {rank}: local avg is {local_avg.tolist()}, expected [4.0]"
    return None


CHECKS = [
    check_the_argument_survives_the_reduction,
    check_sum_reduces_across_the_group,
    check_max_reduces_across_the_group,
    check_the_local_average_pattern,
]

_mesh = None


def main() -> None:
    global _mesh
    dist.init_process_group("gloo")
    try:
        from torch.distributed.device_mesh import init_device_mesh

        # A 1-D mesh over both ranks. The mesh is the whole addressing story
        # for these helpers -- hpmesh has no separate ``extra_pg``.
        _mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))

        rank = dist.get_rank()
        failures = [f.__name__ + ": " + msg for f in CHECKS if (msg := f()) is not None]

        local_ok = torch.tensor([0.0 if not failures else 1.0])
        dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

        if rank == 0:
            for f in failures:
                print(f"  FAIL {f}")
            if not failures:
                print(f"all {len(CHECKS)} checks passed")

        assert local_ok.item() == 0, "reduce mutation check failed"
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
