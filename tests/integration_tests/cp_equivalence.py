"""CP>1 check: sharded-sequence attention must agree with the full-sequence run.

Run under torchrun with 2 ranks:

    torchrun --nproc_per_node=2 tests/cp_equivalence.py

The sequence is split across the CP ranks, so each rank holds a contiguous block
of tokens and the matching block of keys and values. Neither CP strategy is
correct on its own -- a rank that attends its own block against its own block
computes attention over a sequence that does not exist -- so this compares
against a single-rank reference that attends every token against every other.

What each path is checked for:

* KV all-gather: after one all-gather of K and V, every rank holds the full
  sequence and its own query block must come out identical to the reference.
* Ulysses: the all-to-all turns ``(T/cp, H, *)`` into ``(T, H/cp, *)``; with the
  full sequence present and the heads split, the local heads must reproduce
  those same heads of the reference. The round trip back is checked separately,
  since a lossy inverse would corrupt the residual stream without failing the
  forward.

Stands alone from hpmesh's mesh plumbing on purpose, like ``ep_equivalence.py``:
it builds its own mesh so it exercises the redistributions without depending on
how the trainer assembles one.
"""

from __future__ import annotations

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hpmesh.parallel.context_parallel.primitives import (
    HEAD_DIM,
    TOKEN_DIM,
    KVAllGatherContextParallel,
    UlyssesContextParallel,
    cp_redistribute,
)
from hpmesh.utils.spmd_context import set_current_spmd_mesh

SEQ = 16
NUM_HEADS = 4
HEAD_SIZE = 8
DIM = NUM_HEADS * HEAD_SIZE


def _tensors(rank: int, cp_size: int, seed: int = 0):
    """A full contiguous sequence, plus this rank's shard of it."""
    g = torch.Generator().manual_seed(seed)
    full = torch.randn(SEQ, NUM_HEADS, HEAD_SIZE, generator=g)
    lo = rank * (SEQ // cp_size)
    shard = full[lo : lo + SEQ // cp_size]
    return full, shard


def _attention(q_THK, k_THK, v_THV) -> torch.Tensor:
    """Plain causal-free attention, local in H so it works per CP strategy."""
    q = q_THK.transpose(0, 1).float()  # H T K
    k = k_THK.transpose(0, 1).float()  # H T K
    v = v_THV.transpose(0, 1).float()  # H T V
    scores = q @ k.transpose(-1, -2) / (HEAD_SIZE**0.5)
    return (scores.softmax(-1) @ v).transpose(0, 1)  # T H V


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"
    assert SEQ % world == 0

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("cp",))
    failures = []

    with set_current_spmd_mesh(mesh):
        full, local = _tensors(rank, world)

        # -- KV all-gather ---------------------------------------------------
        # Reference: every token attends against the whole sequence.
        with torch.no_grad():
            expected = _attention(full, full, full)

        kv = KVAllGatherContextParallel()
        with torch.no_grad():
            q, k, v = kv(local, local, local)
            got = _attention(q, k, v)

        lo = rank * (SEQ // world)
        want = expected[lo : lo + SEQ // world]
        kv_diff = (got - want).abs().max().item()
        if kv_diff > 1e-5:
            failures.append(f"KV all-gather: {kv_diff:.3e}")

        # Non-vacuity: attending the local shard against itself -- what a no-op
        # redistribution would compute -- must differ materially. Without this
        # the check above would also pass on a redistribution that did nothing
        # and a reference that happened to match.
        if torch.allclose(_attention(local, local, local), want, atol=1e-3):
            failures.append(
                "KV all-gather: unsharded attention matches, test is vacuous"
            )

        # The gather must be full-length on every rank, not just correct at
        # rank-local shapes.
        if k.shape[0] != SEQ or v.shape[0] != SEQ:
            failures.append(f"KV all-gather produced {k.shape[0]} rows, want {SEQ}")

        # -- Ulysses round trip ----------------------------------------------
        # A lossy inverse would corrupt the residual stream silently, so the
        # unshard(shard(x)) == x property is asserted on its own.
        with torch.no_grad():
            scattered = cp_redistribute(
                local, src=spmd.S(TOKEN_DIM), dst=spmd.S(HEAD_DIM)
            )
            restored = cp_redistribute(
                scattered, src=spmd.S(HEAD_DIM), dst=spmd.S(TOKEN_DIM)
            )

        round_trip = (restored - local).abs().max().item()
        if round_trip > 1e-5:
            failures.append(f"Ulysses round trip: {round_trip:.3e}")

        # -- Ulysses head split ----------------------------------------------
        # Each rank ends up with the full sequence and every other head. The
        # heads it holds are the ones the reference computes, so the local run
        # must reproduce exactly those slices.
        ulysses = UlyssesContextParallel()
        with torch.no_grad():
            q, k, v = ulysses.shard(local, local, local)
            got_ulysses = _attention(q, k, v)

        heads_per_rank = NUM_HEADS // world
        head_slice = expected[:, rank * heads_per_rank : (rank + 1) * heads_per_rank]
        ulysses_diff = (got_ulysses - head_slice).abs().max().item()
        if ulysses_diff > 1e-5:
            failures.append(f"Ulysses attention: {ulysses_diff:.3e}")

        if q.shape[0] != SEQ or q.shape[1] != heads_per_rank:
            failures.append(
                f"Ulysses shard shape {tuple(q.shape)}, "
                f"want ({SEQ}, {heads_per_rank}, {HEAD_SIZE})"
            )

        # -- missing CP mesh --------------------------------------------------
        # A no-op fallback here would produce a confidently wrong answer, so the
        # redistribution must refuse rather than silently pass tensors through.
        with set_current_spmd_mesh(None):
            try:
                cp_redistribute(local, src=spmd.S(TOKEN_DIM), dst=spmd.R)
                failures.append("missing CP mesh: no error raised")
            except RuntimeError:
                pass

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"seq={SEQ} heads={NUM_HEADS} head_dim={HEAD_SIZE} cp_size={world}")
        print(f"KV all-gather  max abs diff = {kv_diff:.3e}")
        print(f"Ulysses        max abs diff = {ulysses_diff:.3e}")
        print(f"Ulysses round trip          = {round_trip:.3e}")
        print(f"failed ranks   = {int(local_ok.item())}")
        if failures:
            for f in failures:
                print(f"  FAIL {f}")
        else:
            print("all checks passed")

    assert local_ok.item() == 0, "CP equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
