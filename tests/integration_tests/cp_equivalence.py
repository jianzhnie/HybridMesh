"""CP>1 check: sharded-sequence attention must agree with the full-sequence run.

Run under torchrun with 2 ranks:

    torchrun --nproc_per_node=2 tests/integration_tests/cp_equivalence.py

The sequence is split across the CP ranks, so each rank holds a contiguous block
of tokens and the matching block of keys and values. Neither CP strategy is
correct on its own -- a rank that attends its own block against its own block
computes attention over a sequence that does not exist -- so this compares
against a single-rank reference that attends every token against every other.

This drives ``cp_kernel.cp_all_to_all`` -- the all-to-all the live Ulysses
kernel runs -- rather than a second copy of the redistribution. The kernel's
other inputs (a real HF attention call, query/key/value in HF's
``(batch, heads, seq, dim)`` layout) are the equivalence harnesses' job; what is
checked here is the *exchange itself*:

* KV all-gather: after one all-gather of K and V, every rank holds the full
  sequence and its own query block must come out identical to the reference.
* Ulysses: the all-to-all turns ``(T/cp, H, *)`` into ``(T, H/cp, *)``; with the
  full sequence present and the heads split, the local heads must reproduce
  those same heads of the reference. The round trip back is checked separately,
  since a lossy inverse would corrupt the residual stream without failing the
  forward.

The gather half stands alone from hpmesh's mesh plumbing on purpose, like
``ep_equivalence.py``: it builds its own mesh so it exercises the redistribution
without depending on how the trainer assembles one.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hpmesh.parallel.context_parallel.cp_kernel import cp_all_to_all

SEQ = 16
NUM_HEADS = 4
HEAD_SIZE = 8
DIM = NUM_HEADS * HEAD_SIZE

# The kernel's layout constants: q/k/v arrive HF-shaped (batch, heads, seq, dim).
SEQ_DIM = 2
HEAD_DIM = 1


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


def _gather_kv(local, group):
    """The KV strategy's all-gather: ``(T/cp, *) -> (T, *)`` along the token axis."""
    rows = [torch.empty_like(local) for _ in range(group.size())]
    dist.all_gather(rows, local.contiguous(), group=group)
    return torch.cat(rows, dim=0)


def _seq_to_head(x_THK, group):
    """The kernel's token->head exchange, adapted to this test's ``(T, H, K)``.

    The kernel works on HF's ``(batch, heads, seq, dim)``; this test's tensors are
    ``(T, H, K)``. Widening to a batch of one and putting the token count on the
    seq axis gives exactly that layout, so the assertion lands on the kernel's own
    function rather than on a reimplementation of it.
    """
    x_BHTK = x_THK.permute(1, 0, 2).unsqueeze(0)  # (1, H, T, K)
    moved = cp_all_to_all(x_BHTK, group, scatter_dim=HEAD_DIM, gather_dim=SEQ_DIM)
    return moved[0].permute(1, 0, 2)  # (T, H/cp, K)


def _head_to_seq(x_THK, group):
    """The inverse exchange, ``(T, H/cp, K) -> (T/cp, H, K)``."""
    x_BHTK = x_THK.permute(1, 0, 2).unsqueeze(0)
    moved = cp_all_to_all(x_BHTK, group, scatter_dim=SEQ_DIM, gather_dim=HEAD_DIM)
    return moved[0].permute(1, 0, 2)


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"
    assert SEQ % world == 0

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("cp",))
    group = mesh.get_group()
    failures = []

    full, local = _tensors(rank, world)

    # -- KV all-gather -------------------------------------------------------
    # Reference: every token attends against the whole sequence.
    with torch.no_grad():
        expected = _attention(full, full, full)
        k_full = _gather_kv(local, group)
        got = _attention(local, k_full, k_full)

    lo = rank * (SEQ // world)
    want = expected[lo : lo + SEQ // world]
    kv_diff = (got - want).abs().max().item()
    if kv_diff > 1e-5:
        failures.append(f"KV all-gather: {kv_diff:.3e}")

    # Non-vacuity: attending the local shard against itself -- what a no-op
    # redistribution would compute -- must differ materially. Without this the
    # check above would also pass on a redistribution that did nothing and a
    # reference that happened to match.
    if torch.allclose(_attention(local, local, local), want, atol=1e-3):
        failures.append("KV all-gather: unsharded attention matches, test is vacuous")

    # The gather must be full-length on every rank, not just correct at
    # rank-local shapes.
    if k_full.shape[0] != SEQ:
        failures.append(f"KV all-gather produced {k_full.shape[0]} rows, want {SEQ}")

    # -- Ulysses round trip --------------------------------------------------
    # A lossy inverse would corrupt the residual stream silently, so the
    # unshard(shard(x)) == x property is asserted on its own. Every rank holds
    # every head after the swap, so this checks a full tensor, not a slice.
    with torch.no_grad():
        scattered = _seq_to_head(local, group)
        restored = _head_to_seq(scattered, group)

    round_trip = (restored - local).abs().max().item()
    if round_trip > 1e-5:
        failures.append(f"Ulysses round trip: {round_trip:.3e}")

    # -- Ulysses head split ---------------------------------------------------
    # Each rank ends up with the full sequence and every other head. The heads
    # it holds are the ones the reference computes, so the local run must
    # reproduce exactly those slices.
    heads_per_rank = NUM_HEADS // world
    got_ulysses = _attention(scattered, scattered, scattered)

    head_slice = expected[:, rank * heads_per_rank : (rank + 1) * heads_per_rank]
    ulysses_diff = (got_ulysses - head_slice).abs().max().item()
    if ulysses_diff > 1e-5:
        failures.append(f"Ulysses attention: {ulysses_diff:.3e}")

    if scattered.shape != (SEQ, heads_per_rank, HEAD_SIZE):
        failures.append(
            f"Ulysses shard shape {tuple(scattered.shape)}, "
            f"want ({SEQ}, {heads_per_rank}, {HEAD_SIZE})"
        )

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
