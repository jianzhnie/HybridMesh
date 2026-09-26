"""Vocab-parallel cross-entropy check: sharded == full, value AND gradient.

Run under torchrun with 2 ranks (from the repo root):

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/vocab_parallel_loss_equivalence.py

Why this file exists
--------------------
``LossParallelCrossEntropy`` is 100 lines of TP all-reduce and shard masking,
and until now **none of it had ever executed**: the trainer does not shard the
lm_head, so ``cross_entropy_loss`` always took the plain path. Read the coverage
before trusting a change here -- this is the only thing that runs it.

What it checks
--------------
The reference is ``torch.nn.functional.cross_entropy`` over the *full* logits,
which is exactly what the non-sharded path uses. Each rank holds one column
slice ``[start, end)`` of that same full tensor and runs the parallel path;
value and per-rank gradient are compared against the reference's.

The shard bounds are the risky part, so the cases are chosen to exercise them
rather than to look tidy:

* even split (V=8, tp=2 -> [0,4) [4,8)),
* **uneven split** (V=7, tp=2 -> [0,4) [4,7), and V=5 -> [0,3) [3,5)), where the
  last rank owns fewer classes than the others -- the branch the fused
  ``vocab_shard_bounds`` exists to get right, and the one an even-only test
  would never take,
* ``IGNORE_INDEX`` labels, including a row whose targets all fall in another
  rank's shard,
* ``reduction="none"``, where the shape of the backward's ``grad_output``
  differs (a per-token [T] vector rather than a scalar),
* a full-vocab logits tensor passed *with* a tp_group, which must take the
  plain path -- selection is by shape, so a mismatch here would silently
  compute the wrong thing rather than fail.

Everything runs in fp32 on CPU/gloo. The tolerance is relative, not absolute:
an uneven split sums over a different number of terms than the reference, so a
loss of magnitude ~128 differs by ~1 ULP (measured 1.5e-05 absolute = 1.19e-07
relative = 1.00 * fp32 eps). A tolerance that rejected that would be rejecting
floating point, not catching a bug -- and a shard bug produces O(1) errors, so
the margin is enormous either way.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hpmesh.components.loss import (
    IGNORE_INDEX,
    LossParallelCrossEntropy,
    compute_logprobs,
    cross_entropy_loss,
    vocab_shard_bounds,
)
from hpmesh.utils.batch_invariant import set_batch_invariant_mode

T = 64  # tokens per rank's local shard
SEED = 42
# Relative. See the module docstring: the sanctioned divergence is summation
# order, which sits at ~1 ULP.
RTOL = 1e-4
ATOL = 1e-6

failures: list[str] = []


def _fail(rank: int, msg: str) -> None:
    failures.append(f"rank {rank}: {msg}")


def _agree(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a, b, rtol=RTOL, atol=ATOL)


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _full_logits(vocab: int) -> torch.Tensor:
    """The same ``[T, V]`` tensor on every rank.

    Every rank must slice *the same* matrix, or the comparison is meaningless --
    so this is seeded rather than drawn, and the randn is fully consumed.
    """
    torch.manual_seed(SEED + vocab)
    return torch.randn(T, vocab, dtype=torch.float32)


def _check_sharded_loss(
    rank: int, group: dist.ProcessGroup, *, vocab: int, name: str
) -> None:
    """value + gradient of the parallel path against F.cross_entropy on full."""
    world = dist.get_world_size(group)
    start, end = vocab_shard_bounds(vocab, world, rank)

    logits = _full_logits(vocab)
    # A mix of real targets and IGNORE_INDEX, so both masks are exercised.
    torch.manual_seed(SEED)
    labels = torch.randint(0, vocab, (T,), dtype=torch.long)
    labels[::7] = IGNORE_INDEX

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref = F.cross_entropy(
        ref_logits.float(), labels, reduction="sum", ignore_index=IGNORE_INDEX
    )
    ref.backward()

    local = logits[:, start:end].detach().clone().requires_grad_(True)
    got = cross_entropy_loss(local, labels, tp_group=group, global_vocab_size=vocab)
    got.backward()

    # reduction="sum" all-reduces, so every rank computes the global sum.
    if not _agree(got.detach(), ref.detach()):
        _fail(
            rank,
            f"{name}: loss {got.item():.6f} vs reference {ref.item():.6f} "
            f"(max diff {_max_diff(got.detach(), ref.detach()):.3e})",
        )

    grad_diff = _max_diff(local.grad, ref_logits.grad[:, start:end])
    if not _agree(local.grad, ref_logits.grad[:, start:end]):
        _fail(rank, f"{name}: local grad max diff {grad_diff:.3e}")

    # reduction="none" must agree with the scalar path, per token.
    local2 = logits[:, start:end].detach().clone().requires_grad_(True)
    per_token = LossParallelCrossEntropy.apply(local2, labels, group, vocab, "none")
    ref_none = F.cross_entropy(
        logits.float(), labels, reduction="none", ignore_index=IGNORE_INDEX
    )
    if not _agree(per_token.detach(), ref_none):
        _fail(
            rank,
            f"{name}: reduction='none' max diff "
            f"{_max_diff(per_token.detach(), ref_none):.3e}",
        )
    if not _agree(per_token.detach().sum(), ref.detach()):
        _fail(
            rank,
            f"{name}: sum(none) != sum (diff "
            f"{_max_diff(per_token.detach().sum(), ref.detach()):.3e})",
        )

    # compute_logprobs returns -NLL on the sharded path.
    local3 = logits[:, start:end].detach().clone().requires_grad_(True)
    logprobs = compute_logprobs(local3, labels, tp_group=group, global_vocab_size=vocab)
    if not _agree(logprobs, -ref_none):
        _fail(
            rank,
            f"{name}: compute_logprobs max diff {_max_diff(logprobs, -ref_none):.3e}",
        )

    # return_entropy on the sharded path must not silently drop the entropy,
    # and the no-gather computation must match the full-vocab formula.
    local4 = logits[:, start:end].detach().clone().requires_grad_(True)
    sharded_logprobs, sharded_entropy = compute_logprobs(
        local4, labels, tp_group=group, global_vocab_size=vocab, return_entropy=True
    )
    ref_entropy = torch.logsumexp(logits.float(), dim=-1) - (
        torch.softmax(logits.float(), dim=-1) * logits.float()
    ).sum(dim=-1)
    if not _agree(sharded_logprobs.detach(), -ref_none):
        _fail(rank, f"{name}: sharded logprobs+entropy logprobs diverged")
    if not _agree(sharded_entropy, ref_entropy):
        _fail(
            rank,
            f"{name}: sharded entropy max diff "
            f"{_max_diff(sharded_entropy, ref_entropy):.3e}",
        )
    if sharded_entropy.requires_grad:
        _fail(rank, f"{name}: sharded entropy joined the autograd graph")

    # Batch-invariant mode gathers the shards first, so the logprobs and the
    # entropy must equal the full-vocab computation exactly (same op sequence).
    set_batch_invariant_mode(True)
    try:
        local5 = logits[:, start:end].detach().clone().requires_grad_(True)
        bi_logprobs, bi_entropy = compute_logprobs(
            local5,
            labels,
            tp_group=group,
            global_vocab_size=vocab,
            return_entropy=True,
        )
    finally:
        set_batch_invariant_mode(False)
    plain_logprobs, plain_entropy = compute_logprobs(
        logits.clone(), labels, return_entropy=True
    )
    if not _agree(bi_logprobs.detach(), plain_logprobs.detach()):
        _fail(
            rank,
            f"{name}: batch-invariant gather logprobs max diff "
            f"{_max_diff(bi_logprobs.detach(), plain_logprobs.detach()):.3e}",
        )
    if not _agree(bi_entropy, plain_entropy):
        _fail(
            rank,
            f"{name}: batch-invariant gather entropy max diff "
            f"{_max_diff(bi_entropy, plain_entropy):.3e}",
        )

    # The gather's backward slices the shared full-vocab gradient back to this
    # rank's shard rather than all-reducing it.
    bi_logprobs.sum().backward()
    plain_for_grad = logits.clone().requires_grad_(True)
    compute_logprobs(plain_for_grad, labels).sum().backward()
    if local5.grad is None or plain_for_grad.grad is None:
        _fail(rank, f"{name}: batch-invariant gather backward produced no grad")
    elif not _agree(local5.grad, plain_for_grad.grad[:, start:end]):
        _fail(
            rank,
            f"{name}: batch-invariant gather grad max diff "
            f"{_max_diff(local5.grad, plain_for_grad.grad[:, start:end]):.3e}",
        )


def _check_shape_dispatch(rank: int, group: dist.ProcessGroup, *, vocab: int) -> None:
    """Full-vocab logits + a tp_group must take the plain path, not the parallel one.

    Selection is by ``pred.shape[-1] != global_vocab_size``. If that ever
    inverted, this would compute a sharded loss over a full-vocab tensor: no
    shape error, just a wrong number.

    Note this takes the plain path on *every* rank holding a full-vocab tensor,
    including a rank that is not rank 0 -- it does no collective at all, so it
    is safe to run unconditionally.
    """
    logits = _full_logits(vocab).requires_grad_(True)
    torch.manual_seed(SEED)
    labels = torch.randint(0, vocab, (T,), dtype=torch.long)

    got = cross_entropy_loss(logits, labels, tp_group=group, global_vocab_size=vocab)
    want = F.cross_entropy(
        logits.float(), labels, reduction="sum", ignore_index=IGNORE_INDEX
    )
    if not _agree(got.detach(), want.detach()):
        _fail(
            rank,
            f"full-vocab dispatch: {got.item():.6f} vs {want.item():.6f}",
        )


def _check_rejections(rank: int, group: dist.ProcessGroup, *, vocab: int) -> None:
    """An empty shard is refused rather than silently computing nothing.

    With ``V < tp`` the bounds clamp, so *some* rank ends up with a zero-width
    slice -- ``vocab_shard_bounds(1, 2, 1)`` is ``(1, 1)`` -- and that rank has
    no classes to answer for.

    Only that rank calls. The others must *not*, deliberately: the reason this
    cannot deadlock is that the empty rank raises before the first collective,
    so no one is left waiting on a reduction it will never join. Having the
    other ranks call the same function here would be the hang, not the test.
    """
    world = dist.get_world_size(group)
    start, end = vocab_shard_bounds(vocab, world, rank)
    if end - start != 0:
        return

    local = torch.randn(T, 0, requires_grad=True)
    labels = torch.zeros(T, dtype=torch.long)
    try:
        cross_entropy_loss(local, labels, tp_group=group, global_vocab_size=vocab)
    except ValueError as error:
        if "empty vocab shards" not in str(error):
            _fail(rank, f"empty shard raised the wrong error: {error}")
    else:
        _fail(rank, f"empty shard (V={vocab}) was not rejected on rank {rank}")


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check is written for 2 ranks, got {world}"
    group = dist.group.WORLD

    # -- shard bounds: the even/uneven shapes this file is built around --------
    bounds = {v: vocab_shard_bounds(v, world, rank) for v in (8, 7, 5)}
    expected = {
        (8, 0): (0, 4),
        (8, 1): (4, 8),
        (7, 0): (0, 4),  # uneven: the last rank owns 3 classes
        (7, 1): (4, 7),
        (5, 0): (0, 3),
        (5, 1): (3, 5),
    }
    for (vocab, r), want in expected.items():
        if r != rank:
            continue
        if bounds[vocab] != want:
            _fail(
                rank,
                f"vocab_shard_bounds({vocab}, 2, {r}) = {bounds[vocab]}, want {want}",
            )

    _check_shape_dispatch(rank, group, vocab=8)
    _check_sharded_loss(rank, group, vocab=8, name="even (V=8)")
    _check_sharded_loss(rank, group, vocab=7, name="uneven (V=7)")
    _check_sharded_loss(rank, group, vocab=5, name="uneven (V=5)")

    # V < world: one rank's shard is empty, which has no well-defined answer.
    _check_rejections(rank, group, vocab=1)

    # Every rank must agree every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"vocab-parallel CE, tokens={T} world={world}")
        for vocab in (8, 7, 5):
            print(
                f"  bounds V={vocab}: {vocab_shard_bounds(vocab, world, 0)} "
                f"{vocab_shard_bounds(vocab, world, 1)}"
            )
        print(f"failed ranks = {int(local_ok.item())}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "vocab-parallel loss equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
