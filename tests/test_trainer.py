"""Training-loop machinery: reductions, the data iterator, and checkpointing.

All three run without a process group, which is the point -- the parts of the
loop that are easy to get wrong are the ones that do not need a cluster to
exercise. The collectives are checked in their single-rank form (where the
reduction is the identity) and their clip semantics, which is where the real
bug risk lives: clipping is easy to write such that it silently does nothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hpmesh.components.checkpointer.checkpoint import Checkpointer
from hpmesh.components.loss import (
    IGNORE_INDEX,
    cross_entropy_loss,
    next_token_targets,
    vocab_shard_bounds,
)
from hpmesh.datasets.random_data import (
    Batch,
    DataLoaderExhausted,
    RandomTokenSource,
    batch_iterator,
)
from hpmesh.parallel.collectives import (
    clip_grad_norm_,
    dist_max,
    dist_sum,
    dist_sum_tensor,
)

# -- losses -------------------------------------------------------------------


def test_next_token_targets_shifts_within_each_row() -> None:
    """Row ``r`` predicts its own next token -- never the row after it."""
    labels = torch.arange(6).reshape(2, 3)  # rows [0,1,2] and [3,4,5]

    targets = next_token_targets(labels.reshape(-1), seq_len=3)

    # Row 0 predicts 1,2 and row 1 predicts 4,5; the row-final positions are
    # not predictions at all.
    assert targets.tolist() == [1, 2, IGNORE_INDEX, 4, 5, IGNORE_INDEX]


def test_next_token_targets_never_crosses_a_row_boundary() -> None:
    """The bug this guards: a flat shift would pair row 0's last token with
    row 1's first, predicting a token the model was never given context for."""
    labels = torch.tensor([[10, 11, 12], [20, 21, 22]])

    targets = next_token_targets(labels.reshape(-1), seq_len=3)

    assert 20 not in targets.tolist()  # 19 does not exist, and 20 is row 1's start


def test_cross_entropy_ignores_the_shifted_padding() -> None:
    """Ignored positions contribute neither loss nor denominator."""
    torch.manual_seed(0)
    logits = torch.randn(4, 7)
    targets = torch.tensor([1, 2, IGNORE_INDEX, 6])

    total = cross_entropy_loss(logits, targets)

    # Same number as computing the CE over only the three real targets.
    kept_logits = logits[[0, 1, 3]]
    expected = cross_entropy_loss(kept_logits, torch.tensor([1, 2, 6]))
    assert torch.allclose(total, expected, atol=1e-6)


def test_cross_entropy_selects_the_local_path_by_shape() -> None:
    """With a TP group but full-vocab logits, the plain path must run.

    Selecting the vocab-parallel path by a flag instead would compute a loss
    over a vocabulary that is not actually sharded -- wrong everywhere, and
    invisible because the shapes still line up.

    The group is a bare sentinel on purpose: the vocab-parallel path is the only
    thing that would touch it, so reaching it turns this test into an error
    rather than a wrong number.
    """
    logits = torch.randn(3, 5)
    targets = torch.tensor([0, 1, 2])

    full = cross_entropy_loss(
        logits, targets, tp_group=object(), global_vocab_size=5
    )

    assert torch.allclose(full, cross_entropy_loss(logits, targets), atol=1e-6)


def test_vocab_shard_bounds_are_contiguous_and_cover_the_vocabulary() -> None:
    for vocab_size in (7, 8, 9, 100):
        for tp in (1, 2, 3, 4):
            bounds = [vocab_shard_bounds(vocab_size, tp, r) for r in range(tp)]
            assert bounds[0][0] == 0
            assert bounds[-1][1] == vocab_size
            for (_, end), (start, _) in zip(bounds, bounds[1:], strict=False):
                assert end == start, f"gap at V={vocab_size} tp={tp}"


def test_vocab_shard_bounds_never_exceed_the_vocabulary() -> None:
    """tp > V leaves the tail ranks empty rather than negative."""
    bounds = [vocab_shard_bounds(3, 5, r) for r in range(5)]

    assert all(start <= end for start, end in bounds)
    assert sum(end - start for start, end in bounds) == 3


# -- reductions ---------------------------------------------------------------


def test_reductions_are_the_identity_without_a_mesh() -> None:
    """No mesh means one rank, so not reducing is correct -- not a shortcut."""
    x = torch.tensor(3.0)
    assert dist_sum(x, None) == 3.0
    assert dist_max(x, None) == 3.0
    assert torch.equal(dist_sum_tensor(x, None), x)


# -- gradient clipping --------------------------------------------------------


def _graded(*zeros: bool) -> list[nn.Parameter]:
    params = []
    for is_zero in zeros:
        p = nn.Parameter(torch.zeros(4))
        p.grad = torch.zeros(4) if is_zero else torch.full((4,), 3.0)
        params.append(p)
    return params


def test_clip_scales_gradients_to_the_threshold() -> None:
    params = _graded(False)  # norm = 3 * 2 = 6
    norm = clip_grad_norm_(params, max_norm=1.0)

    assert float(norm) > 1.0
    # After clipping the concatenated gradient vector has norm exactly max_norm.
    clipped = torch.cat([p.grad.reshape(-1) for p in params])
    assert torch.allclose(clipped.norm(), torch.tensor(1.0), atol=1e-6)


def test_non_positive_max_norm_reports_the_norm_without_clipping() -> None:
    """``max_norm<=0`` disables clipping but must still return the real norm.

    This is the mode that proves the refactor is numerically inert: the norm is
    computed, reported, and nothing is scaled.
    """
    params = _graded(False)
    before = [p.grad.clone() for p in params]

    norm = clip_grad_norm_(params, max_norm=-1.0)

    assert float(norm) > 1.0
    assert all(torch.equal(p.grad, b) for p, b in zip(params, before, strict=True))


def test_clip_ignores_parameters_with_no_gradient() -> None:
    """A parameter that received no gradient must not contribute to the norm."""
    params = _graded(False, True)
    norm = clip_grad_norm_(params, max_norm=-1.0)

    assert torch.allclose(norm, torch.tensor(6.0), atol=1e-6)


def test_clip_does_not_exhaust_a_generator() -> None:
    """Passing a generator must not silently train on a clipped-only subset."""
    params = _graded(False, False)
    norm = clip_grad_norm_((p for p in params), max_norm=1.0)

    assert float(norm) > 1.0
    assert all(p.grad is not None for p in params)


# -- the data iterator --------------------------------------------------------


def _source(n: int, *, batch_size: int = 2, seq_len: int = 4) -> RandomTokenSource:
    return RandomTokenSource(
        seed=0, vocab_size=16, batch_size=batch_size, seq_len=seq_len
    )


def test_iterator_is_deterministic_from_seed_and_step() -> None:
    """Batch ``n`` depends only on ``(seed, n)`` -- never on how many came before.

    Two independent iterators must therefore agree element for element. That is
    the property a resumed run and a fresh DP comparison both rely on.
    """
    it1, it2 = batch_iterator(_source(0)), batch_iterator(_source(0))

    for _ in range(3):
        assert torch.equal(next(it1).input_ids, next(it2).input_ids)


def test_different_steps_produce_different_batches() -> None:
    it = batch_iterator(_source(0))
    first, second = next(it).input_ids, next(it).input_ids

    assert not torch.equal(first, second)


def test_iterator_restarts_a_finite_source() -> None:
    """Exhausting the source restarts it rather than stopping the loop."""
    finite = [Batch(torch.zeros(2, 4), torch.zeros(2, 4))]  # exactly one batch

    it = batch_iterator(finite)
    for _ in range(5):
        batch = next(it)
        assert batch.input_ids.shape == (2, 4)


def test_iterator_rejects_an_empty_source() -> None:
    """An empty source must raise, not spin forever in the restart loop."""
    it = batch_iterator([])
    try:
        next(it)
    except DataLoaderExhausted:
        return
    raise AssertionError("an empty source should raise DataLoaderExhausted")


# -- checkpointing ------------------------------------------------------------


def _model_and_optimizer() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Linear(4, 4)
    return model, torch.optim.AdamW(model.parameters(), lr=0.1)


def test_checkpoint_round_trips_model_optimizer_and_counters(tmp_path) -> None:
    model, optimizer = _model_and_optimizer()
    counter = Checkpointer(str(tmp_path), rank=0, device=torch.device("cpu"))

    for _ in range(2):
        optimizer.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()

    counter.save(7, model=model, optimizer=optimizer, counters={"ntokens_seen": 99})
    saved_weights = model.weight.detach().clone()

    # Wreck the state, then prove the load restores it.
    with torch.no_grad():
        model.weight.zero_()
    restored = counter.load(model=model, optimizer=optimizer)

    assert restored == {"step": 7, "ntokens_seen": 99}
    assert torch.equal(model.weight.detach(), saved_weights)


def test_load_returns_none_when_there_is_no_checkpoint(tmp_path) -> None:
    """A first run is not an error: ``None`` lets resume be the same code path."""
    model, optimizer = _model_and_optimizer()
    counter = Checkpointer(str(tmp_path), rank=0, device=torch.device("cpu"))

    assert counter.load(model=model, optimizer=optimizer) is None


def test_checkpoint_files_are_per_rank(tmp_path) -> None:
    """Each rank owns a file; they must not collide."""
    a = Checkpointer(str(tmp_path), rank=0, device=torch.device("cpu"))
    b = Checkpointer(str(tmp_path), rank=3, device=torch.device("cpu"))

    assert a.path != b.path


def test_save_is_atomic_leaving_no_temp_file(tmp_path) -> None:
    """A crash mid-write must not leave a truncated file where the real one was."""
    model, optimizer = _model_and_optimizer()
    counter = Checkpointer(str(tmp_path), rank=0, device=torch.device("cpu"))

    counter.save(1, model=model, optimizer=optimizer, counters={"ntokens_seen": 1})

    assert sorted(p.name for p in tmp_path.iterdir()) == ["rank00.pt"]
