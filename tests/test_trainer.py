"""Training-loop machinery: reductions, the data iterator, and checkpointing.

The reductions and the iterator run without a process group, which is the point
-- the parts of the loop that are easy to get wrong are the ones that do not need
a cluster to exercise. The collectives are checked in their single-rank form
(where the reduction is the identity) and their clip semantics, which is where
the real bug risk lives: clipping is easy to write such that it silently does
nothing.

Checkpointing is exercised through the real ``CheckpointManager``, which runs
single-process as long as no process group is initialized -- so the tests cover
the DCP path the trainer actually uses rather than a substitute. The optimizer
cases are deliberately built on *fresh* objects: restoring into an optimizer that
has already taken a step hides the bug this suite exists to catch, because a
cold Adam has no ``exp_avg`` tensors for DCP to write into.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hpmesh.components.checkpointer import (
    DATALOADER,
    TRAIN_STATE,
    CheckpointManager,
)
from hpmesh.components.loss import (
    IGNORE_INDEX,
    cross_entropy_loss,
    next_token_targets,
    vocab_shard_bounds,
)
from hpmesh.datasets.random_data import (
    Batch,
    DataLoaderExhausted,
    RandomTokenDataLoader,
    RandomTokenSource,
    batch_iterator,
)
from hpmesh.parallel.collectives import (
    clip_grad_norm_,
    dist_max,
    dist_sum,
    dist_sum_tensor,
)
from hpmesh.trainer.config import CheckpointConfig
from hpmesh.trainer.trainer import Trainer

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

    full = cross_entropy_loss(logits, targets, tp_group=object(), global_vocab_size=5)

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


class _TrainState:
    """Stand-in for the two counters the Trainer contributes to a checkpoint."""

    def __init__(self) -> None:
        self.step = 0
        self.ntokens_seen = 0

    def state_dict(self) -> dict[str, int]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]


def _manager(
    folder: str, model: nn.Module, optimizer, state: _TrainState, **overrides
) -> CheckpointManager:
    # keep_latest_k=0 keeps the default runs unbounded, so a test that asserts on
    # what is on disk is describing the save path rather than the purge thread.
    # Tests about retention pass keep_latest_k explicitly.
    config = {"keep_latest_k": 0, **overrides}
    return CheckpointManager(
        CheckpointConfig(enable=True, folder="checkpoint", **config),
        model_parts=[model],
        optimizer=optimizer,
        states={TRAIN_STATE: state},
        folder=folder,
    )


def _step(model: nn.Module, optimizer: torch.optim.Optimizer, times: int = 2) -> None:
    for _ in range(times):
        optimizer.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()


def _optimizer_state(optimizer) -> dict:
    return {
        param_id: {k: v.clone() for k, v in state.items()}
        for param_id, state in optimizer.state_dict()["state"].items()
    }


def test_checkpoint_round_trips_model_optimizer_and_counters(tmp_path) -> None:
    """The three things a resume needs, restored into brand-new objects."""
    model, optimizer = _model_and_optimizer()
    state = _TrainState()
    state.step, state.ntokens_seen = 7, 99
    _step(model, optimizer)

    saved_weights = model.weight.detach().clone()
    saved_optim = _optimizer_state(optimizer)

    manager = _manager(str(tmp_path), model, optimizer, state, interval=1)
    assert manager.save(7) is True
    manager.close()

    torch.manual_seed(0)
    fresh_model, fresh_optimizer = _model_and_optimizer()
    with torch.no_grad():
        fresh_model.weight.zero_()
    fresh_state = _TrainState()

    resumed = _manager(str(tmp_path), fresh_model, fresh_optimizer, fresh_state)
    assert resumed.load(-1) is True
    resumed.close()

    assert (fresh_state.step, fresh_state.ntokens_seen) == (7, 99)
    assert torch.equal(fresh_model.weight.detach(), saved_weights)

    # The load-bearing case. A fresh Adam has no exp_avg to write into, so a
    # manager that did not materialize the state first would report success
    # while leaving the optimizer cold.
    restored_optim = fresh_optimizer.state_dict()["state"]
    assert restored_optim.keys() == saved_optim.keys()
    for param_id, saved in saved_optim.items():
        for key, value in saved.items():
            assert torch.equal(restored_optim[param_id][key], value)


def test_checkpoint_save_includes_optimizer_state_without_a_prior_step(
    tmp_path,
) -> None:
    """A save at step 1 must carry optimizer state, not just the weights.

    The first checkpoint of a run is taken by an optimizer that has just stepped
    for the first time -- but any save before that would have written a model-only
    file, which restores cleanly and silently resumes from a cold optimizer.
    """
    folder = tmp_path / "checkpoint"
    model, optimizer = _model_and_optimizer()
    state = _TrainState()

    manager = _manager(str(tmp_path), model, optimizer, state, interval=1)
    manager.save(1)
    manager.close()

    assert (folder / "step-1" / ".metadata").is_file()

    fresh_model, fresh_optimizer = _model_and_optimizer()
    fresh_state = _TrainState()
    resumed = _manager(str(tmp_path), fresh_model, fresh_optimizer, fresh_state)
    resumed.load(-1)
    resumed.close()

    assert _optimizer_state(fresh_optimizer) != {}


def test_load_returns_false_when_there_is_no_checkpoint(tmp_path) -> None:
    """A first run is not an error: ``False`` lets resume be the same code path."""
    model, optimizer = _model_and_optimizer()
    manager = _manager(str(tmp_path), model, optimizer, _TrainState())
    assert manager.load(-1) is False
    manager.close()


def test_checkpoint_is_sharded_over_steps_not_over_ranks(tmp_path) -> None:
    """Every rank writes into one shared step directory; DCP records the layout.

    The old per-rank files (``rank00.pt``) could not express a sharded tensor:
    they worked only while one rank owned the whole parameter.
    """
    model, optimizer = _model_and_optimizer()
    manager = _manager(str(tmp_path), model, optimizer, _TrainState(), interval=1)

    manager.save(1)
    manager.save(3)
    manager.close()

    steps = sorted(p.name for p in (tmp_path / "checkpoint").iterdir())
    assert steps == ["step-1", "step-3"]
    # ``.metadata`` is what marks a step directory resumable; it is written per
    # step, not per rank, so its presence proves nothing collided.
    assert (tmp_path / "checkpoint" / "step-3" / ".metadata").is_file()


def test_retention_keeps_the_latest_k_and_deletes_the_rest(tmp_path) -> None:
    model, optimizer = _model_and_optimizer()
    manager = _manager(
        str(tmp_path), model, optimizer, _TrainState(), interval=1, keep_latest_k=2
    )

    for step in (1, 2, 3, 4):
        manager.save(step)
    manager.close()

    remaining = sorted(p.name for p in (tmp_path / "checkpoint").iterdir())
    # keep_latest_k counts the checkpoint the next save is about to take, so 2
    # retained slots leave the two most recent on disk.
    assert remaining == ["step-3", "step-4"]


def test_step_discovery_ignores_unparseable_directory_names(tmp_path) -> None:
    """A stray name must not be parsed into a step the loader would then pick."""
    model, optimizer = _model_and_optimizer()
    manager = _manager(str(tmp_path), model, optimizer, _TrainState(), interval=1)
    manager.save(4)
    manager.close()

    checkpoint_folder = tmp_path / "checkpoint"
    (checkpoint_folder / "step-007").mkdir()
    (checkpoint_folder / "notes").mkdir()

    fresh_model, fresh_optimizer = _model_and_optimizer()
    resumed = _manager(str(tmp_path), fresh_model, fresh_optimizer, _TrainState())
    assert resumed.load(-1) is True
    resumed.close()


# -- the dataloader seam ------------------------------------------------------
#
# The trainer drives whichever ``BaseDataLoader`` the config names, and the two
# implementations disagree about what a batch is: the synthetic one yields
# ``(B, T)`` rows of one document each, the Grain one a flat packed stream. The
# tests below pin the reconciliation, because everything downstream -- the
# shift, the denominator, the attention backend's packing check -- is written
# against what ``_as_batch`` returns.


def _random_batch(batch_size: int = 4, seq_len: int = 6) -> Batch:
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 32, (batch_size, seq_len), generator=generator)
    return Batch(input_ids=ids, labels=ids.clone())


def test_as_batch_passes_the_synthetic_shape_through_unchanged() -> None:
    batch = _random_batch()
    input_ids, labels, positions, num_valid = Trainer._as_batch(batch)
    assert torch.equal(input_ids, batch.input_ids)
    # Positions are the wrapper's arange default, and the synthetic source
    # counts its own tokens.
    assert positions is None
    assert num_valid is None


def test_as_batch_shifts_the_synthetic_labels_within_a_row() -> None:
    """Row ``r`` must never predict row ``r + 1``'s first token.

    The synthetic source hands over labels equal to its inputs; the shift is
    the trainer's. Doing it globally would pair each row's last position with
    the next document's first token -- a target the model had no context for.
    """
    batch = _random_batch(batch_size=3, seq_len=4)
    _, labels, _, _ = Trainer._as_batch(batch)

    expected = next_token_targets(batch.labels.reshape(-1), seq_len=4)
    assert torch.equal(labels, expected)
    # Every row-final position is excluded, one per row.
    assert int((labels == IGNORE_INDEX).sum()) == 3
    # And the surviving pairs are the intra-row ones.
    flat = labels.reshape(-1, 4)
    assert torch.equal(flat[:, :3], batch.labels[:, 1:])


def test_as_batch_consumes_the_grain_batch_without_consuming_its_tensors() -> None:
    """The collator's counts and mask are read here, so the model never sees them.

    Anything left in the dict becomes a model kwarg, so ``num_valid_tokens``
    (a plain int) and ``padding_mask`` (which the forward would reject) both
    have to be taken out rather than merely read.
    """
    grain_batch = {
        "input": torch.arange(8),
        "labels": torch.full((8,), IGNORE_INDEX),
        "positions": torch.arange(8),
        "padding_mask": torch.zeros(8, dtype=torch.bool),
        "num_valid_tokens": 5,
    }

    input_ids, labels, positions, num_valid = Trainer._as_batch(grain_batch)

    assert torch.equal(input_ids, torch.arange(8))
    assert positions is not None and torch.equal(positions, torch.arange(8))
    assert num_valid == 5
    # Only the model's own kwargs survive.
    assert set(grain_batch) == {"input", "labels"}
    assert labels is not None


def test_as_batch_leaves_the_grain_labels_alone() -> None:
    """The collator already shifted and masked them; shifting again would be wrong."""
    grain_batch = {
        "input": torch.arange(5),
        "labels": torch.tensor([1, 2, IGNORE_INDEX, 4, IGNORE_INDEX]),
        "num_valid_tokens": 2,
    }
    _, labels, _, _ = Trainer._as_batch(grain_batch)
    assert torch.equal(labels, torch.tensor([1, 2, IGNORE_INDEX, 4, IGNORE_INDEX]))


def test_loss_sum_does_not_count_ignored_positions() -> None:
    """The denominator must be the predictable labels, not every label."""
    logits = torch.randn(6, 8)
    labels = torch.tensor([1, 2, IGNORE_INDEX, 4, IGNORE_INDEX, IGNORE_INDEX])

    loss_sum, num_valid = Trainer._loss_sum(logits, labels)

    assert loss_sum.ndim == 0
    assert num_valid == 3
    # Passing the count explicitly must agree with recounting it.
    _, again = Trainer._loss_sum(logits, labels, num_valid_tokens=3)
    assert again == num_valid


def test_loss_sum_makes_one_prediction_per_predictable_label() -> None:
    """``logits[t]`` scores ``labels[t]``: the two are already aligned."""
    logits = torch.randn(4, 8)
    labels = torch.tensor([1, 2, 3, 4])
    loss_sum, num_valid = Trainer._loss_sum(logits, labels)
    assert num_valid == 4

    expected = F.cross_entropy(logits.float(), labels, reduction="sum")
    assert torch.allclose(loss_sum, expected)


def test_checkpoint_carries_a_dataloader_read_position(tmp_path) -> None:
    """Resuming a real corpus must resume the *data*, not just the weights.

    Without this the run would restore trained weights and then re-read the
    corpus from the beginning, silently training a second pass over the start
    of the data while the step counter said otherwise.
    """
    model, optimizer = _model_and_optimizer()
    loader = RandomTokenDataLoader(
        seed=3, vocab_size=16, batch_size=4, seq_len=6, dp_rank=0, dp_world_size=1
    )

    manager = CheckpointManager(
        CheckpointConfig(enable=True, folder="checkpoint", keep_latest_k=0, interval=1),
        model_parts=[model],
        optimizer=optimizer,
        states={TRAIN_STATE: _TrainState(), DATALOADER: loader},
        folder=str(tmp_path),
    )
    for _ in range(3):
        next(iter(loader))
    assert manager.save(3)
    manager.close()

    # A fresh loader restored from the checkpoint must continue where the old
    # one stopped rather than restart.
    resumed_loader = RandomTokenDataLoader(
        seed=3, vocab_size=16, batch_size=4, seq_len=6, dp_rank=0, dp_world_size=1
    )
    fresh_model, fresh_optimizer = _model_and_optimizer()
    resumed = CheckpointManager(
        CheckpointConfig(enable=True, folder="checkpoint", keep_latest_k=0, interval=1),
        model_parts=[fresh_model],
        optimizer=fresh_optimizer,
        states={TRAIN_STATE: _TrainState(), DATALOADER: resumed_loader},
        folder=str(tmp_path),
    )
    assert resumed.load(-1) is True
    resumed.close()

    # Where the original would have gone next.
    reference = RandomTokenDataLoader(
        seed=3, vocab_size=16, batch_size=4, seq_len=6, dp_rank=0, dp_world_size=1
    )
    for _ in range(3):
        next(iter(reference))
    expected = next(iter(reference))

    got = next(iter(resumed_loader))
    assert torch.equal(got.input_ids, expected.input_ids)
    assert torch.equal(got.labels, expected.labels)
