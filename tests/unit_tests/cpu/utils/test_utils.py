"""Pure-function edges: vocabulary sharding, target shifting, coalesced IO.

These are the pieces that a training run calls constantly but never *varies* --
the shapes that reach them depend on the config, and the configs the suite
already exercises are all well-formed. So the branches that decide what to do
with an empty shard, a ragged label stream, or a remote URI have no coverage
from the end-to-end tests no matter how many of those run.

Everything here is CPU-only and needs no process group; the pieces that do
(sharded cross-entropy) are covered in ``integration_tests/``.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from hpmesh.components.loss import (
    IGNORE_INDEX,
    next_token_targets,
    vocab_shard_bounds,
)
from hpmesh.utils import filesystem
from hpmesh.utils.gc import GarbageCollection

# -- vocab_shard_bounds -------------------------------------------------------


def test_the_last_shard_absorbs_the_remainder() -> None:
    """``V=10, tp=3`` -> ``[0,4) [4,8) [8,10)``: even except the final chunk."""
    assert vocab_shard_bounds(10, 3, 0) == (0, 4)
    assert vocab_shard_bounds(10, 3, 1) == (4, 8)
    assert vocab_shard_bounds(10, 3, 2) == (8, 10)


def test_a_size_wider_than_the_vocabulary_yields_empty_not_negative() -> None:
    """The clamp exists so callers reject an empty shard rather than slicing backwards.

    Without the ``min(global_vocab_size, ...)``, rank 7 of a 8-way split of a
    4-token vocabulary would get ``start=4, end=... `` past the end and a
    negative length. The callers (the sharded cross-entropy) turn the empty
    slice into a loud error; this only guarantees the bounds are sane.
    """
    start, end = vocab_shard_bounds(4, 8, 7)
    assert start == 4
    assert end >= start, "end must never precede start"
    assert end == start, "rank 7 owns nothing"


def test_sharding_is_total_and_disjoint() -> None:
    """Every vocabulary entry is owned by exactly one rank -- no gaps, no overlap."""
    vocab, tp = 10, 3
    owned: list[int] = []
    for rank in range(tp):
        start, end = vocab_shard_bounds(vocab, tp, rank)
        owned.extend(range(start, end))
    assert owned == list(range(vocab))


@pytest.mark.parametrize(
    "vocab,tp,rank,match",
    [
        (0, 2, 0, "global_vocab_size must be >= 1"),
        (-1, 2, 0, "global_vocab_size must be >= 1"),
        (10, 0, 0, "tp_world_size must be >= 1"),
        (10, 2, 2, "is outside"),
        (10, 2, -1, "is outside"),
    ],
)
def test_bad_bounds_arguments_are_rejected(
    vocab: int, tp: int, rank: int, match: str
) -> None:
    """A zero/negative degree or an out-of-range rank is a config bug, not a slice."""
    with pytest.raises(ValueError, match=match):
        vocab_shard_bounds(vocab, tp, rank)


# -- next_token_targets -------------------------------------------------------


def test_the_last_position_of_every_row_is_ignored() -> None:
    """The shift is within a row: the boundary token predicts nothing.

    Positions ``[2, 3, -100] [5, 6, -100]``: row 0's last slot would predict
    token 4, which belongs to the *next* document, so it is masked rather than
    letting the model train on a prediction it had no context for.
    """
    labels = torch.tensor([1, 2, 3, 4, 5, 6])
    targets = next_token_targets(labels, seq_len=3)
    assert targets.tolist() == [2, 3, IGNORE_INDEX, 5, 6, IGNORE_INDEX]


def test_the_shift_does_not_cross_a_row_boundary() -> None:
    """A flat shift would place 4 at index 3; the row-wise one must not."""
    labels = torch.tensor([1, 2, 3, 4, 5, 6])
    targets = next_token_targets(labels, seq_len=3).view(2, 3)
    assert targets[0].tolist() == [2, 3, IGNORE_INDEX]
    assert targets[1].tolist() == [5, 6, IGNORE_INDEX]
    # Index 3 of the flat stream is the first slot of row 1, which predicts 5 --
    # if the implementation had shifted flat, it would hold 4.
    assert targets[1][0].item() == 5


def test_labels_the_source_tensor_is_not_mutated() -> None:
    """``targets`` is built with ``full_like``; the caller's labels survive."""
    labels = torch.tensor([1, 2, 3, 4])
    before = labels.clone()
    next_token_targets(labels, seq_len=2)
    assert torch.equal(labels, before)


def test_a_ragged_label_count_is_rejected() -> None:
    """5 labels cannot be whole rows of 2 -- silently dropping one hides a data bug."""
    with pytest.raises(ValueError, match="not a whole number of rows"):
        next_token_targets(torch.zeros(5, dtype=torch.long), seq_len=2)


# -- filesystem routing -------------------------------------------------------


def test_a_remote_uri_is_routed_to_fsspec_not_the_local_filesystem() -> None:
    """The branch is on the scheme, so a local path must not pay fsspec's cost."""
    assert filesystem.is_remote("gs://bucket/path") is True
    assert filesystem.is_remote("s3://bucket/path") is True
    assert filesystem.is_remote("/abs/path") is False
    assert filesystem.is_remote("relative/path") is False


def test_local_queries_answer_without_a_remote_backend() -> None:
    """exists/isdir/isfile on a plain path go through os, and must be correct."""
    with tempfile.TemporaryDirectory() as tmp:
        a_file = os.path.join(tmp, "a.txt")
        with open(a_file, "w") as handle:
            handle.write("x")
        assert filesystem.exists(a_file) is True
        assert filesystem.isfile(a_file) is True
        assert filesystem.isdir(a_file) is False
        assert filesystem.isdir(tmp) is True
        assert filesystem.exists(os.path.join(tmp, "missing")) is False


def test_listdir_returns_names_not_paths() -> None:
    """Callers join the results themselves; returning full paths would double up."""
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("b.txt", "a.txt"):
            with open(os.path.join(tmp, name), "w") as handle:
                handle.write("x")
        assert sorted(filesystem.listdir(tmp)) == ["a.txt", "b.txt"]


# -- GarbageCollection --------------------------------------------------------


def test_a_non_positive_collection_frequency_is_rejected() -> None:
    """0 would make the modulo-by-frequency either a ZeroDivision or a no-op."""
    with pytest.raises(ValueError, match="gc_freq must be a positive integer"):
        GarbageCollection(gc_freq=0)
    with pytest.raises(ValueError, match="gc_freq must be a positive integer"):
        GarbageCollection(gc_freq=-1)


def test_a_positive_frequency_disables_the_automatic_collector() -> None:
    """The class owns collection timing, so CPython's own trigger must be off."""
    import gc as _gc

    _gc.enable()
    collector = GarbageCollection(gc_freq=1000)
    try:
        assert collector.gc_freq == 1000
        assert _gc.isenabled() is False, "the collector takes over scheduling"
    finally:
        # Do not leak the disabled state into whichever test runs next.
        _gc.enable()
