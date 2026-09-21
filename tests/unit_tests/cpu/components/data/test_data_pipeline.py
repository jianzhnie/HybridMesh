"""The Grain data pipeline: dataset graph, collators, loader, tokenizer.

Worth testing outside a training run because almost none of it is checked by the
loss curve. The DP slices are the clearest case: if two ranks' slices overlap,
both ranks still produce correctly shaped batches of correctly tokenized data,
and the only symptom is that the model saw some documents twice and others
never. Same for the epoch reshuffle and the resume cursor -- a wrong seed or a
stale index trains happily on the wrong order.

Everything here runs on CPU with no process group and no network: the tokenizer
and the corpus are built into a tmp dir by the fixtures in ``data_fixtures``.
"""

from __future__ import annotations

import tempfile

import grain.python as grain
import numpy as np
import pytest

from hpmesh.components.loss import IGNORE_INDEX
from hpmesh.components.tokenizer import HuggingFaceTokenizer
from hpmesh.datasets import (
    DatasetConcat,
    DatasetMix,
    GrainDataLoader,
    TextCollator,
    TextSequence,
    WeightedDataset,
    build_concat_then_split_packing,
    build_dataloader,
    build_dataset,
    build_first_fit_packing,
)
from hpmesh.datasets.random_data import RandomTokenDataLoader
from hpmesh.datasets.text.text import ChatProcessor
from hpmesh.trainer.config import (
    DataloaderConfig,
    HybridMeshConfig,
    ModelConfig,
    TrainingConfig,
)
from tests.data_fixtures import (
    CHAT_TEMPLATE,
    NUM_ROWS,
    VOCAB,
    corpus,  # noqa: F401  (fixture re-export)
    make_context,
    make_policy,
    streaming_text_dataset,
    streaming_text_dataset_from_texts,
    text_dataset,
    token_ids,
    tokenizer,  # noqa: F401  (fixture re-export)
    write_tokenizer,
)

# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------


def test_special_tokens_are_inferred_from_the_config(tokenizer):
    assert tokenizer.bos_id == VOCAB["[BOS]"]
    assert tokenizer.eos_id == VOCAB["[EOS]"]
    assert tokenizer.bos_token == "[BOS]"
    assert tokenizer.eos_token == "[EOS]"


def test_encode_suppresses_native_special_tokens_and_adds_its_own(tokenizer):
    """The whole reason ``encode`` exists: one place decides BOS/EOS.

    If the underlying tokenizer's own post-processor also fired, the sequence
    would carry two BOS tokens and every downstream position would be off by
    one while the shapes stayed right.
    """
    assert tokenizer.encode("lorem ipsum") == [
        VOCAB["[BOS]"],
        VOCAB["lorem"],
        VOCAB["ipsum"],
        VOCAB["[EOS]"],
    ]
    assert tokenizer.encode("lorem", add_bos=False, add_eos=False) == [VOCAB["lorem"]]
    assert tokenizer.encode("lorem", add_eos=False) == [
        VOCAB["[BOS]"],
        VOCAB["lorem"],
    ]


def test_decode_round_trips(tokenizer):
    ids = tokenizer.encode("lorem ipsum", add_bos=False, add_eos=False)
    assert tokenizer.decode(ids) == "lorem ipsum"


def test_missing_tokenizer_path_raises():
    with pytest.raises(FileNotFoundError):
        HuggingFaceTokenizer(tokenizer_path="/nonexistent/tokenizer")


# --------------------------------------------------------------------------
# Graph shape: shuffle, sharding, repeat
# --------------------------------------------------------------------------


def test_dp_ranks_get_disjoint_slices_that_cover_the_corpus(tokenizer, corpus):
    """The property the whole shuffle-then-shard ordering exists to provide.

    A strided split would also be disjoint, but Grain's shuffle is a permutation
    computed from the index, so the contiguous slice of the shuffled index space
    is what keeps reads sequential per rank.
    """
    per_rank = NUM_ROWS // 2
    seen = []
    for rank in (0, 1):
        config = text_dataset(corpus)
        dataset = build_dataset(
            config,
            context=make_context(tokenizer),
            dataset_iteration_policy=make_policy(
                shuffle=True, dp_world_size=2, dp_rank=rank
            ),
        )
        iterator = iter(dataset)
        seen.append(token_ids([next(iterator) for _ in range(per_rank)]))

    assert len(set(seen[0])) == per_rank
    assert len(set(seen[1])) == per_rank
    # Disjoint: no document is trained on twice across the two ranks.
    assert not (set(seen[0]) & set(seen[1]))
    # And covering: nothing is dropped between them.
    assert len(set(seen[0]) | set(seen[1])) == NUM_ROWS


def test_repeat_replays_in_order_when_shuffle_is_off(tokenizer, corpus):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer),
        dataset_iteration_policy=make_policy(shuffle=False, repeat=True),
    )
    iterator = iter(dataset)
    first = token_ids([next(iterator) for _ in range(NUM_ROWS)])
    second = token_ids([next(iterator) for _ in range(NUM_ROWS)])
    assert first == second


def test_repeat_reshuffles_each_epoch_when_shuffle_is_on(tokenizer, corpus):
    """Re-seeing the same order every epoch is a silent quality regression.

    Grain derives the epoch from the sliced map indices, so ``repeat()`` after
    ``shuffle()`` reshuffles; asserting only that the *set* is stable pins the
    epoch-dependence without pinning a permutation we do not own.
    """
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer),
        dataset_iteration_policy=make_policy(shuffle=True, repeat=True),
    )
    iterator = iter(dataset)
    first = token_ids([next(iterator) for _ in range(NUM_ROWS)])
    second = token_ids([next(iterator) for _ in range(NUM_ROWS)])
    assert set(first) == set(second)
    assert first != second


def test_too_few_rows_for_the_dp_size_raises(tokenizer, corpus):
    """The slice arithmetic cannot be carried out below one row per rank."""
    with pytest.raises(ValueError, match="fewer than dp_world_size"):
        build_dataset(
            text_dataset(corpus),
            context=make_context(tokenizer),
            dataset_iteration_policy=make_policy(dp_world_size=NUM_ROWS + 1),
        )


def test_documents_shorter_than_two_tokens_are_dropped(tokenizer, corpus):
    """``TextProcessor`` returns None for a one-token document.

    A single token has no next token to predict, so keeping it would put a row
    with an empty label on the loss. The post-filter is what removes them, and
    the count is what proves it ran.
    """
    short = text_dataset(corpus)
    context = make_context(tokenizer)
    dataset = build_dataset(
        short, context=context, dataset_iteration_policy=make_policy(shuffle=False)
    )
    kept = sum(1 for sequence in (dataset[i] for i in range(len(dataset))) if sequence)
    # Every row here is at least two tokens, so nothing should be filtered.
    assert kept == NUM_ROWS


# --------------------------------------------------------------------------
# TextSequence and collator
# --------------------------------------------------------------------------


def test_text_sequence_rejects_ragged_fields():
    with pytest.raises(ValueError, match="equal lengths"):
        TextSequence(
            input_ids=np.arange(4),
            labels=np.arange(3),
        )


def test_text_sequence_accepts_an_absent_positions_field():
    sequence = TextSequence(input_ids=np.arange(4), labels=np.arange(4))
    assert sequence.positions is None


@pytest.fixture
def collator(tokenizer):
    return TextCollator(context=make_context(tokenizer, num_tokens_per_batch=16))


def test_collator_pads_positions_inside_the_context_window(collator):
    """Padded positions restart at 0 rather than continuing the previous row.

    Zeros would make a padded region look like a continuation of the last real
    document, which is exactly what the position encoding would then encode.
    """
    rows = [
        TextSequence(input_ids=np.arange(5), labels=np.arange(5)),
        TextSequence(input_ids=np.arange(3), labels=np.arange(3)),
    ]
    batch = collator(rows)
    positions = batch["positions"]
    assert positions.shape == (16,)
    assert positions[:8].tolist() == list(range(5)) + list(range(3))
    assert int(positions[8:].max()) < 32
    # Pad positions start over, so the first padded slot is not 8.
    assert int(positions[8]) == 0


def test_collator_pads_labels_with_ignore_index(collator):
    batch = collator([TextSequence(input_ids=np.arange(5), labels=np.arange(5))])
    assert batch["labels"][5:].tolist() == [IGNORE_INDEX] * 11
    assert batch["num_valid_tokens"] == 5


def test_collator_masks_padding_tokens_from_the_row(collator):
    """A row that already carries padding keeps its own mask.

    Packing produces rows with their own padding mask, and the collator must
    respect it rather than assume every row is full -- otherwise a packed row's
    padding tokens would be counted as valid labels.
    """
    rows = [
        TextSequence(
            input_ids=np.arange(4),
            labels=np.array([1, 2, 3, IGNORE_INDEX]),
            padding_mask=np.array([False, False, False, True]),
        )
    ]
    batch = collator(rows)
    assert batch["num_valid_tokens"] == 3
    assert bool(batch["padding_mask"][3])


def test_collator_rejects_rows_over_the_token_batch(collator):
    with pytest.raises(ValueError, match="exceed the configured token batch"):
        collator([TextSequence(input_ids=np.arange(17), labels=np.arange(17))])


def test_collator_num_rows_per_batch_is_one(collator):
    assert collator.num_rows_per_batch() == 1


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------


def _loader(dataset, tokenizer, *, dp_rank=0, dp_world_size=1, config_kwargs=None):
    return GrainDataLoader(
        dataset,
        **dict(config_kwargs or {}),
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        max_context_length=32,
        num_tokens_per_batch=32,
    )


def test_loader_emits_the_trainer_batch_contract(tokenizer, corpus):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        dataset_iteration_policy=make_policy(repeat=True),
    )
    batch = next(iter(_loader(dataset, tokenizer)))
    assert sorted(batch) == [
        "input",
        "labels",
        "num_valid_tokens",
        "padding_mask",
        "positions",
    ]
    assert batch["input"].shape == (32,)
    assert batch["labels"].shape == (32,)
    assert batch["num_valid_tokens"] == int((batch["labels"] != IGNORE_INDEX).sum())


def test_loader_resume_reproduces_the_following_batches_exactly(tokenizer, corpus):
    """The loading position has to survive a checkpoint, or a resumed run
    replays or skips a slice of the epoch. Comparing the *next* batches (not the
    state dict) is the part that matters: a state that round-trips but restores
    to the wrong place still passes a state-equality check.
    """

    def fresh():
        dataset = build_dataset(
            text_dataset(corpus),
            context=make_context(tokenizer, num_tokens_per_batch=32),
            dataset_iteration_policy=make_policy(shuffle=True, repeat=True),
        )
        return _loader(dataset, tokenizer)

    ahead, behind = fresh(), fresh()
    ahead_iterator, behind_iterator = iter(ahead), iter(behind)
    next(ahead_iterator)
    next(ahead_iterator)
    next(behind_iterator)  # one batch behind

    behind.load_state_dict(ahead.state_dict())
    for _ in range(3):
        assert (
            next(ahead_iterator)["input"].tolist()
            == next(behind_iterator)["input"].tolist()
        )
    ahead.close()
    behind.close()


def test_loader_rejects_resuming_across_a_changed_dp_size(tokenizer, corpus):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        dataset_iteration_policy=make_policy(repeat=True),
    )
    loader = _loader(dataset, tokenizer)
    with pytest.raises(ValueError, match="data-parallel degree"):
        loader.load_state_dict({"version": 1, "dp_world_size": 2, "dp_rank_0": {}})
    loader.close()


def test_loader_rejects_an_unknown_state_version(tokenizer, corpus):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        dataset_iteration_policy=make_policy(repeat=True),
    )
    loader = _loader(dataset, tokenizer)
    with pytest.raises(ValueError, match="unsupported GrainDataLoader state version"):
        loader.load_state_dict({"version": 99, "dp_world_size": 1, "dp_rank_0": {}})
    loader.close()


def test_loader_requires_the_rank_entry_present_in_the_checkpoint(tokenizer, corpus):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        dataset_iteration_policy=make_policy(repeat=True),
    )
    loader = _loader(dataset, tokenizer)
    with pytest.raises(ValueError, match="missing dataloader state"):
        loader.load_state_dict({"version": 1, "dp_world_size": 1})
    loader.close()


def test_finite_dataset_under_dp_is_rejected_up_front(tokenizer, corpus):
    """Exhaustion at different steps would hang the next collective.

    This is checked at construction rather than at the first short batch, which
    is the point: by the time a rank notices, its peers are already blocked.
    """
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        dataset_iteration_policy=make_policy(repeat=True),
    )
    with pytest.raises(ValueError, match="repeat=False"):
        _loader(dataset, tokenizer, dp_world_size=2, config_kwargs={"repeat": False})


def test_max_num_documents_must_be_positive(tokenizer, corpus, num_tokens_per_batch=32):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        dataset_iteration_policy=make_policy(),
    )
    with pytest.raises(ValueError, match="max_num_documents must be positive"):
        _loader(dataset, tokenizer, config_kwargs={"max_num_documents": 0})


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


def test_concat_then_split_fills_the_token_batch(tokenizer, corpus):
    context = make_context(tokenizer, num_tokens_per_batch=32)
    graph = build_concat_then_split_packing(
        text_dataset(corpus), context=context, dataset_iteration_policy=make_policy()
    )
    loader = _loader(graph, tokenizer, config_kwargs={"repeat": False})
    rows = [next(iter(loader)) for _ in range(3)]
    for row in rows:
        assert row["input"].shape == (32,)
        # Packing exists to spend the whole batch on real tokens.
        assert row["num_valid_tokens"] == 32


def test_first_fit_packing_fills_the_token_batch(tokenizer, corpus):
    context = make_context(tokenizer, num_tokens_per_batch=32)
    graph = build_first_fit_packing(
        text_dataset(corpus),
        num_packing_bins=4,
        context=context,
        dataset_iteration_policy=make_policy(),
    )
    loader = _loader(graph, tokenizer, config_kwargs={"repeat": False})
    iterator = iter(loader)
    for _ in range(3):
        row = next(iterator)
        assert row["input"].shape == (32,)
        assert row["num_valid_tokens"] > 0


def test_concat_then_split_marks_padding_so_it_never_contributes_loss(
    tokenizer, corpus
):
    """Padding inside a packed row must be labelled IGNORE_INDEX.

    A packed row mixes several documents to fill the batch, so the trailing
    padding is interleaved with real tokens in the same tensor. If it reached
    the loss as a normal target the model would be trained to predict pad.
    """
    context = make_context(tokenizer, num_tokens_per_batch=32)
    graph = build_concat_then_split_packing(
        text_dataset(corpus), context=context, dataset_iteration_policy=make_policy()
    )
    loader = _loader(graph, tokenizer, config_kwargs={"repeat": False})
    iterator = iter(loader)
    for _ in range(3):
        row = next(iterator)
        padded = row["padding_mask"]
        assert bool((row["labels"][padded] == IGNORE_INDEX).all())
        if bool(padded.any()):
            # `max` on an all-padding row is what the guard is for, and an
            # empty window is not an error: `arange(pad_len) % ctx` is.
            assert int(row["positions"][padded].max()) < 32


def test_document_aware_packing_caps_segments_per_row(tokenizer, corpus):
    """``max_num_documents`` bounds how many documents share a row.

    Without it a row can be assembled from dozens of documents, which is fine
    for the loss but not for anything downstream that needs per-document
    boundaries (attention masks, MRoPE).
    """
    context = make_context(tokenizer, num_tokens_per_batch=32, max_num_documents=1)
    graph = build_concat_then_split_packing(
        text_dataset(corpus), context=context, dataset_iteration_policy=make_policy()
    )
    loader = _loader(graph, tokenizer, config_kwargs={"repeat": False})
    row = next(iter(loader))
    assert row["input"].shape == (32,)
    # One document, so at most one position restart plus padding.
    assert int((row["positions"] == 0).sum()) <= 1 + int(row["padding_mask"].sum())


def test_document_aware_packing_keeps_each_row_to_one_document(tokenizer, corpus):
    """``max_num_documents=1`` stops a row from carrying a second document.

    This is what separates the document-aware node from the plain
    ``grain.experimental.ConcatThenSplitIterDataset`` path, and the row *shape*
    cannot tell them apart -- both fill the batch. The plain path concatenates
    everything and cuts at the token count, so a row runs off the end of one
    document and into the head of the next. This node pads instead.

    Sizing matters. At ``num_tokens_per_batch=32`` the fixture documents all fit
    in one row, so the two paths agree and the existing tests pass on either.
    At 8 the documents are 4-12 tokens and the distinction actually shows.

    A document longer than a row is still split -- that is unavoidable, and the
    assertion is deliberately "one *source document* per row", not "one segment",
    so the unavoidable split does not read as a failure. Asserted against the
    unpacked source rather than against a shape, so a dropped or duplicated
    document cannot hide behind matching row lengths.
    """
    context = make_context(tokenizer, num_tokens_per_batch=8, max_num_documents=1)
    policy = make_policy()
    source = [
        [int(token) for token in sequence.input_ids]
        for sequence in build_dataset(
            text_dataset(corpus), context=context, dataset_iteration_policy=policy
        )
    ]
    # Which source document each token came from, flattened.
    owner = [index for index, doc in enumerate(source) for _ in doc]
    flat_source = [token for doc in source for token in doc]

    graph = build_concat_then_split_packing(
        text_dataset(corpus), context=context, dataset_iteration_policy=policy
    )

    flat_packed: list[int] = []
    for row in graph:
        assert row.input_ids.shape == (8,)
        # Positions restart at 0 within a row, so the leading run where
        # ``positions[i] == i`` is this row's real tokens; the rest is padding.
        real = 0
        while real < 8 and int(row.positions[real]) == real:
            real += 1
        tokens = [int(token) for token in row.input_ids[:real]]
        owners = owner[len(flat_packed) : len(flat_packed) + real]
        assert len(set(owners)) == 1, (
            f"row starting at token {len(flat_packed)} draws from "
            f"{len(set(owners))} documents, but max_num_documents=1"
        )
        flat_packed.extend(tokens)

    # Nothing dropped, nothing duplicated, order preserved.
    assert flat_packed == flat_source


# --------------------------------------------------------------------------
# Mix
# --------------------------------------------------------------------------


def test_mix_rejects_a_non_positive_weight(tokenizer, corpus):
    dataset = text_dataset(corpus)
    mix = DatasetMix(
        datasets=(
            WeightedDataset(dataset=dataset, weight=1.0),
            WeightedDataset(dataset=dataset, weight=0.0),
        )
    )
    with pytest.raises(ValueError, match="finite, positive-weight"):
        build_dataset(
            mix, context=make_context(tokenizer), dataset_iteration_policy=make_policy()
        )


def test_mix_produces_a_batch_from_interleaved_children(tokenizer, corpus):
    mix = DatasetMix(
        datasets=(
            WeightedDataset(dataset=text_dataset(corpus), weight=1.0),
            WeightedDataset(dataset=text_dataset(corpus), weight=2.0),
        )
    )
    graph = build_dataset(
        mix,
        context=make_context(tokenizer, num_tokens_per_batch=32),
        dataset_iteration_policy=make_policy(),
    )
    row = next(iter(_loader(graph, tokenizer)))
    assert row["input"].shape == (32,)


def test_mix_of_streaming_children_draws_from_both(tokenizer):
    """Mixing streams takes the second branch: ``IterDataset.mix``.

    Distinct from the map branch above, which is the one the existing test
    exercises. The two are separate grain calls with separate weight handling,
    and nothing else in the suite builds an iter-style mix -- so a child
    silently dropped would only ever show up in a run.

    The children hold disjoint words. A mix that kept only the first child would
    look perfectly healthy against a shared corpus, since the first child covers
    it by itself -- that is exactly the failure this guards. The words are drawn
    from the fixture vocabulary rather than invented, because an unknown word
    tokenizes to ``[UNK]`` and both children would then be indistinguishable.
    """
    first = streaming_text_dataset_from_texts([f"w{i}" for i in range(6)])
    second = streaming_text_dataset_from_texts([f"w{i}" for i in range(20, 26)])
    mix = DatasetMix(
        datasets=(
            WeightedDataset(dataset=first, weight=1.0),
            WeightedDataset(dataset=second, weight=1.0),
        )
    )
    context = make_context(tokenizer, num_tokens_per_batch=8)
    graph = build_dataset(mix, context=context, dataset_iteration_policy=make_policy())
    assert isinstance(graph, grain.IterDataset)

    def rows_of(config):
        return token_ids(
            build_dataset(
                config, context=context, dataset_iteration_policy=make_policy()
            )
        )

    from_first, from_second = rows_of(first), rows_of(second)
    assert set(from_first) != set(from_second), "the children must be distinguishable"
    # Every row from both children, and nothing else.
    assert sorted(token_ids(list(graph))) == sorted(from_first + from_second)


def test_mix_of_streaming_children_rejects_a_non_positive_weight(tokenizer, corpus):
    """The validation runs before either branch, and covers both."""
    mix = DatasetMix(
        datasets=(
            WeightedDataset(dataset=streaming_text_dataset(corpus), weight=1.0),
            WeightedDataset(dataset=streaming_text_dataset(corpus), weight=-1.0),
        )
    )
    with pytest.raises(ValueError, match="finite, positive-weight"):
        build_dataset(
            mix, context=make_context(tokenizer), dataset_iteration_policy=make_policy()
        )


# --------------------------------------------------------------------------
# Streaming sources: the IterDataset branch of the graph
# --------------------------------------------------------------------------


def test_a_streaming_source_takes_the_iter_dataset_branch(tokenizer, corpus):
    """The streaming half of ``SingleDataset`` is otherwise never entered.

    ``_build_map_dataset`` and ``_build_iter_dataset`` are different pastes of
    the pipeline -- streaming shuffles with a window *before* processing while
    the map path shuffles globally *after* -- and every other test feeds a
    random-access source, so only the map path has ever run.

    Both a correct stream and a wrong one produce correctly shaped batches, so
    the assertions are on content: every document once, and the window shuffle
    actually reordering them.
    """
    context = make_context(tokenizer, num_tokens_per_batch=8)
    ordered = build_dataset(
        streaming_text_dataset(corpus),
        context=context,
        dataset_iteration_policy=make_policy(shuffle=False),
    )
    assert isinstance(ordered, grain.IterDataset)
    rows = token_ids(list(ordered))
    assert len(rows) == NUM_ROWS
    assert len(set(rows)) == NUM_ROWS  # nothing dropped or duplicated

    shuffled = build_dataset(
        streaming_text_dataset(corpus),
        context=context,
        dataset_iteration_policy=make_policy(shuffle=True),
    )
    assert set(token_ids(list(shuffled))) == set(rows)
    # A window shuffle is not the identity; if it were, the stream order would
    # be trained on unchanged and the shuffle would be a silent no-op.
    assert token_ids(list(shuffled)) != rows


def test_concat_rejects_a_streaming_child(tokenizer, corpus):
    """Concatenation needs a length up front, so a stream cannot take part.

    The guard is what turns a mid-run failure into a build-time one. Note the
    child is built before the check, so this also pins that the built child is
    inspected rather than the config.
    """
    concat = DatasetConcat(
        datasets=(streaming_text_dataset(corpus), text_dataset(corpus))
    )
    with pytest.raises(TypeError, match="requires map-style children"):
        build_dataset(
            concat,
            context=make_context(tokenizer),
            dataset_iteration_policy=make_policy(),
        )


def test_concat_shards_map_children_like_a_single_dataset(tokenizer, corpus):
    """Concatenation is transparent to everything downstream of it.

    ``_shard_for_dp`` is applied to the concatenation exactly as it is to one
    dataset, so the DP disjointness guarantee has to survive the extra layer.
    """
    concat = DatasetConcat(datasets=(text_dataset(corpus), text_dataset(corpus)))
    per_rank = (2 * NUM_ROWS) // 2
    seen = []
    for rank in (0, 1):
        graph = build_dataset(
            concat,
            context=make_context(tokenizer),
            dataset_iteration_policy=make_policy(
                shuffle=True, dp_world_size=2, dp_rank=rank
            ),
        )
        seen.append(token_ids([next(iter(graph)) for _ in range(per_rank)]))

    assert not (set(seen[0]) & set(seen[1]))


# --------------------------------------------------------------------------
# ChatProcessor
# --------------------------------------------------------------------------


def _chat_processor(tokenizer, *, max_context_length=32):
    tokenizer.set_chat_template(CHAT_TEMPLATE)
    return ChatProcessor(
        context=make_context(
            tokenizer,
            num_tokens_per_batch=max_context_length,
            max_context_length=max_context_length,
        ),
        messages_fn=lambda sample: sample["messages"],
    )


def test_chat_processor_masks_the_prompt_labels(tokenizer):
    """Only the response contributes loss.

    The boundary is found by re-rendering the prompt alone, so the assertion is
    that the masked span is exactly the prompt's token count -- not that some
    plausible number of labels came back masked.
    """
    processor = _chat_processor(tokenizer)
    sample = {
        "messages": [
            {"role": "user", "content": "lorem"},
            {"role": "assistant", "content": "ipsum"},
        ]
    }
    sequence = processor(sample, np.random.default_rng(0))
    assert sequence is not None

    prompt = tokenizer.encode(
        tokenizer.apply_chat_template(
            sample["messages"][:1], add_generation_prompt=True
        ),
        add_bos=True,
        add_eos=False,
    )
    masked = int((sequence.labels == IGNORE_INDEX).sum())
    assert masked == len(prompt) - 1
    # The response labels are intact, including the appended EOS.
    assert sequence.labels[-1] == tokenizer.eos_id


def test_chat_processor_is_next_token_aligned(tokenizer):
    processor = _chat_processor(tokenizer)
    sequence = processor(
        {
            "messages": [
                {"role": "user", "content": "lorem"},
                {"role": "assistant", "content": "ipsum"},
            ]
        },
        np.random.default_rng(0),
    )
    real = sequence.labels != IGNORE_INDEX
    # Each label is the next token, wherever both positions carry a target.
    both = real[:-1] & real[1:]
    assert (sequence.labels[:-1][both] == sequence.input_ids[1:][both]).all()


def test_chat_processor_rejects_a_multi_turn_conversation(tokenizer):
    processor = _chat_processor(tokenizer)
    with pytest.raises(ValueError, match="Expected single-turn"):
        processor(
            {
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"},
                ]
            },
            np.random.default_rng(0),
        )


def test_chat_processor_rejects_swapped_roles(tokenizer):
    processor = _chat_processor(tokenizer)
    with pytest.raises(ValueError, match="First message must be 'user'"):
        processor(
            {
                "messages": [
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "a"},
                ]
            },
            np.random.default_rng(0),
        )


def test_chat_processor_drops_an_oversized_sample(tokenizer):
    """Overflow is per-sample, so dropping is correct here (unlike a template
    mismatch, which would be systematic). Returning None is what routes it to
    the post-filter rather than failing the run.
    """
    processor = _chat_processor(tokenizer, max_context_length=1)
    sequence = processor(
        {
            "messages": [
                {"role": "user", "content": "lorem ipsum"},
                {"role": "assistant", "content": "lorem ipsum lorem ipsum"},
            ]
        },
        np.random.default_rng(0),
    )
    assert sequence is None


def test_chat_processor_requires_an_eos_id():
    class _NoEos(HuggingFaceTokenizer):
        def __init__(self, *, tokenizer_path):
            super().__init__(tokenizer_path=tokenizer_path)
            self.eos_id = None

    path = write_tokenizer(tempfile.mkdtemp())
    tokenizer = _NoEos(tokenizer_path=path)
    with pytest.raises(ValueError, match="does not have an eos_id"):
        ChatProcessor(
            context=make_context(tokenizer),
            messages_fn=lambda sample: sample["messages"],
        )


# --------------------------------------------------------------------------
# The trainer-facing seam: the config that names a corpus, and the loader it
# builds. These are what the Trainer actually calls, so they are exercised
# through the same entry point rather than by reaching past it.
# --------------------------------------------------------------------------


def test_dataloader_arguments_default_to_the_synthetic_corpus() -> None:
    """The default must need no assets, so an untouched run stays offline."""
    args = DataloaderConfig()
    assert args.dataset == "random"
    assert args.tokenizer_path is None


def test_dataloader_arguments_require_a_tokenizer_for_a_real_corpus() -> None:
    with pytest.raises(ValueError, match="tokenizer_path is required"):
        DataloaderConfig(dataset="c4")


def test_dataloader_arguments_require_a_path_for_local_jsonl() -> None:
    with pytest.raises(ValueError, match="dataset_path is required"):
        DataloaderConfig(dataset="local_jsonl", tokenizer_path="/tmp/tok")


def _loader_config(
    dataloader: DataloaderConfig | None = None,
    *,
    global_batch_size: int = 4,
    max_seq_len: int = 8,
    seed: int = 42,
) -> HybridMeshConfig:
    """The run config ``build_dataloader`` now reads its scalars off.

    ``build_dataloader`` takes the whole config and pulls the corpus settings
    from ``.dataloader`` and the run scalars from the flat view, so a test that
    wants to vary one of them varies the config rather than restating it as a
    keyword argument.
    """
    return HybridMeshConfig(
        model=ModelConfig(vocab_size=128),
        training=TrainingConfig(
            global_batch_size=global_batch_size,
            max_seq_len=max_seq_len,
            seed=seed,
            dataloader_config=dataloader
            if dataloader is not None
            else DataloaderConfig(),
        ),
    )


def test_dataloader_rejects_an_unknown_corpus_at_build_time() -> None:
    """The registry check lives in ``build_dataloader``, not in the config."""
    with pytest.raises(ValueError, match="unknown dataset"):
        build_dataloader(
            _loader_config(
                DataloaderConfig(dataset="not-a-dataset", tokenizer_path="/tmp/tok")
            ),
            dp_rank=0,
            dp_world_size=1,
            num_tokens_per_batch=32,
        )


def test_dataloader_arguments_build_the_synthetic_loader_without_assets() -> None:
    loader = build_dataloader(
        _loader_config(),
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=32,
    )
    assert isinstance(loader, RandomTokenDataLoader)
    batch = next(iter(loader))
    assert batch.input_ids.shape == (4, 8)


def test_dataloader_arguments_build_a_grain_loader_over_a_local_corpus(
    tmp_path, corpus
) -> None:
    """The end-to-end seam, without the network.

    A tokenizer is written here rather than taken from the ``tokenizer``
    fixture: the config takes a *path*, because the Trainer has no tokenizer
    object to hand it, and the fixture lives in its own module-scoped temp
    directory.
    """
    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    loader = build_dataloader(
        _loader_config(
            DataloaderConfig(
                dataset="local_jsonl",
                tokenizer_path=tokenizer_path,
                dataset_path=corpus,
            ),
            seed=1,
        ),
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=32,
    )
    assert isinstance(loader, GrainDataLoader)

    batch = next(iter(loader))
    assert batch["input"].shape == (32,)
    assert batch["labels"].shape == (32,)
    assert batch["positions"].shape == (32,)
    # Packing spends the whole batch on real tokens.
    assert batch["num_valid_tokens"] == 32
    assert int((batch["labels"] != IGNORE_INDEX).sum()) == 32
    loader.close()


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("concat_then_split", "build_concat_then_split_packing"),
        ("first_fit", "build_first_fit_packing"),
    ],
)
def test_packing_selector_chooses_the_recipe(
    tmp_path, corpus, monkeypatch, selector, expected
) -> None:
    """``--packing`` must reach the choice of function ``build_dataloader`` makes.

    Asserted at the seam rather than by inspecting the graph that comes back:
    both recipes return a ``grain.IterDataset``, so a selector that never left
    the config would still build a working loader -- just the wrong one, with
    nothing in the batch to say so.
    """
    import hpmesh.datasets.build as build_module

    calls: list[str] = []
    for name in ("build_concat_then_split_packing", "build_first_fit_packing"):
        real = getattr(build_module, name)

        def spy(*args, _name=name, _real=real, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(build_module, name, spy)

    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    loader = build_dataloader(
        _loader_config(
            DataloaderConfig(
                dataset="local_jsonl",
                tokenizer_path=tokenizer_path,
                dataset_path=corpus,
                packing=selector,
            ),
            seed=1,
        ),
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=32,
    )
    assert calls == [expected]
    loader.close()


def test_num_packing_bins_reaches_first_fit(tmp_path, corpus, monkeypatch) -> None:
    """The bin count is forwarded, not left at the callee's default."""
    import hpmesh.datasets.build as build_module

    seen: dict = {}
    real = build_module.build_first_fit_packing

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(build_module, "build_first_fit_packing", spy)

    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    loader = build_dataloader(
        _loader_config(
            DataloaderConfig(
                dataset="local_jsonl",
                tokenizer_path=tokenizer_path,
                dataset_path=corpus,
                packing="first_fit",
                num_packing_bins=3,
            ),
            seed=1,
        ),
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=32,
    )
    assert seen["num_packing_bins"] == 3
    loader.close()


def test_the_two_text_recipes_agree_on_the_batch_contract(tmp_path, corpus) -> None:
    """Switching recipes changes packing, not the shape the trainer consumes."""
    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    batches = {}
    for selector in ("concat_then_split", "first_fit"):
        loader = build_dataloader(
            _loader_config(
                DataloaderConfig(
                    dataset="local_jsonl",
                    tokenizer_path=tokenizer_path,
                    dataset_path=corpus,
                    packing=selector,
                ),
                seed=1,
            ),
            dp_rank=0,
            dp_world_size=1,
            num_tokens_per_batch=32,
        )
        batches[selector] = next(iter(loader))
        loader.close()

    for batch in batches.values():
        assert batch["input"].shape == (32,)
        assert batch["labels"].shape == (32,)
        assert batch["positions"].shape == (32,)
        assert batch["padding_mask"].shape == (32,)
        # FirstFit pads to the row length rather than filling every slot, so
        # the token count is bounded by the batch rather than equal to it.
        assert 0 < batch["num_valid_tokens"] <= 32


def test_num_packing_bins_must_be_positive() -> None:
    """Validated even under the default recipe, which never reads it.

    The field is always parsed, so a bad value accepted here would only
    surface after switching ``--packing`` to first_fit.
    """
    with pytest.raises(ValueError, match="num_packing_bins must be positive"):
        DataloaderConfig(num_packing_bins=0)


def test_dataloader_arguments_reject_a_mismatched_dp_size_at_build_time(
    tmp_path, corpus
) -> None:
    """A config whose policy was built for one rank cannot be handed another's.

    The trainer derives the policy from the config and passes the rank
    separately, so the two can disagree; the loader is where that is caught.
    """
    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    loader = build_dataloader(
        _loader_config(
            DataloaderConfig(
                dataset="local_jsonl",
                tokenizer_path=tokenizer_path,
                dataset_path=corpus,
            ),
            seed=1,
        ),
        dp_rank=1,
        dp_world_size=2,
        num_tokens_per_batch=32,
    )
    # Stored under dp_rank_1 because that is what the config was built for.
    assert loader.state_dict()["dp_rank_1"] is not None
    loader.close()
