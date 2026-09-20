# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared fixtures for the data-layer tests.

Not a conftest: these are plain helpers the test modules import by name, and
pytest puts this directory on ``sys.path`` for exactly that. Keeping them here
rather than in one test module means ``test_multimodal_data`` does not have to
import ``test_data_pipeline`` to reach them.
"""

from __future__ import annotations

import json
import os

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from hpmesh.components.tokenizer import HuggingFaceTokenizer
from hpmesh.datasets import (
    DatasetBuildContext,
    DatasetIterationPolicy,
    IndexedJsonlSource,
    SingleDatasetConfig,
)
from hpmesh.datasets.hf.text import TextProcessor

# A whitespace WordLevel vocabulary: small enough to reason about by hand, and
# whitespace-based rather than BPE so a document's tokens are predictable.
VOCAB: dict[str, int] = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
    "lorem": 4,
    "ipsum": 5,
}
VOCAB.update({f"w{i}": 6 + i for i in range(64)})

NUM_ROWS = 40

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>{{ message['content'] }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def write_tokenizer(dirpath: str, *, extra_vocab: dict[str, int] | None = None) -> str:
    """Write a self-contained tokenizer to ``dirpath`` and return the path.

    Built from scratch rather than downloaded so no test needs the Hub.

    ``extra_vocab`` entries are registered as *added* tokens, not merely as
    vocabulary entries. ``MultiModalTokenizer`` validates its five vision
    placeholders against ``get_added_tokens_decoder()``, so a token that is only
    in the base vocabulary is invisible to it.
    """
    from tokenizers import AddedToken

    os.makedirs(dirpath, exist_ok=True)
    extras = dict(extra_vocab or {})
    vocab = dict(VOCAB)
    vocab.update(extras)

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if extras:
        tokenizer.add_tokens(
            [AddedToken(content=token, special=True) for token in extras]
        )
        added = {
            token.content: token_id
            for token_id, token in tokenizer.get_added_tokens_decoder().items()
        }
        for token, token_id in extras.items():
            assert added.get(token) == token_id, (
                f"tokenizer assigned {token!r} id {added.get(token)}, "
                f"expected {token_id}"
            )
    tokenizer.save(os.path.join(dirpath, "tokenizer.json"))

    with open(os.path.join(dirpath, "tokenizer_config.json"), "w") as handle:
        json.dump(
            {
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "add_bos_token": True,
                "add_eos_token": True,
            },
            handle,
        )
    return dirpath


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory) -> HuggingFaceTokenizer:
    path = write_tokenizer(str(tmp_path_factory.mktemp("tokenizer")))
    return HuggingFaceTokenizer(tokenizer_path=path)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> str:
    """A JSONL file whose every row tokenizes to a distinct token sequence.

    Distinctness matters: the DP-disjointness test compares sets of documents,
    and a vocabulary that collapsed two rows to the same `[UNK]` run would make
    that comparison pass for the wrong reason.
    """
    path = str(tmp_path_factory.mktemp("corpus") / "rows.jsonl")
    with open(path, "w") as handle:
        for i in range(NUM_ROWS):
            handle.write(
                json.dumps({"text": f"w{i} " + "lorem ipsum " * (i % 5 + 1)}) + "\n"
            )
    return path


def make_policy(**overrides) -> DatasetIterationPolicy:
    fields = dict(
        seed=7,
        shuffle=False,
        repeat=False,
        dp_rank=0,
        dp_world_size=1,
        streaming_shuffle_buffer_size=100,
    )
    fields.update(overrides)
    return DatasetIterationPolicy(**fields)


def make_context(
    tokenizer,
    *,
    num_tokens_per_batch=64,
    max_context_length=32,
    max_num_documents=None,
):
    import grain.python as grain

    return DatasetBuildContext(
        tokenizer=tokenizer,
        max_context_length=max_context_length,
        num_tokens_per_batch=num_tokens_per_batch,
        read_options=grain.ReadOptions(),
        max_num_documents=max_num_documents,
    )


def text_dataset(corpus, *, processor=TextProcessor) -> SingleDatasetConfig:
    return SingleDatasetConfig(
        source=IndexedJsonlSource(patterns=(corpus,)),
        processor=processor,
        post_filters=(lambda sample: sample is not None,),
    )


def token_ids(sequences) -> list[tuple[int, ...]]:
    return [tuple(sequence.input_ids.tolist()) for sequence in sequences]
