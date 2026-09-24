"""Text dataset recipes: plain-text and single-turn chat.

Vendored from torchtitan ``hf_datasets/text_datasets.py``. A processor is
constructed with its build context and takes the few per-dataset knobs
(``text_fn``, ``messages_fn``) as keyword arguments.

``ChatProcessor`` is the one place in the data layer that can fail on a whole
dataset rather than a sample. Locating the prompt/response boundary by
re-rendering the prompt assumes the chat template emits it as a textual prefix
of the full render, which is a property of the template and the tokenizer
together. When that assumption breaks, every sample breaks with it, so this
raises instead of dropping -- dropping would silently train on a fraction of
the data while the loss curve looked healthy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ...components.loss import IGNORE_INDEX
from ...utils.logger_utils import get_logger
from ..dataset import SampleProcessor, SingleDataset, TextSequence
from ..sources import (
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
    IndexedJsonlSource,
)
from ..types import DatasetBuildContext

logger = get_logger(__name__)

__all__ = [
    "ChatProcessor",
    "DATASETS",
    "TextProcessor",
    "make_local_jsonl",
    "make_local_jsonl_sft",
]


def _read_text(sample: dict[str, Any]) -> str:
    return sample["text"]


class TextProcessor(SampleProcessor):
    """Tokenizes plain text into next-token input and label pairs."""

    def __init__(
        self,
        *,
        context: DatasetBuildContext,
        text_fn: Callable[[dict[str, Any]], str] = _read_text,
    ) -> None:
        self._tokenizer = context.tokenizer
        self._text_fn = text_fn

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        del rng
        input_ids = np.asarray(
            self._tokenizer.encode(self._text_fn(sample), add_bos=True, add_eos=True),
            dtype=np.int64,
        )
        if len(input_ids) < 2:
            return None
        return TextSequence(
            input_ids=input_ids[:-1],
            labels=input_ids[1:],
        )


def _require_token_prefix(full_tokens: list[int], prompt_tokens: list[int]) -> None:
    """Raise if prompt_tokens is not an exact prefix of full_tokens.

    ChatProcessor locates the prompt/response boundary by re-rendering the
    prompt alone and requiring it to tokenize to a prefix of the full
    conversation. That holds only when rendering the prompt with
    ``add_generation_prompt=True`` produces a textual prefix of the full render
    and the tokenizer does not merge characters across that seam. Both are
    properties of the template and tokenizer together rather than of an
    individual sample.

    Raise instead of dropping the sample: a mismatch means the label boundary
    is unknown, and because the cause is systematic it would fire for most
    samples, so dropping would silently train on a fraction of the dataset.
    The overflow path drops because an oversized example really is per-sample.
    """
    if full_tokens[: len(prompt_tokens)] != prompt_tokens:
        raise ValueError(
            "Prompt tokens are not an exact prefix of the full conversation "
            "tokens, so the prompt/response boundary cannot be located. "
            "ChatProcessor requires that rendering the prompt with "
            "add_generation_prompt=True yields a textual prefix of the full "
            "render, and that the tokenizer does not merge characters across "
            "that seam. A template that rewrites earlier turns when later ones "
            "are present, or turn separators that only merge in context, break "
            "this assumption."
        )


class ChatProcessor(SampleProcessor):
    """Tokenizes one single-turn chat sample and masks prompt labels."""

    def __init__(
        self,
        *,
        context: DatasetBuildContext,
        messages_fn: Callable[[dict[str, Any]], list[dict[str, str]]],
    ) -> None:
        if context.tokenizer.eos_id is None:
            raise ValueError(
                "Tokenizer does not have an eos_id set. "
                "ChatProcessor requires a tokenizer with a valid EOS token."
            )
        self._tokenizer = context.tokenizer
        self._eos_id = context.tokenizer.eos_id
        self._max_context_length = context.max_context_length
        self._messages_fn = messages_fn
        self._logged_first_sample = False

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> None:
        """Validate that messages are a single-turn [user, assistant] pair."""
        # TODO(data-sft-multiturn): Multi-turn needs per-turn spans that survive
        # templates which rewrite earlier turns, so prefix re-rendering is not
        # enough. See the RFC in #3304 and the implementation in #2769.
        if len(messages) != 2:
            raise ValueError(
                f"Expected single-turn [user, assistant], got {len(messages)} messages"
            )
        if messages[0]["role"] != "user":
            raise ValueError(
                f"First message must be 'user', got '{messages[0]['role']}'"
            )
        if messages[1]["role"] != "assistant":
            raise ValueError(
                f"Second message must be 'assistant', got '{messages[1]['role']}'"
            )

    def _tokenize_sample(self, sample: dict[str, Any]) -> TextSequence | None:
        """Tokenize a single-turn sample and mask prompt labels.

        Returns None if the sample exceeds `seq_len`, avoiding
        training on truncated responses.

        Uses incremental prefix re-tokenization to find the prompt/response
        token boundary, avoiding BPE merge errors.
        """
        messages = self._messages_fn(sample)
        self._validate_messages(messages)

        # The tokenizer defaults add_generation_prompt=True (torchtitan parity);
        # a full-conversation render must not grow a trailing generation prompt.
        full_text = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=False
        )
        # Strip extra newline and ensure the sequence ends with EOS without duplicates
        full_text = full_text.rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._eos_id:
            full_tokens.append(self._eos_id)

        if not self._logged_first_sample:
            logger.info(f"[ChatProcessor] First sample full:\n{full_text}")
            self._logged_first_sample = True

        # TODO(data-sft-overflow): Consider truncating oversized examples instead.
        # Causal loss remains valid for the retained response prefix.
        # Drop oversized examples rather than truncating.
        if len(full_tokens) - 1 > self._max_context_length:
            logger.debug(
                "Dropping sample: token count exceeds "
                f"max_context_length={self._max_context_length}"
            )
            return None

        # Find prompt/response boundary by tokenizing just the user message
        # with add_generation_prompt=True.
        prompt_text = self._tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True
        )
        prompt_tokens = self._tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        _require_token_prefix(full_tokens, prompt_tokens)
        prompt_len = len(prompt_tokens)

        tokens = np.asarray(full_tokens, dtype=np.int64)
        input_ids = tokens[:-1]
        labels = tokens[1:].copy()
        labels[: max(prompt_len - 1, 0)] = IGNORE_INDEX
        return TextSequence(
            input_ids=input_ids,
            labels=labels,
        )

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        del rng
        return self._tokenize_sample(sample)


def make_local_jsonl(*, path: str) -> SingleDataset:
    """Build the ``local_jsonl`` recipe over a caller-supplied corpus.

    A function rather than an entry in :data:`DATASETS`: the path is a runtime
    argument, so it cannot live in a module-level dict without a global.
    """
    return SingleDataset(
        source=IndexedJsonlSource(patterns=(path,)),
        processor=TextProcessor,
        post_filters=(lambda sample: sample is not None,),
    )


def make_local_jsonl_sft(
    *, path: str, prompt_field: str, response_field: str
) -> SingleDataset:
    """Build a single-turn supervised-chat recipe from a local JSONL file."""

    class _LocalJsonlChatProcessor(ChatProcessor):
        def __init__(self, *, context: DatasetBuildContext) -> None:
            def messages(sample: dict[str, Any]) -> list[dict[str, str]]:
                try:
                    prompt = sample[prompt_field]
                    response = sample[response_field]
                except KeyError as exc:
                    raise KeyError(
                        f"local_jsonl_sft row lacks configured field {exc.args[0]!r}"
                    ) from exc
                if not isinstance(prompt, str) or not isinstance(response, str):
                    raise TypeError(
                        "local_jsonl_sft prompt and response fields must be strings"
                    )
                return [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]

            super().__init__(context=context, messages_fn=messages)

    return SingleDataset(
        source=IndexedJsonlSource(patterns=(path,)),
        processor=_LocalJsonlChatProcessor,
        post_filters=(lambda sample: sample is not None,),
    )


DATASETS: dict[str, SingleDataset] = {
    "c4": SingleDataset(
        source=HuggingFaceStreamingSource(
            path="allenai/c4",
            name="en",
            split="train",
        ),
        processor=TextProcessor,
        post_filters=(lambda sample: sample is not None,),
    ),
    "c4_test": SingleDataset(
        source=HuggingFaceRandomAccessSource(
            path="json",
            split="train",
            load_dataset_kwargs={
                "data_files": "tests/assets/c4_test/data.json",
            },
        ),
        processor=TextProcessor,
        post_filters=(lambda sample: sample is not None,),
    ),
    "c4_validation": SingleDataset(
        source=HuggingFaceStreamingSource(
            path="allenai/c4",
            name="en",
            split="validation",
        ),
        processor=TextProcessor,
        post_filters=(lambda sample: sample is not None,),
    ),
}
