"""Dataloader config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(kw_only=True)
class DataloaderConfig:
    """Where the micro-batches come from.

    ``random`` (the default) keeps the synthetic corpus and needs no assets, so
    the default run is unchanged and reproducible offline. Any other value
    names a recipe from ``datasets.text.text.DATASETS`` or
    ``datasets.multimodal.mm_datasets.MM_DATASETS``, or the built-in
    ``local_jsonl`` --
    which is deliberately NOT in either dict, because its corpus path is a
    runtime argument rather than a constant.

    Every non-``random`` dataset builds its graph on Grain, which needs a
    tokenizer, so ``tokenizer_path`` is required there and unused otherwise.
    Multimodal recipes additionally need the optional dependencies
    torchvision/Pillow (and av for video) and the five ``mm_*_token`` strings
    below, which must exist as added tokens in that tokenizer.
    """

    dataset: str = field(
        default="random",
        metadata={
            "help": "Corpus selector: 'random' (synthetic, no assets) | "
            "'local_jsonl' | a key of datasets.text.text.DATASETS | a key of "
            "datasets.multimodal.mm_datasets.MM_DATASETS (needs torchvision)"
        },
    )
    tokenizer_path: str | None = field(
        default=None,
        metadata={
            "help": "Directory holding the tokenizer. Required unless --dataset random."
        },
    )
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Corpus path. Required for --dataset local_jsonl."},
    )
    prompt_field: str = field(
        default="prompt",
        metadata={"help": "Prompt field used by dataset=local_jsonl_sft."},
    )
    response_field: str = field(
        default="response",
        metadata={"help": "Assistant field used by dataset=local_jsonl_sft."},
    )
    chat_renderer: str | None = field(
        default=None,
        metadata={
            "help": "Multi-turn SFT via the optional renderers package: name "
            "of its renderer config class (e.g. 'Qwen3RendererConfig'). Only "
            "valid with dataset=local_jsonl_sft, whose rows must then carry a "
            "messages list; replaces the single-turn chat-template path. "
            "Needs `pip install renderers==0.1.11`."
        },
    )
    messages_field: str = field(
        default="messages",
        metadata={
            "help": "Row field holding the multi-turn conversation. Used only "
            "when chat_renderer is set."
        },
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Globally shuffle before sharding across DP ranks"},
    )
    streaming_shuffle_buffer_size: int = field(
        default=1_000,
        metadata={"help": "Streaming rows retained per rank for approximate shuffle"},
    )
    num_prefetch_batches: int = field(
        default=2,
        metadata={"help": "Collated batches queued per rank for the trainer"},
    )
    packing: Literal["concat_then_split", "first_fit"] = field(
        default="concat_then_split",
        metadata={
            "help": "Text packing recipe. 'concat_then_split' concatenates "
            "documents and chunks them into fixed-length rows; 'first_fit' "
            "lays documents of up to the context window into bins instead, "
            "which sustains single-document rows. Ignored for multimodal "
            "recipes, which pack whole documents regardless."
        },
    )
    num_packing_bins: int = field(
        default=8,
        metadata={
            "help": "Candidate rows 'first_fit' keeps open. More bins can "
            "reduce padding, but buffer more documents. Ignored otherwise."
        },
    )
    max_num_documents: int | None = field(
        default=None,
        metadata={
            "help": "Cap on documents packed into one row. None leaves the "
            "frontier unconstrained."
        },
    )
    mm_image_token: str = field(
        default="<|image_pad|>",
        metadata={"help": "Image placeholder token. Multimodal recipes only."},
    )
    mm_video_token: str = field(
        default="<|video_pad|>",
        metadata={"help": "Video placeholder token. Multimodal recipes only."},
    )
    mm_vision_start_token: str = field(
        default="<|vision_start|>",
        metadata={
            "help": "Token opening a vision placeholder run. Multimodal recipes only."
        },
    )
    mm_vision_end_token: str = field(
        default="<|vision_end|>",
        metadata={
            "help": "Token closing a vision placeholder run. Multimodal recipes only."
        },
    )
    mm_pad_token: str = field(
        default="<|endoftext|>",
        metadata={"help": "Padding token. Multimodal recipes only."},
    )

    def __post_init__(self) -> None:
        if self.dataset != "random" and not self.tokenizer_path:
            raise ValueError(
                f"tokenizer_path is required for dataset '{self.dataset}'. "
                "Only 'random' runs without a tokenizer."
            )
        if self.dataset in {"local_jsonl", "local_jsonl_sft"} and not self.dataset_path:
            raise ValueError(f"dataset_path is required for dataset {self.dataset!r}")
        if self.dataset == "local_jsonl_sft" and (
            not self.prompt_field.strip() or not self.response_field.strip()
        ):
            raise ValueError("local_jsonl_sft field names cannot be empty")
        if self.chat_renderer is not None:
            if self.dataset != "local_jsonl_sft":
                raise ValueError(
                    f"chat_renderer requires dataset='local_jsonl_sft', got "
                    f"{self.dataset!r}"
                )
            if not self.messages_field.strip():
                raise ValueError(
                    "messages_field cannot be empty when chat_renderer is set"
                )
        # Membership in ``datasets.text.text.DATASETS`` /
        # ``datasets.multimodal.mm_datasets.MM_DATASETS`` is checked by
        # ``datasets/build.py`` at build time, not here: reading the registries
        # would import the datasets package into the config layer.
        if self.max_num_documents is not None and self.max_num_documents <= 0:
            raise ValueError("max_num_documents must be positive")
        # Validated here even though only 'first_fit' reads it: the field is
        # always parsed, so a bad value would otherwise be accepted silently
        # under the default recipe and only fail after switching to first_fit.
        if self.num_packing_bins <= 0:
            raise ValueError("num_packing_bins must be positive")
