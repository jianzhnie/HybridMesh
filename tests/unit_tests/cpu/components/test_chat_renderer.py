"""Optional renderers-library path: gating, adapter, and processor semantics.

The ``renderers`` package is not installed in this environment, so the
library seam is exercised against a fake module injected into
``sys.modules`` -- the fake reproduces the interface hpmesh calls
(``create_renderer``, ``build_training_sample``, top-level config classes
with a ``name``), not the real renderers' token-level output. Numerical
parity with the real library is unverified until the dependency lands and
these tests are re-run against it; the gating test (no package installed)
runs for real.
"""

from __future__ import annotations

from tests.caps import require_env

require_env('grain')


import sys
import tempfile
import types

import numpy as np
import pytest

from hpmesh.components.loss import IGNORE_INDEX
from hpmesh.components.tokenizer import HuggingFaceTokenizer
from hpmesh.config import (
    DataloaderConfig,
    HybridMeshConfig,
    ModelConfig,
    TrainingConfig,
)
from hpmesh.datasets.build import build_dataloader
from hpmesh.datasets.text.renderer import (
    RENDERERS_INSTALL_HINT,
    RendererTokenizerWrapper,
    build_chat_renderer,
)
from hpmesh.datasets.text.text import ChatProcessor
from tests.data_fixtures import (
    make_context,
    tokenizer,  # noqa: F401  (fixture re-export)
    write_tokenizer,
)


class _FakeRendererConfig:
    name = "qwen3"


class _FakeAutoRendererConfig:
    name = "auto"


class _FakeDefaultRendererConfig:
    name = "default"


class _FakeRendered:
    def __init__(self, token_ids, loss_mask, multi_modal_data=None):
        self.token_ids = token_ids
        self.loss_mask = loss_mask
        self.multi_modal_data = multi_modal_data


def _fake_build_training_sample(renderer, messages, ensure_final_stop=True):
    """Assistant content tokens supervised, everything else masked."""
    token_ids: list[int] = []
    loss_mask: list[bool] = []
    for message in messages:
        marker = renderer.tokenizer.encode(f"<|{message['role']}|>")
        content = renderer.tokenizer.encode(message["content"])
        token_ids += marker + content
        loss_mask += [False] * len(marker)
        loss_mask += [message["role"] == "assistant"] * len(content)
    if ensure_final_stop and renderer.tokenizer.eos_token_id is not None:
        token_ids.append(renderer.tokenizer.eos_token_id)
        loss_mask.append(True)
    return _FakeRendered(token_ids, loss_mask)


class _FakeRenderer:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config


@pytest.fixture
def fake_renderers(monkeypatch):
    module = types.ModuleType("renderers")
    module.Qwen3RendererConfig = _FakeRendererConfig
    module.AutoRendererConfig = _FakeAutoRendererConfig
    module.DefaultRendererConfig = _FakeDefaultRendererConfig
    module.create_renderer = lambda *, tokenizer, config: _FakeRenderer(
        tokenizer, config
    )
    module.build_training_sample = _fake_build_training_sample
    monkeypatch.setitem(sys.modules, "renderers", module)
    return module


# --------------------------------------------------------------------------
# gating
# --------------------------------------------------------------------------


def test_build_chat_renderer_without_the_package_raises_an_install_hint(
    tokenizer, monkeypatch
):
    """Real environment check: ``renderers`` is not installed here."""
    monkeypatch.delitem(sys.modules, "renderers", raising=False)
    with pytest.raises(ImportError, match="renderers==0.1.11"):
        build_chat_renderer(
            tokenizer=tokenizer, renderer_name="Qwen3RendererConfig"
        )
    assert "pip install" in RENDERERS_INSTALL_HINT


def test_build_chat_renderer_rejects_an_unknown_name(tokenizer, fake_renderers):
    with pytest.raises(ValueError, match="no top-level config named"):
        build_chat_renderer(tokenizer=tokenizer, renderer_name="NotARenderer")


@pytest.mark.parametrize("name", ["AutoRendererConfig", "DefaultRendererConfig"])
def test_build_chat_renderer_refuses_auto_and_default(
    tokenizer, fake_renderers, name
):
    with pytest.raises(ValueError, match="Pick the model's renderer"):
        build_chat_renderer(tokenizer=tokenizer, renderer_name=name)


def test_build_chat_renderer_returns_a_library_renderer(tokenizer, fake_renderers):
    renderer = build_chat_renderer(
        tokenizer=tokenizer, renderer_name="Qwen3RendererConfig"
    )
    assert isinstance(renderer, _FakeRenderer)
    assert isinstance(renderer.tokenizer, RendererTokenizerWrapper)


# --------------------------------------------------------------------------
# RendererTokenizerWrapper
# --------------------------------------------------------------------------


def test_wrapper_exposes_the_hf_style_surface(tokenizer):
    wrapper = RendererTokenizerWrapper(tokenizer)
    assert wrapper.name_or_path == tokenizer.tokenizer_path
    assert wrapper.bos_token_id == tokenizer.bos_id
    assert wrapper.eos_token_id == tokenizer.eos_id
    assert wrapper.unk_token_id is None
    assert wrapper.encode("lorem") == [4]
    assert wrapper.decode([4, 5]) == "lorem ipsum"
    assert wrapper.convert_tokens_to_ids("lorem") == 4
    assert wrapper.convert_tokens_to_ids(["lorem", "ipsum"]) == [4, 5]
    out = wrapper("lorem ipsum", add_special_tokens=False, return_offsets_mapping=True)
    assert out["input_ids"] == [4, 5]
    assert out["offset_mapping"] == [(0, 5), (6, 11)]


# --------------------------------------------------------------------------
# ChatProcessor on the renderer path
# --------------------------------------------------------------------------

MULTI_TURN = [
    {"role": "user", "content": "lorem"},
    {"role": "assistant", "content": "ipsum"},
    {"role": "user", "content": "ipsum"},
    {"role": "assistant", "content": "lorem ipsum"},
]


def _renderer_processor(tokenizer, fake_renderers, *, max_context_length=32):
    return ChatProcessor(
        context=make_context(
            tokenizer,
            num_tokens_per_batch=max_context_length,
            max_context_length=max_context_length,
        ),
        messages_fn=lambda sample: sample["messages"],
        renderer=build_chat_renderer(
            tokenizer=tokenizer, renderer_name="Qwen3RendererConfig"
        ),
    )


def test_renderer_processor_supervises_every_assistant_turn(
    tokenizer, fake_renderers
):
    """The mask follows the rendered history, not a single prompt prefix."""
    processor = _renderer_processor(tokenizer, fake_renderers)
    sequence = processor({"messages": MULTI_TURN}, np.random.default_rng(0))
    assert sequence is not None

    rendered = _fake_build_training_sample(
        processor._renderer, MULTI_TURN
    )
    np.testing.assert_array_equal(
        sequence.input_ids, np.asarray(rendered.token_ids[:-1])
    )
    expected_labels = np.asarray(rendered.token_ids[1:])
    expected_labels[~np.asarray(rendered.loss_mask[1:], dtype=bool)] = IGNORE_INDEX
    np.testing.assert_array_equal(sequence.labels, expected_labels)
    # Both assistant turns contribute loss, the user turns do not.
    supervised = sequence.labels != IGNORE_INDEX
    assert supervised.sum() == 1 + 2 + 1  # "ipsum" + "lorem ipsum" + final stop


def test_renderer_processor_is_next_token_aligned(tokenizer, fake_renderers):
    processor = _renderer_processor(tokenizer, fake_renderers)
    sequence = processor({"messages": MULTI_TURN}, np.random.default_rng(0))
    real = sequence.labels != IGNORE_INDEX
    both = real[:-1] & real[1:]
    assert (sequence.labels[:-1][both] == sequence.input_ids[1:][both]).all()


def test_renderer_processor_does_not_require_an_eos_id(fake_renderers):
    """The renderer owns the terminal stop token, so EOS inference is moot."""

    class _NoEos(HuggingFaceTokenizer):
        def __init__(self, *, tokenizer_path):
            super().__init__(tokenizer_path=tokenizer_path)
            self.eos_id = None

    path = write_tokenizer(tempfile.mkdtemp())
    tokenizer = _NoEos(tokenizer_path=path)
    processor = _renderer_processor(tokenizer, fake_renderers)
    assert processor({"messages": MULTI_TURN}, np.random.default_rng(0)) is not None


def test_renderer_processor_rejects_a_non_assistant_final_turn(
    tokenizer, fake_renderers
):
    processor = _renderer_processor(tokenizer, fake_renderers)
    with pytest.raises(ValueError, match="end with an assistant message"):
        processor({"messages": MULTI_TURN[:-1]}, np.random.default_rng(0))


def test_renderer_processor_rejects_multimodal_content(
    tokenizer, fake_renderers, monkeypatch
):
    def multimodal(renderer, messages, ensure_final_stop=True):
        rendered = _fake_build_training_sample(renderer, messages)
        rendered.multi_modal_data = {"image": object()}
        return rendered

    monkeypatch.setattr(
        sys.modules["renderers"], "build_training_sample", multimodal
    )
    processor = _renderer_processor(tokenizer, fake_renderers)
    with pytest.raises(ValueError, match="text-only"):
        processor({"messages": MULTI_TURN}, np.random.default_rng(0))


def test_renderer_processor_drops_an_oversized_sample(tokenizer, fake_renderers):
    processor = _renderer_processor(
        tokenizer, fake_renderers, max_context_length=1
    )
    assert (
        processor({"messages": MULTI_TURN}, np.random.default_rng(0)) is None
    )


# --------------------------------------------------------------------------
# config validation and the build_dataloader seam
# --------------------------------------------------------------------------


def test_chat_renderer_defaults_off() -> None:
    config = DataloaderConfig()
    assert config.chat_renderer is None
    assert config.messages_field == "messages"


def test_chat_renderer_requires_the_sft_dataset() -> None:
    with pytest.raises(ValueError, match="chat_renderer requires"):
        DataloaderConfig(dataset="local_jsonl", tokenizer_path="/tmp/tok",
                         dataset_path="/tmp/rows.jsonl",
                         chat_renderer="Qwen3RendererConfig")


def test_chat_renderer_rejects_an_empty_messages_field() -> None:
    with pytest.raises(ValueError, match="messages_field cannot be empty"):
        DataloaderConfig(
            dataset="local_jsonl_sft",
            tokenizer_path="/tmp/tok",
            dataset_path="/tmp/rows.jsonl",
            chat_renderer="Qwen3RendererConfig",
            messages_field=" ",
        )


def test_make_local_jsonl_sft_multiturn_builds_a_renderer_processor(
    tokenizer, fake_renderers, tmp_path
):
    """The recipe build_dataloader wires when chat_renderer is set.

    Iterating the built graph needs real Grain; the wiring under test is the
    recipe itself -- processor class, configured messages field, and the
    renderer it was handed.
    """
    from hpmesh.datasets.text.text import make_local_jsonl_sft_multiturn

    corpus_path = tmp_path / "rows.jsonl"
    corpus_path.write_text('{"conversation": []}\n')
    renderer = build_chat_renderer(
        tokenizer=tokenizer, renderer_name="Qwen3RendererConfig"
    )
    recipe = make_local_jsonl_sft_multiturn(
        path=str(corpus_path), messages_field="conversation", renderer=renderer
    )
    processor = recipe.processor(context=make_context(tokenizer))
    assert processor._renderer is renderer
    assert processor._messages_fn({"conversation": MULTI_TURN}) == MULTI_TURN
    with pytest.raises(KeyError, match="messages field"):
        processor._messages_fn({"messages": MULTI_TURN})
    with pytest.raises(TypeError, match="list of message dicts"):
        processor._messages_fn({"conversation": "not a list"})


def test_build_dataloader_without_the_package_fails_loudly(tmp_path, monkeypatch):
    """Enabling chat_renderer with no renderers installed is a build-time
    ImportError, not a silent fall back to the template path."""
    monkeypatch.delitem(sys.modules, "renderers", raising=False)
    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    corpus_path = tmp_path / "rows.jsonl"
    corpus_path.write_text('{"messages": []}\n')
    with pytest.raises(ImportError, match="renderers==0.1.11"):
        build_dataloader(
            HybridMeshConfig(
                model=ModelConfig(vocab_size=128),
                training=TrainingConfig(
                    global_batch_size=4,
                    max_seq_len=32,
                    seed=1,
                    dataloader_config=DataloaderConfig(
                        dataset="local_jsonl_sft",
                        tokenizer_path=tokenizer_path,
                        dataset_path=str(corpus_path),
                        chat_renderer="Qwen3RendererConfig",
                    ),
                ),
            ),
            dp_rank=0,
            dp_world_size=1,
            num_tokens_per_batch=32,
        )
