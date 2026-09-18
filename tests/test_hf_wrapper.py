"""Tests for the thin HF wrapper.

The wrapper's job is plumbing: expose the decoder's parts under stable names, add
the batch dim HF expects, feed RoPE explicit ``position_ids``, and pass the
attention mask down untouched. Flex attention cannot run on CPU (its inductor
lowering is CUDA-only), so these tests install a probe attention function that
records what the wrapper handed down and then runs SDPA.
"""

from __future__ import annotations

import pytest
import torch
from transformers.modeling_utils import AttentionInterface

from hpmesh.models.hf_wrapper import (
    _ATTN_IMPLEMENTATION,
    HFTransformerModel,
    build_model_config,
)

_HIDDEN = 32
_VOCAB = 128


@pytest.fixture
def model() -> HFTransformerModel:
    config = build_model_config(
        "qwen3",
        seq_len=64,
        arch_overrides={
            "vocab_size": _VOCAB,
            "hidden_size": _HIDDEN,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    )
    return HFTransformerModel(config).eval()


@pytest.fixture
def probe_attention():
    """Swap in a recording attention impl, restoring the registry afterwards."""
    seen: dict = {}

    def probe(module, query, key, value, attention_mask, **kwargs):
        seen["mask"] = attention_mask
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(module, query, key, value, None, **kwargs)

    original = AttentionInterface._global_mapping.get(_ATTN_IMPLEMENTATION)
    AttentionInterface._global_mapping[_ATTN_IMPLEMENTATION] = probe
    try:
        yield seen
    finally:
        if original is None:
            AttentionInterface._global_mapping.pop(_ATTN_IMPLEMENTATION, None)
        else:
            AttentionInterface._global_mapping[_ATTN_IMPLEMENTATION] = original


def test_parts_are_exposed_under_stable_names(model: HFTransformerModel) -> None:
    assert model.tok_embeddings is not None
    assert len(model.layers) == 2
    assert model.norm is not None
    assert model.lm_head is not None
    assert model.rotary_emb is not None


def test_named_children_flattens_the_decoder(model: HFTransformerModel) -> None:
    names = [name for name, _ in model.named_children()]
    assert names == ["tok_embeddings", "layers", "norm", "lm_head", "rotary_emb"]


def test_state_dict_does_not_duplicate_tensors(model: HFTransformerModel) -> None:
    """The decoder is aliased for convenience; it must not be registered twice."""
    keys = list(model.state_dict())
    assert not any(key.startswith("_decoder.") for key in keys)
    # Every parameter appears exactly once.
    assert len(keys) == len(set(keys))


def test_attention_masks_is_a_block_mask(model: HFTransformerModel) -> None:
    positions = torch.arange(8)
    mask = model.get_attention_masks(positions=positions)
    assert type(mask).__name__ == "BlockMask"


def test_forward_applies_lm_head_and_gradients_flow(
    model: HFTransformerModel, probe_attention
) -> None:
    input_ids = torch.randint(0, _VOCAB, (16,))
    logits = model(input_ids, positions=torch.arange(16))

    assert logits.shape == (16, _VOCAB)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_positions_drive_rope(model: HFTransformerModel, probe_attention) -> None:
    """Shifting every position must change the output -- otherwise RoPE is idling."""
    input_ids = torch.randint(0, _VOCAB, (16,))

    with torch.no_grad():
        base = model(input_ids, positions=torch.arange(16))
        shifted = model(input_ids, positions=torch.arange(16) + 3)
        repeated = model(input_ids, positions=torch.arange(16))

    assert not torch.allclose(base, shifted)
    assert torch.equal(base, repeated)


def test_attention_mask_is_passed_down_unchanged(
    model: HFTransformerModel, probe_attention
) -> None:
    input_ids = torch.randint(0, _VOCAB, (16,))
    sentinel = model.get_attention_masks(positions=torch.arange(16))

    model(input_ids, positions=torch.arange(16), attention_masks=sentinel)

    assert probe_attention["mask"] is sentinel
