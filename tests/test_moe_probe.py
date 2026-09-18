"""Tests for the HF MoE probe.

Each HF family spells the same MoE differently, so these tests pin the spellings
that actually vary: where the block lives (``mlp`` vs Mixtral's
``block_sparse_moe``), how many experts and how top-k is named, whether a shared
expert exists, and whether routing normalizes. A probe that silently reports "no
MoE" is the failure mode worth guarding -- everything downstream keys off it.
"""

from __future__ import annotations

import pytest
from transformers import AutoConfig, AutoModelForCausalLM

from hpmesh.models.moe_probe import (
    detect_moe_layers,
    probe_moe_layer,
    probe_moe_model,
)

_VOCAB = 256
_SEQ = 64


def _build(architecture: str, **overrides):
    defaults = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": _VOCAB,
        "max_position_embeddings": _SEQ,
    }
    config = AutoConfig.for_model(architecture, **{**defaults, **overrides})
    return config, AutoModelForCausalLM.from_config(config)


@pytest.fixture(scope="module")
def qwen3_moe():
    config, model = _build(
        "qwen3_moe",
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
    )
    detect_moe_layers(model)
    return config, model


@pytest.fixture(scope="module")
def mixtral():
    config, model = _build(
        "mixtral",
        num_local_experts=8,
        num_experts_per_tok=2,
    )
    detect_moe_layers(model)
    return config, model


def test_dense_model_has_no_moe_layers() -> None:
    config, model = _build("llama")
    detect_moe_layers(model)

    assert not any(layer.moe_enabled for layer in model.model.layers)
    assert probe_moe_model(model, config) == {}


def test_probe_reads_the_standard_hf_moe_block(qwen3_moe) -> None:
    config, model = qwen3_moe
    arch = probe_moe_layer(model.model.layers[0], config)

    assert arch is not None
    assert arch.num_experts == 8
    assert arch.top_k == 2
    assert arch.moe_intermediate_size == 32
    assert arch.dim == 64
    assert arch.score_func == "softmax"
    assert arch.shared_expert is None
    # No group-limited routing on this family.
    assert arch.num_expert_groups is None


def test_probe_finds_mixtral_block_sparse_moe(mixtral) -> None:
    """Mixtral names its block ``block_sparse_moe``, not ``mlp``."""
    config, model = mixtral
    arch = probe_moe_layer(model.model.layers[0], config)

    assert arch is not None
    assert arch.num_experts == 8
    assert arch.moe_intermediate_size == 128
    # Mixtral renormalizes the top-k weights but exposes no ``norm_topk_prob``,
    # so a softmax router must be inferred as normalizing.
    assert arch.route_norm is True


def test_probe_detects_sigmoid_gated_shared_expert() -> None:
    config, model = _build(
        "qwen2_moe",
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
    )
    detect_moe_layers(model)
    arch = probe_moe_layer(model.model.layers[0], config)

    assert arch is not None
    assert arch.shared_expert is not None
    assert arch.shared_expert.hidden_dim == 64
    assert arch.shared_expert.has_sigmoid_gate is True


def test_dense_layer_in_a_mixed_model_probes_none() -> None:
    config, model = _build(
        "qwen3_moe",
        num_hidden_layers=3,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[1],
    )
    detect_moe_layers(model)

    assert [layer.moe_enabled for layer in model.model.layers] == [True, False, True]
    assert probe_moe_layer(model.model.layers[1], config) is None


def test_probe_moe_model_returns_only_moe_layers() -> None:
    config, model = _build(
        "qwen3_moe",
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[0],
    )
    detect_moe_layers(model)
    found = probe_moe_model(model, config)

    assert list(found) == [1]
