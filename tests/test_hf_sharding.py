"""Tests for the HF sharding-config declarations.

Mirrors the torchtitan experiment's ``test_hf_sharding.py``.
``set_hf_sharding_configs`` only annotates modules -- nothing reads the configs
until a later ``model.parallelize()`` -- so it is testable on CPU without a
process group.
"""

from __future__ import annotations

import torch.nn as nn

from hpmesh.parallel.hf_sharding import (
    _set_dsa_indexer_sharding,
    set_hf_sharding_configs,
)


class _AttentionWithIndexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.indexer = nn.Sequential(nn.Linear(4, 4))


def test_dsa_indexer_rejects_tensor_parallelism() -> None:
    attention = _AttentionWithIndexer()

    try:
        _set_dsa_indexer_sharding(attention, enable_sp=True)
    except NotImplementedError as exc:
        assert "tensor parallelism" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError under TP")


def test_dsa_indexer_is_replicated_without_tp() -> None:
    attention = _AttentionWithIndexer()
    _set_dsa_indexer_sharding(attention, enable_sp=False)

    for module in attention.indexer.modules():
        assert getattr(module, "_sharding_config", None) is not None


class _TinyHFBlock(nn.Module):
    """The minimal HF decoder-layer shape ``_set_layer_sharding_configs`` reads."""

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp.down_proj = nn.Linear(hidden, hidden, bias=False)


class _TinyHFModel(nn.Module):
    def __init__(self, hidden: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(16, hidden)
        self.layers = nn.ModuleList([_TinyHFBlock(hidden) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, 16, bias=False)


def _assert_all_configured(root: nn.Module) -> None:
    """Every param/buffer-bearing module must carry a sharding config."""
    for name, module in root.named_modules():
        has_state = (
            next(module.parameters(recurse=False), None) is not None
            or next(module.buffers(recurse=False), None) is not None
        )
        if has_state:
            assert getattr(module, "_sharding_config", None) is not None, name


def test_set_hf_sharding_configs_covers_every_module() -> None:
    model = _TinyHFModel()
    set_hf_sharding_configs(model, enable_sp=False)

    _assert_all_configured(model)
    # Attention gets the flex kernel carrying the local SPMD region.
    for layer in model.layers:
        assert getattr(layer.self_attn, "_titan_flex_kernel", None) is not None


def test_set_hf_sharding_configs_is_idempotent() -> None:
    model = _TinyHFModel()
    set_hf_sharding_configs(model, enable_sp=True)
    before = {
        name: id(module._sharding_config)
        for name, module in model.named_modules()
        if getattr(module, "_sharding_config", None) is not None
    }

    set_hf_sharding_configs(model, enable_sp=True)
    after = {
        name: id(module._sharding_config)
        for name, module in model.named_modules()
        if getattr(module, "_sharding_config", None) is not None
    }

    assert before.keys() == after.keys()


def test_unconfigured_layer_module_fails_loud() -> None:
    """The per-layer backstop names any state-bearing module left unconfigured.

    The backstop walks each decoder layer's subtree (it does not audit the
    top-level model), so the stowaway sits inside a layer -- which is where a
    new architecture variant would actually add one.
    """
    model = _TinyHFModel()
    model.layers[0].mystery = nn.Linear(8, 8)

    try:
        set_hf_sharding_configs(model, enable_sp=False)
    except ValueError as exc:
        assert "mystery" in str(exc)
    else:
        raise AssertionError("expected ValueError naming the unsharded module")
