"""The FSDP contract: what the sharding layer reads off a model, and how.

``fsdp.py`` is written against torchtitan's ``Decoder`` shape -- a ``ModuleDict``
of layers plus four named modules on the top-level object. hpmesh trains HF
models, whose layout differs on both counts. These tests pin the two places that
reconcile them, since a break in either is invisible until either:

* the model silently fails to shard (an adapter that stops taking effect), or
* FSDP raises partway through a distributed run (a name that stopped resolving).

No process group is needed: the layer iterator is pure container logic, and the
accessors are plain attribute lookups on a tiny offline model.
"""

from __future__ import annotations

import pytest
import torch.nn as nn
from torch.nn import ModuleDict, ModuleList

from hpmesh.bundle import HFModelWrapper
from hpmesh.parallel.fsdp import iter_transformer_layers


class _DecoderLike(nn.Module):
    """A HF-shaped decoder: the parts nested one level below the CausalLM."""

    def __init__(self, vocab_size: int, hidden: int, num_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden)


class _CausalLMLike(nn.Module):
    """Stands in for ``AutoModelForCausalLM``: ``model`` plus a sibling ``lm_head``.

    Mirrors the real nesting -- the head is a child of the CausalLM, not of the
    decoder, which is why the wrapper has to reach up a level for it.
    """

    def __init__(self, vocab_size: int, hidden: int, num_layers: int, tied: bool):
        super().__init__()
        self.model = _DecoderLike(vocab_size, hidden, num_layers)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight


def _wrapper(*, tied: bool = False) -> HFModelWrapper:
    return HFModelWrapper(
        _CausalLMLike(vocab_size=32, hidden=8, num_layers=3, tied=tied)
    )


# -- container layout --------------------------------------------------------


def test_iter_transformer_layers_accepts_a_hf_module_list() -> None:
    layers = ModuleList([nn.Linear(4, 4) for _ in range(3)])

    assert [idx for idx, _ in iter_transformer_layers(layers)] == [0, 1, 2]


def test_iter_transformer_layers_accepts_a_torchtitan_module_dict() -> None:
    layers = ModuleDict({str(i): nn.Linear(4, 4) for i in range(3)})

    assert [idx for idx, _ in iter_transformer_layers(layers)] == ["0", "1", "2"]


def test_iter_transformer_layers_yields_blocks_in_order() -> None:
    layers = ModuleList([nn.Linear(4, 4) for _ in range(3)])
    as_dict = ModuleDict({str(i): m for i, m in enumerate(layers)})

    from_list = [block for _, block in iter_transformer_layers(layers)]
    from_dict = [block for _, block in iter_transformer_layers(as_dict)]

    assert all(a is b for a, b in zip(from_list, from_dict, strict=True))


# -- accessors ---------------------------------------------------------------


def test_decoder_parts_resolve_off_the_top_level_wrapper() -> None:
    """The FSDP layer reads these off the model it is handed, not off model.model."""
    wrapper = _wrapper()
    decoder = wrapper.model.model

    assert wrapper.tok_embeddings is decoder.embed_tokens
    assert wrapper.layers is decoder.layers
    assert wrapper.norm is decoder.norm
    assert wrapper.lm_head is wrapper.model.lm_head


def test_layers_are_a_container_torchtitan_can_iterate() -> None:
    wrapper = _wrapper()

    assert len(list(iter_transformer_layers(wrapper.layers))) == 3


def test_untied_head_reports_no_weight_tying() -> None:
    assert _wrapper(tied=False).enable_weight_tying is False


def test_tied_head_is_detected_by_parameter_identity() -> None:
    """FSDP2's own check: the two modules own one Parameter object."""
    wrapper = _wrapper(tied=True)

    assert wrapper.enable_weight_tying is True
    assert wrapper.tok_embeddings.weight is wrapper.lm_head.weight


def test_missing_head_reports_no_weight_tying() -> None:
    """A model whose head was stripped must not be grouped with the embedding."""
    wrapper = _wrapper(tied=True)
    wrapper.model.lm_head = None

    assert wrapper.enable_weight_tying is False


def test_tie_word_embeddings_flag_is_not_what_decides() -> None:
    """Intent and reality can disagree; identity is what FSDP acts on."""
    wrapper = _wrapper(tied=True)
    wrapper.model.config = type("C", (), {"tie_word_embeddings": False})()

    assert wrapper.enable_weight_tying is True


@pytest.mark.parametrize("tied", [False, True])
def test_accessors_do_not_add_duplicate_parameters(tied: bool) -> None:
    """The accessors must alias, not re-register, the underlying modules.

    Assigning ``self.x = module`` inside the wrapper would register a second
    copy, putting the same tensors in the state dict twice -- once under the
    HF name and once under the torchtitan name. These are properties, so the
    parameter count must equal the HF model's own.
    """
    wrapper = _wrapper(tied=tied)

    assert len(list(wrapper.parameters())) == len(list(wrapper.model.parameters()))
