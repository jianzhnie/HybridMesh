"""Tests for the pipeline stage split.

``generate_llm_fqn_per_model_part`` is pure arithmetic, so it is tested directly.
``split_model_into_stages`` needs a process group (``PipelineStage`` requires
one); the tests below create a single-rank gloo group, which is enough: with a
size-1 ``pp`` mesh every stage is local, so the module ownership and the
chained forward can be checked without any p2p.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config
from hpmesh.parallel.pipeline_parallel.pipeline import (
    generate_llm_fqn_per_model_part,
    split_model_into_stages,
)


def test_single_stage_owns_everything() -> None:
    parts = generate_llm_fqn_per_model_part(1, 3)

    assert parts == [
        [
            "tok_embeddings",
            "layers.0",
            "layers.1",
            "layers.2",
            "norm",
            "lm_head",
            "rotary_emb",
        ]
    ]


def test_layers_are_partitioned_in_order_without_gaps() -> None:
    parts = generate_llm_fqn_per_model_part(4, 10)

    layer_names = [
        name for part in parts for name in part if name.startswith("layers.")
    ]
    assert layer_names == [f"layers.{i}" for i in range(10)]


def test_embedding_lands_on_the_first_stage_and_head_on_the_last() -> None:
    parts = generate_llm_fqn_per_model_part(3, 6)

    assert "tok_embeddings" in parts[0]
    assert "lm_head" in parts[-1]
    assert "norm" in parts[-1]
    # Neither appears anywhere else.
    for part in parts[:-1]:
        assert "lm_head" not in part
        assert "norm" not in part
    for part in parts[1:]:
        assert "tok_embeddings" not in part


def test_rotary_emb_is_replicated_on_every_stage() -> None:
    """Each stage runs its own layers, so each needs the rope buffers."""
    parts = generate_llm_fqn_per_model_part(3, 6)

    assert all("rotary_emb" in part for part in parts)


def test_input_and_output_weights_shift_the_boundary() -> None:
    """A heavier embedding should pull fewer transformer layers onto stage 0."""
    light = generate_llm_fqn_per_model_part(2, 8, input_weight=1, output_weight=1)
    heavy = generate_llm_fqn_per_model_part(2, 8, input_weight=3, output_weight=1)

    def count_layers(part: list[str]) -> int:
        return sum(1 for name in part if name.startswith("layers."))

    assert count_layers(heavy[0]) < count_layers(light[0])


def test_looped_schedule_split_gives_every_virtual_stage_a_layer() -> None:
    """The 4-virtual-stage split the Interleaved1F1B equivalence check runs on.

    A looped schedule over pp=2 defaults to two stages per rank (4 stages), so
    the model must be deep enough that no stage is left holding only the
    embedding or only the head.
    """
    parts = generate_llm_fqn_per_model_part(4, 6)

    assert parts == [
        ["tok_embeddings", "layers.0", "rotary_emb"],
        ["layers.1", "layers.2", "rotary_emb"],
        ["layers.3", "layers.4", "rotary_emb"],
        ["layers.5", "norm", "lm_head", "rotary_emb"],
    ]


def test_zero_stages_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        generate_llm_fqn_per_model_part(0, 4)


def test_more_stages_than_effective_layers_is_rejected() -> None:
    # 2 layers + 1 + 1 = 4 effective layers, so 5 stages cannot be filled.
    with pytest.raises(ValueError, match="cannot be greater than effective"):
        generate_llm_fqn_per_model_part(5, 2)


def test_weight_larger_than_a_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="input_weight"):
        generate_llm_fqn_per_model_part(2, 2, input_weight=5, output_weight=1)


# -- split_model_into_stages, over a single-rank gloo group --------------------

_NUM_LAYERS = 4


@pytest.fixture(scope="module")
def pp_mesh(tmp_path_factory):
    """A size-1 ``pp`` mesh: every stage lands on this rank, no p2p needed."""
    created = not dist.is_initialized()
    if created:
        store = dist.FileStore(str(tmp_path_factory.mktemp("pg") / "store"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    yield init_device_mesh("cpu", (1,), mesh_dim_names=("pp",))
    if created:
        dist.destroy_process_group()


def _model() -> HFTransformerModel:
    torch.manual_seed(0)
    config = build_model_config(
        "llama",
        seq_len=32,
        arch_overrides={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": _NUM_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )
    return HFTransformerModel(config).eval()


def _split(pp_mesh):
    """Both stages of a 2-stage split, local on this rank."""
    model = _model()
    module_names = generate_llm_fqn_per_model_part(2, _NUM_LAYERS)
    stages, model_parts = split_model_into_stages(
        model, pp_mesh, "1F1B", torch.device("cpu"), module_names
    )
    return model, stages, model_parts


def test_split_gives_each_stage_only_its_own_modules(pp_mesh) -> None:
    _, stages, model_parts = _split(pp_mesh)
    first, last = model_parts

    assert len(stages) == 2 and len(model_parts) == 2

    # First stage: real embedding and its layers; head/norm blanked.
    # (4 layers + input_weight 1 + output_weight 1 = 6 effective, 3 per stage:
    # the first stage's 3 are the embedding + 2 layers, the last's are 2
    # layers + norm/head.)
    assert not isinstance(first.tok_embeddings, nn.Identity)
    assert len(first.layers) == 2
    assert isinstance(first.norm, nn.Identity)
    assert isinstance(first.lm_head, nn.Identity)

    # Last stage: real norm and head; embedding blanked.
    assert isinstance(last.tok_embeddings, nn.Identity)
    assert len(last.layers) == 2
    assert not isinstance(last.norm, nn.Identity)
    assert not isinstance(last.lm_head, nn.Identity)

    # RoPE is replicated: both stages run layers, so both keep the buffers.
    assert not isinstance(first.rotary_emb, nn.Identity)
    assert not isinstance(last.rotary_emb, nn.Identity)

    # No parameter lives on both stages, and the kept layers keep their
    # ORIGINAL indices -- a renumbered ModuleList would collide the two
    # stages' keys in one shared checkpoint namespace.
    first_keys = {k for k, _ in first.named_parameters()}
    last_keys = {k for k, _ in last.named_parameters()}
    assert first_keys.isdisjoint(last_keys)
    assert any("layers.1." in k for k in first_keys)
    assert any("layers.2." in k for k in last_keys)


def test_split_stages_chain_to_the_full_forward(pp_mesh) -> None:
    """first(ids) then last(hidden) must reproduce the unsplit model's logits.

    This is the property pipeline execution relies on: the stage boundary
    carries hidden states, and a non-first stage's forward consumes them
    (``tok_embeddings`` is an Identity there, so the wrapper skips the
    embedding lookup).
    """
    model, _, (first, last) = _split(pp_mesh)

    ids = torch.randint(32, (16,))
    with torch.no_grad():
        reference = model(ids)
        chained = last(first(ids))

    assert torch.allclose(chained, reference, atol=1e-6)
