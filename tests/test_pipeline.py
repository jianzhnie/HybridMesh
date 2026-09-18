"""Tests for the pipeline stage split.

``generate_llm_fqn_per_model_part`` is pure arithmetic, so it is tested directly.
``split_model_into_stages`` needs a process group (``PipelineStage`` requires
one), so it is exercised only where a distributed backend is available.
"""

from __future__ import annotations

import pytest

from hpmesh.parallel.pipeline import generate_llm_fqn_per_model_part


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
