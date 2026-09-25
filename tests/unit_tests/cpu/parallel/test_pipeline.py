"""Tests for the pipeline stage split.

``generate_llm_fqn_per_model_part`` is pure arithmetic, so it is tested directly.
``split_model_into_stages`` needs a process group (``PipelineStage`` requires
one); the tests below create a single-rank gloo group, which is enough: with a
size-1 ``pp`` mesh every stage is local, so the module ownership and the
chained forward can be checked without any p2p.
"""

from __future__ import annotations

from tests.caps import require_env

require_env('pipelining', 'flex_attention', 'spmd_types')


import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from hpmesh.models.hf_factory import build_model_config
from hpmesh.models.hf_wrapper import HFTransformerModel
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.parallel.pipeline_parallel.apply import (
    _prepend_first_stage_modules,
    _validate_microbatches,
    apply_pp,
)
from hpmesh.parallel.pipeline_parallel.pipeline import (
    generate_llm_fqn_per_model_part,
    split_model_into_stages,
)
from hpmesh.trainer import ParallelConfig


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


def test_the_output_weight_is_guarded_independently_of_the_input_weight() -> None:
    """The head has its own bound; a small input weight must not shield it.

    ``input_weight`` and ``output_weight`` are separate arguments on the public
    signature, but the sibling test above raises on ``input_weight`` -- so it
    would still pass if this branch never ran. That is the shape of a guard
    that looks covered and is not.
    """
    with pytest.raises(ValueError, match="output_weight"):
        generate_llm_fqn_per_model_part(2, 2, input_weight=1, output_weight=5)


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

    torch.testing.assert_close(chained, reference, rtol=1e-5, atol=1e-6)


def test_the_smallest_fillable_split_works() -> None:
    """``num_stages == num_effective_layers`` is the tightest legal split.

    The guard just below this rejects one stage more, so this is the boundary
    where ``layers_per_stage`` is exactly 1 and the split must still succeed --
    the case a ``>`` vs ``>=`` slip would break.
    """
    # 2 layers + 1 + 1 = 4 effective layers, so 4 stages is exactly fillable.
    stages = generate_llm_fqn_per_model_part(4, 2, input_weight=1, output_weight=1)

    assert len(stages) == 4
    assert all(stage for stage in stages), "no stage may be empty"


# -- apply_pp guards: microbatch validation and the cp/ep refusal --------------


def _dims(*, pp: int = 2, ep: int = 1, world_size: int = 2) -> ParallelDims:
    """Mesh-free ``ParallelDims``: the guards run before any mesh is touched."""
    return ParallelDims(
        dp_replicate=1,
        dp_shard=-1,
        cp=1,
        tp=1,
        pp=pp,
        ep=ep,
        world_size=world_size,
    )


def test_zero_or_negative_microbatches_is_rejected() -> None:
    """A 0 would otherwise surface as a ZeroDivisionError on the divisibility
    check -- torchtitan raises this in its config's ``__post_init__``, which
    hpmesh's config does not do, so ``_validate_microbatches`` owns it."""
    for n in (0, -2):
        with pytest.raises(ValueError, match="num_pp_microbatches"):
            _validate_microbatches(
                _dims(), ParallelConfig(num_pp_microbatches=n), global_batch_size=8
            )


def test_microbatch_divisibility_is_still_enforced() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _validate_microbatches(
            _dims(), ParallelConfig(num_pp_microbatches=3), global_batch_size=8
        )
    # 8 rows over dp=1, 4 microbatches: legal.
    _validate_microbatches(
        _dims(), ParallelConfig(num_pp_microbatches=4), global_batch_size=8
    )


def test_pp_with_ep_is_refused_loudly() -> None:
    """pp+ep is not wired through the pipeline; it must raise at setup, not
    silently drop the EP degree."""
    dims = _dims(pp=2, ep=2, world_size=8)
    cfg = ParallelConfig(pipeline_parallel_size=2, expert_parallel_size=2)
    with pytest.raises(NotImplementedError, match="does not compose"):
        apply_pp(
            nn.Module(),
            parallel_dims=dims,
            cfg=cfg,
            device=torch.device("cpu"),
            global_batch_size=8,
        )


# -- first_stage_module_fqns: co-locating extra modules with stage 0 ----------


def _model_with_vision_encoder() -> HFTransformerModel:
    """The five-part wrapper plus one extra top-level child, as a multimodal
    model would carry it."""
    model = _model()
    model.vision_encoder = nn.Linear(16, 16, bias=False)
    return model


def test_first_stage_modules_are_prepended_to_stage_0() -> None:
    model = _model_with_vision_encoder()
    parts = generate_llm_fqn_per_model_part(2, _NUM_LAYERS)
    stage0_before = list(parts[0])
    rest_before = [list(part) for part in parts[1:]]

    _prepend_first_stage_modules(parts, model, ["vision_encoder"])

    assert parts[0] == ["vision_encoder"] + stage0_before
    assert parts[1:] == rest_before


def test_first_stage_module_order_is_preserved() -> None:
    model = _model_with_vision_encoder()
    model.audio_encoder = nn.Linear(16, 16, bias=False)
    parts = generate_llm_fqn_per_model_part(2, _NUM_LAYERS)

    _prepend_first_stage_modules(parts, model, ["audio_encoder", "vision_encoder"])

    assert parts[0][:2] == ["audio_encoder", "vision_encoder"]


def test_absent_first_stage_modules_are_skipped() -> None:
    """A caller may list modules only some model variants carry."""
    model = _model()  # no vision_encoder
    parts = generate_llm_fqn_per_model_part(2, _NUM_LAYERS)
    expected = [list(part) for part in parts]

    _prepend_first_stage_modules(parts, model, ["vision_encoder"])

    assert parts == expected


def test_first_stage_module_already_owned_by_the_split_is_rejected() -> None:
    """A decoder part would get a live copy on two stages and collide their
    state-dict keys in one checkpoint."""
    model = _model()
    parts = generate_llm_fqn_per_model_part(2, _NUM_LAYERS)

    with pytest.raises(ValueError, match="already assigned"):
        _prepend_first_stage_modules(parts, model, ["norm"])


def test_duplicate_first_stage_module_is_rejected() -> None:
    model = _model_with_vision_encoder()
    parts = generate_llm_fqn_per_model_part(2, _NUM_LAYERS)

    with pytest.raises(ValueError, match="more than once"):
        _prepend_first_stage_modules(
            parts, model, ["vision_encoder", "vision_encoder"]
        )


def test_split_keeps_first_stage_module_fqns_stable(pp_mesh) -> None:
    """Stage 0 holds the real extra module under its unsplit name; every other
    stage blanks it, and no parameter key moves or collides."""
    model = _model_with_vision_encoder()
    unsplit_keys = {k for k, _ in model.named_parameters()}
    module_names = generate_llm_fqn_per_model_part(2, _NUM_LAYERS)
    _prepend_first_stage_modules(module_names, model, ["vision_encoder"])

    _, model_parts = split_model_into_stages(
        model, pp_mesh, "1F1B", torch.device("cpu"), module_names
    )
    first, last = model_parts

    assert not isinstance(first.vision_encoder, nn.Identity)
    assert isinstance(last.vision_encoder, nn.Identity)

    first_keys = {k for k, _ in first.named_parameters()}
    last_keys = {k for k, _ in last.named_parameters()}
    assert any(k.startswith("vision_encoder.") for k in first_keys)
    assert first_keys.isdisjoint(last_keys)
    # FQN stability: every key on either stage is a key of the unsplit model.
    assert first_keys | last_keys <= unsplit_keys


def test_split_without_first_stage_modules_is_unchanged(pp_mesh) -> None:
    """Default behavior: the five-part split owns exactly the same keys as
    before the option existed (the extra child is blanked on both stages)."""
    model = _model_with_vision_encoder()
    module_names = generate_llm_fqn_per_model_part(2, _NUM_LAYERS)

    _, model_parts = split_model_into_stages(
        model, pp_mesh, "1F1B", torch.device("cpu"), module_names
    )

    for part in model_parts:
        assert isinstance(part.vision_encoder, nn.Identity)
        assert not any(
            k.startswith("vision_encoder.") for k, _ in part.named_parameters()
        )
