"""Unit tests for the parts that run without a GPU / process group.

Covers the behaviors verified during prototyping: the world_size constraint,
deterministic synthetic data, DP batch slicing, and the model wrapper. The config
is the grouped HybridMeshConfig; the flat view (cfg.dp, cfg.steps, ...) is what
the trainer/mesh layer reads.
"""

from __future__ import annotations

import pytest
import torch

from hpmesh.components.loss import IGNORE_INDEX, next_token_targets
from hpmesh.mesh import build_parallel_dims
from hpmesh.models.hf_wrapper import (
    _ATTN_IMPLEMENTATION,
    HFTransformerModel,
    build_model_config,
    build_model_config_for,
)
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.trainer import HybridMeshConfig, ParallelConfig, TrainingConfig
from hpmesh.trainer.trainer import Trainer


def _cfg(**parallel_kw) -> HybridMeshConfig:
    return HybridMeshConfig(parallel=ParallelConfig(**parallel_kw))


def test_derive_dp_derives_from_world_size() -> None:
    cfg = _cfg(data_parallel_shard_degree=-1)
    assert cfg.derive_dp(world_size=8) == 8
    assert cfg.derive_dp(world_size=4) == 4


def test_derive_dp_rejects_inconsistent_degrees() -> None:
    cfg = _cfg(data_parallel_shard_degree=1)
    with pytest.raises(ValueError):
        cfg.derive_dp(world_size=2)


def test_derive_dp_rejects_indivisible_world() -> None:
    cfg = _cfg(data_parallel_shard_degree=-1, tensor_parallel_degree=3)
    with pytest.raises(ValueError):
        cfg.derive_dp(world_size=8)  # 8 % 3 != 0


def test_derive_dp_narrows_by_the_non_dp_degrees() -> None:
    # tp=2 consumes half the ranks; the rest are data-parallel.
    cfg = _cfg(data_parallel_shard_degree=-1, tensor_parallel_degree=2)
    assert cfg.derive_dp(world_size=8) == 4


def test_build_parallel_dims_resolves_against_world_size() -> None:
    # Single process -> no process group and no parallelism to describe.
    assert build_parallel_dims(HybridMeshConfig(), world_size=1) is None

    cfg = _cfg(data_parallel_shard_degree=-1, tensor_parallel_degree=2)
    pd = build_parallel_dims(cfg, world_size=8)
    assert isinstance(pd, ParallelDims)
    # tp=2 over 8 ranks leaves 4 for data parallelism; dp_shard=-1 resolves here.
    assert (pd.tp, pd.dp_shard) == (2, 4)


def test_derive_dp_matches_parallel_dims_resolution() -> None:
    # The config helper and the torchtitan class must agree, or the trainer and
    # the mesh would disagree about how many ranks go to data parallelism.
    cfg = _cfg(data_parallel_shard_degree=-1, tensor_parallel_degree=2)
    pd = build_parallel_dims(cfg, world_size=8)
    assert cfg.derive_dp(world_size=8) == pd.dp_shard


def test_cp_only_still_needs_a_loss_reduction() -> None:
    """cp > 1 with dp = 1 shards the sequence but leaves the DP axis empty.

    The loss is summed over each rank's own *slice* of the sequence, so with
    only CP on a dp-only reduction would be over a size-1 group: every rank
    would report its own shard's loss as the whole batch's. The trainer
    therefore gates the loss reduce-group on ``dp_cp_enabled`` (dp *or* cp)
    rather than on how dense the mesh is.

    Only the flags are asserted here -- the group sizes they select need a live
    process group (``get_optional_mesh`` builds meshes). The sizes themselves
    are pinned by the ``expected_sizes`` table in ``parallel_dims.py``, which
    is what makes the property sufficient: ``loss`` is defined there as
    ``dp_replicate * dp_shard * cp``, so choosing it is choosing a group that
    spans the cp axis. The end-to-end version runs under torchrun in
    ``tests/cp_wiring_equivalence.py``.
    """
    cfg = _cfg(data_parallel_shard_degree=1, context_parallel_degree=2)
    pd = build_parallel_dims(cfg, world_size=2)
    assert isinstance(pd, ParallelDims)

    assert pd.cp_enabled
    assert not pd.dp_enabled
    # The property the trainer gates on: either axis alone is enough. Gating on
    # cp alone (or on dp alone) is the bug this pins.
    assert pd.dp_cp_enabled


def test_cp_must_divide_seq_len() -> None:
    with pytest.raises(ValueError):
        HybridMeshConfig(
            parallel=ParallelConfig(context_parallel_degree=3),
            training=TrainingConfig(max_seq_len=64),
        )


def test_gradient_accumulation_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(gradient_accumulation_steps=0)


def test_accumulation_and_gc_freq_reach_the_flat_view() -> None:
    """The trainer reads both off ``cfg``, not off ``cfg.training``.

    The flat view is a hand-written list of properties, so a new field on
    ``TrainingConfig`` stays invisible to the trainer until its passthrough
    exists. These two are the newest, and the failure mode is an AttributeError
    on the first training step rather than at parse time.
    """
    cfg = HybridMeshConfig(
        training=TrainingConfig(gradient_accumulation_steps=3, gc_freq=7)
    )
    assert cfg.gradient_accumulation_steps == 3
    assert cfg.gc_freq == 7
    # The defaults the trainer runs with when nothing is passed.
    assert HybridMeshConfig().gradient_accumulation_steps == 1
    assert HybridMeshConfig().gc_freq == 50


def _bare_trainer(cfg: HybridMeshConfig) -> Trainer:
    """A Trainer with __init__ bypassed, for testing pure data helpers."""
    t = Trainer.__new__(Trainer)
    t.cfg = cfg
    return t


def test_synthetic_batch_is_deterministic() -> None:
    cfg = HybridMeshConfig(
        training=TrainingConfig(global_batch_size=8, max_seq_len=16, seed=42)
    )
    t = _bare_trainer(cfg)
    # Two independent iterators over the same config must agree: that is what
    # makes two runs comparable and what makes every DP rank see one global batch.
    b1 = next(t._data_iterator())
    b2 = next(t._data_iterator())
    assert torch.equal(b1.input_ids, b2.input_ids)
    assert b1.input_ids.shape == (8, 16)
    assert torch.equal(b1.labels, b1.input_ids)


def test_dp_slice_partitions_global_batch() -> None:
    # Simulate 2 DP ranks without a process group by driving the slice math directly.
    cfg = HybridMeshConfig(
        training=TrainingConfig(global_batch_size=8, max_seq_len=16, seed=42)
    )
    t = _bare_trainer(cfg)
    batch = next(t._data_iterator())
    per = cfg.global_batch_size // 2
    r0 = batch.input_ids[0:per]
    r1 = batch.input_ids[per : 2 * per]
    assert torch.equal(torch.cat([r0, r1]), batch.input_ids)


# -- the model wrapper (merged in from the former tests/test_hf_wrapper.py) ----
#
# The wrapper's job is plumbing: expose the decoder's parts under stable names,
# add the batch dim HF expects, feed RoPE explicit ``position_ids``, and route
# attention through a mask. These tests run on the sdpa fallback, so the mask
# they exercise is the one sdpa gets -- for the flex path's own mask handling,
# see the ``_apply_attention`` tests at the end of this file.

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


def test_forward_applies_lm_head_and_gradients_flow(model: HFTransformerModel) -> None:
    input_ids = torch.randint(0, _VOCAB, (16,))
    logits = model(input_ids, positions=torch.arange(16))

    assert logits.shape == (16, _VOCAB)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_positions_drive_rope(model: HFTransformerModel) -> None:
    """Shifting every position must change the output -- otherwise RoPE is idling."""
    input_ids = torch.randint(0, _VOCAB, (16,))

    with torch.no_grad():
        base = model(input_ids, positions=torch.arange(16))
        shifted = model(input_ids, positions=torch.arange(16) + 3)
        repeated = model(input_ids, positions=torch.arange(16))

    assert not torch.allclose(base, shifted)
    assert torch.equal(base, repeated)


# -- the attention seam --------------------------------------------------------


def test_sdpa_fallback_gives_the_decoder_no_mask(model: HFTransformerModel) -> None:
    """Off CUDA the wrapper must NOT hand a BlockMask to the decoder.

    sdpa ignores ``is_causal`` whenever a mask is present and derives causality
    from the mask instead, so a BlockMask would land in ``attn_mask=`` and either
    crash (``BlockMask`` has no ``ndim``) or silently disable masking. The
    decoder is handed neither mask nor ``is_causal``: HF then builds what it
    needs itself, which is the path it is tested against.
    """
    assert model.model.config._attn_implementation != _ATTN_IMPLEMENTATION

    kwargs = model._apply_attention(torch.arange(16), None)

    assert kwargs == {"attention_mask": None}


def test_flex_backend_gets_the_block_mask(model: HFTransformerModel) -> None:
    """The flex path is the one that consumes the mask, and only it gets it."""
    model.model.config._attn_implementation = _ATTN_IMPLEMENTATION

    kwargs = model._apply_attention(torch.arange(16), None)

    assert type(kwargs["attention_mask"]).__name__ == "BlockMask"
    assert kwargs["is_causal"] is False


def test_flex_backend_passes_an_explicit_mask_through(
    model: HFTransformerModel,
) -> None:
    model.model.config._attn_implementation = _ATTN_IMPLEMENTATION
    sentinel = model.get_attention_masks(positions=torch.arange(16))

    kwargs = model._apply_attention(torch.arange(16), sentinel)

    assert kwargs["attention_mask"] is sentinel


def test_non_flex_backend_rejects_a_packed_sequence(model: HFTransformerModel) -> None:
    """Packing cannot be expressed by the fallback; it must fail, not go unmasked.

    Two documents back to back: the position counter restarts at index 3, which
    is what marks the boundary ``get_attention_masks`` would have masked.
    """
    packed = torch.tensor([0, 1, 2, 0, 1, 2])

    with pytest.raises(ValueError, match="packed sequence"):
        model._apply_attention(packed, None)


# -- the config path the trainer uses ------------------------------------------


def test_build_model_config_for_offline_arch() -> None:
    """A bare architecture name builds a local model from cfg's explicit sizes."""
    cfg = HybridMeshConfig(training=TrainingConfig(seed=42))

    config = build_model_config_for(cfg)

    assert config.model_type == cfg.hf_model
    assert config.vocab_size == cfg.vocab_size
    assert config.hidden_size == cfg.hidden_size
    assert config.num_hidden_layers == cfg.num_hidden_layers
    assert config.max_position_embeddings >= cfg.max_seq_len


def test_wrapper_forward_returns_logits_the_trainer_can_score() -> None:
    """The contract the training loop relies on: flat ids in, flat logits out.

    Loss is the trainer's business now, so this pins the boundary -- the wrapper
    yields one logit row per input token, and the trainer's next-token
    cross-entropy over those rows is a finite scalar.

    The labels handed to ``_loss_sum`` are already next-token aligned, which is
    what ``preprocess_inputs`` produces for both loaders; ``_loss_sum`` does no
    shifting of its own. A sequence whose every position is predictable
    therefore contributes one prediction per token.
    """
    cfg = HybridMeshConfig(
        training=TrainingConfig(seed=42, max_seq_len=32, global_batch_size=2)
    )

    model = HFTransformerModel(build_model_config_for(cfg)).eval()
    ids = torch.randint(0, cfg.vocab_size, (cfg.global_batch_size * cfg.max_seq_len,))

    with torch.no_grad():
        logits = model(ids)

    assert logits.shape == (ids.shape[0], cfg.vocab_size)
    loss_sum = Trainer._loss_sum(logits, ids)
    assert loss_sum.ndim == 0
    assert float(loss_sum) > 0

    # The model's own path marks its row ends IGNORE_INDEX, and those are then
    # excluded from the denominator rather than silently counted. The count
    # comes from ``_count_valid_tokens``, which sees the unsharded labels -- the
    # loss itself only skips the ignored rows.
    row_aware = next_token_targets(ids, seq_len=cfg.max_seq_len)
    counted = int((row_aware != IGNORE_INDEX).sum())
    assert counted == ids.shape[0] - cfg.global_batch_size
    row_loss = Trainer._loss_sum(logits, row_aware)
    assert row_loss.ndim == 0
    # Row-final positions predict nothing, so they contribute nothing.
    assert float(row_loss) < float(loss_sum)
