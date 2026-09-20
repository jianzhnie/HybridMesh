"""`HFTransformerModel`: the seam between the training loop and HuggingFace.

The wrapper's job is plumbing -- expose the decoder's parts under stable names,
add the batch dim HF expects, feed RoPE explicit `position_ids`, and route
attention through a mask -- but three separate layers read those names (FSDP
walks `layers`, the parallel layer renames `tp_plan`, the trainer scores the
logits), so a silent change here propagates everywhere.

These run on the sdpa fallback, so the mask exercised below is the one sdpa
gets. The flex path's own mask handling is covered at the end of the file by
flipping `_attn_implementation` on the built config.
"""

from __future__ import annotations

import pytest
import torch

from hpmesh.components.loss import IGNORE_INDEX, next_token_targets
from hpmesh.models.hf_wrapper import (
    _ATTN_IMPLEMENTATION,
    HFTransformerModel,
    build_model_config,
    build_model_config_for,
)
from hpmesh.trainer import HybridMeshConfig, TrainingConfig

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


# -- part names ---------------------------------------------------------------


def test_named_children_flattens_the_decoder(model: HFTransformerModel) -> None:
    """Five parts, in the order the parallel layer walks them.

    The names are a contract: ``apply_tp`` matches TP-plan entries against these
    paths, and FSDP walks the same children to find the transformer blocks.
    """
    names = [name for name, _ in model.named_children()]
    assert names == ["tok_embeddings", "layers", "norm", "lm_head", "rotary_emb"]


def test_state_dict_does_not_duplicate_tensors(model: HFTransformerModel) -> None:
    """The decoder is aliased for convenience; it must not be registered twice."""
    keys = list(model.state_dict())
    assert not any(key.startswith("_decoder.") for key in keys)
    # Every parameter appears exactly once.
    assert len(keys) == len(set(keys))


# -- forward ------------------------------------------------------------------


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

    with pytest.raises(AssertionError):
        torch.testing.assert_close(base, shifted, rtol=1e-5, atol=1e-8)
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


def test_attention_masks_is_a_block_mask(model: HFTransformerModel) -> None:
    positions = torch.arange(8)
    mask = model.get_attention_masks(positions=positions)
    assert type(mask).__name__ == "BlockMask"


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


def test_the_mask_type_follows_the_corpus_rather_than_being_configured() -> None:
    """Packed corpora need the document mask; the synthetic one must not pay for it.

    ``build_model_config_for`` derives ``attn_mask_type`` from the dataset
    selector because the two cannot be set independently without one of them
    being wrong: every non-random corpus is packed by ``datasets/build.py``, and
    nothing in the config names packing separately. Setting it by hand -- which
    is what the equivalence tests used to do -- is the drift this prevents.
    """
    from dataclasses import replace

    from hpmesh.trainer.config import DataloaderConfig

    training = TrainingConfig(seed=42)
    synthetic = build_model_config_for(HybridMeshConfig(training=training))
    assert synthetic.attn_mask_type == "causal"

    packed = build_model_config_for(
        HybridMeshConfig(
            training=replace(
                training,
                dataloader_config=DataloaderConfig(
                    dataset="local_jsonl",
                    tokenizer_path="/tmp/tokenizer",
                    dataset_path="/tmp/corpus",
                ),
            )
        )
    )
    assert packed.attn_mask_type == "block_causal"


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
    from hpmesh.trainer.trainer import Trainer

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
