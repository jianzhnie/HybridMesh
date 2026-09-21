"""Activation checkpointing: wiring, bitwise equivalence, and recompute proof.

These run on a real ``HFTransformerModel`` over a tiny offline LLaMA, on CPU
with no process group -- AC is per-layer wrapping, not a distributed feature;
the distributed composition is pinned in
``tests/integration_tests/ac_equivalence.py`` (torchrun).

Three things are pinned:

* the ``apply_*`` contract: no-op when off, loud error on an unknown mode;
* numerics: logits and gradients with ``mode="full"`` must be BITWISE equal
  to the uncheckpointed run (``preserve_rng_state=True`` restores the RNG for
  the recompute, and CPU kernels are deterministic);
* the memory trade is real: a forward hook on each wrapped layer's inner
  module must fire twice per forward/backward (once in forward, once in the
  backward-time recompute), versus once without AC.
"""

from __future__ import annotations

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config
from hpmesh.parallel.activation_checkpoint import apply_ac
from hpmesh.parallel.parallelize_hf import parallelize_hf_transformers
from hpmesh.trainer.config import ParallelConfig

_VOCAB = 32
_HIDDEN = 16
_SEQ = 24
_NUM_LAYERS = 3


def _model(seed: int = 0) -> HFTransformerModel:
    """Deterministically initialized tiny decoder; identical across calls."""
    torch.manual_seed(seed)
    config = build_model_config(
        "llama",
        seq_len=_SEQ,
        arch_overrides={
            "vocab_size": _VOCAB,
            "hidden_size": _HIDDEN,
            "intermediate_size": 32,
            "num_hidden_layers": _NUM_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )
    return HFTransformerModel(config)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1234)
    ids = torch.randint(_VOCAB, (_SEQ,), generator=g)
    return ids, torch.arange(_SEQ)


def _loss_and_backward(model: HFTransformerModel) -> torch.Tensor:
    ids, positions = _batch()
    logits = model(ids, positions=positions)
    logits.sum().backward()
    return logits.detach()


# -- apply_* contract ----------------------------------------------------------


def test_noop_when_mode_is_none() -> None:
    model = _model()
    layers_before = list(model.layers)

    assert apply_ac(model, "none") is model
    assert list(model.layers) == layers_before


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="selective"):
        apply_ac(_model(), "selective")


def test_full_wraps_every_layer() -> None:
    model = apply_ac(_model(), "full")

    assert len(model.layers) == _NUM_LAYERS
    assert all(isinstance(layer, CheckpointWrapper) for layer in model.layers)


def test_parallelize_hf_transformers_wires_ac_before_fsdp() -> None:
    """The one entry point applies AC on the plain (single-device) path too."""
    model = parallelize_hf_transformers(
        _model(),
        cfg=ParallelConfig(backend="gloo"),
        mesh=None,
        parallel_dims=None,
        activation_checkpoint="full",
    )

    assert all(isinstance(layer, CheckpointWrapper) for layer in model.layers)


# -- numerics ------------------------------------------------------------------


def test_full_ac_matches_uncheckpointed_bitwise() -> None:
    ref = _model()
    ref_logits = _loss_and_backward(ref)
    ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters()}

    model = apply_ac(_model(), "full")
    logits = _loss_and_backward(model)

    assert torch.equal(logits, ref_logits)
    for name, p in model.named_parameters():
        # The wrapper inserts a ``_checkpoint_wrapped_module`` level into FQNs.
        ref_name = name.replace("._checkpoint_wrapped_module", "")
        assert p.grad is not None, f"{name} got no gradient under AC"
        assert torch.equal(p.grad, ref_grads[ref_name]), f"{name} grad differs"


# -- recompute proof -----------------------------------------------------------


def _forward_counts(model: HFTransformerModel) -> list[int]:
    """One forward+backward, counting each inner layer's forward invocations."""
    counts = [0] * len(model.layers)

    def make_hook(idx: int):
        def hook(module, args, output) -> None:
            counts[idx] += 1

        return hook

    handles = []
    for idx, layer in enumerate(model.layers):
        # Reach through the checkpoint wrapper to the layer itself, so the
        # backward-time recompute (which calls the inner forward) is counted.
        inner = getattr(layer, "_checkpoint_wrapped_module", layer)
        handles.append(inner.register_forward_hook(make_hook(idx)))
    try:
        _loss_and_backward(model)
    finally:
        for handle in handles:
            handle.remove()
    return counts


def test_backward_recomputes_each_layer() -> None:
    model = apply_ac(_model(), "full")
    assert _forward_counts(model) == [2] * _NUM_LAYERS


def test_no_recompute_without_ac() -> None:
    """Non-vacuity: the count of 2 above is AC's recompute, not the baseline."""
    assert _forward_counts(_model()) == [1] * _NUM_LAYERS
