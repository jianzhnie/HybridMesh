"""Activation checkpointing: wiring, bitwise equivalence, and recompute proof.

These run on a real ``HFTransformerModel`` over a tiny offline LLaMA (and, for
the selective path, a tiny offline Qwen3MoE), on CPU with no process group --
AC is per-layer wrapping, not a distributed feature; the distributed
composition is pinned in ``tests/integration_tests/ac_equivalence.py``
(torchrun).

Four things are pinned:

* the ``apply_*`` contract: no-op when off, loud error on an unknown mode or a
  selective mode without its config;
* numerics: logits and gradients with ``mode="full"`` and ``mode="selective"``
  must be BITWISE equal to the uncheckpointed run (``preserve_rng_state=True``
  restores the RNG for the recompute, and CPU kernels are deterministic);
* the memory trade is real: a forward hook on each wrapped layer's inner
  module must fire twice per forward/backward (once in forward, once in the
  backward-time recompute), versus once without AC;
* the selective save set's shape -- that upstream's ``topk`` entry is absent
  (HF's routers mutate it in place, which torch's selective checkpoint rejects)
  and that the fqn->shape expansion emits ``(in, out)``.

Plus the two non-wrapping modes: ``memory_budget`` sets (and validates) its
one ``torch._functorch.config`` global and refuses a torch that lacks it or a
run without compile; ``region`` is a loud ``NotImplementedError`` naming its
unlock conditions.
"""

from __future__ import annotations

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.utils.checkpoint import CheckpointPolicy

from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config
from hpmesh.parallel.activation_checkpoint import (
    VALID_AC_MODES,
    _get_default_save_ops,
    _mm_recompute_shapes,
    _selective_policy,
    apply_ac,
)
from hpmesh.parallel.parallelize_hf import parallelize_hf_transformers
from hpmesh.trainer.config import (
    MemoryBudgetACConfig,
    ParallelConfig,
    SelectiveACConfig,
    TrainingConfig,
)

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
    with pytest.raises(ValueError, match="unknown-mode"):
        apply_ac(_model(), "unknown-mode")


def test_selective_needs_its_config() -> None:
    """The mode is not silently downgraded to full when its config is missing."""
    with pytest.raises(ValueError, match="SelectiveACConfig"):
        apply_ac(_model(), "selective")


def test_valid_modes_are_the_configs_accepted_set() -> None:
    assert VALID_AC_MODES == ("none", "full", "selective", "memory_budget")


def test_region_mode_is_a_loud_not_implemented() -> None:
    """RegionAC is not silently aliased to another mode: its unlock conditions
    (a torch_remat dependency plus model-declared remat regions) are named."""
    with pytest.raises(NotImplementedError, match="torch_remat"):
        apply_ac(_model(), "region")


def test_full_wraps_every_layer() -> None:
    model = apply_ac(_model(), "full")

    assert len(model.layers) == _NUM_LAYERS
    assert all(isinstance(layer, CheckpointWrapper) for layer in model.layers)


def test_selective_wraps_every_layer() -> None:
    model = apply_ac(_model(), "selective", selective=SelectiveACConfig())

    assert len(model.layers) == _NUM_LAYERS
    assert all(isinstance(layer, CheckpointWrapper) for layer in model.layers)


def test_parallelize_hf_transformers_wires_ac_before_fsdp() -> None:
    """The one entry point applies AC on the plain (single-device) path too."""
    model = parallelize_hf_transformers(
        _model(),
        cfg=ParallelConfig(),
        mesh=None,
        parallel_dims=None,
        activation_checkpoint="full",
    )

    assert all(isinstance(layer, CheckpointWrapper) for layer in model.layers)


def test_parallelize_hf_transformers_threads_selective_ac() -> None:
    """``selective_ac`` reaches ``apply_ac`` through the entry point.

    Without this, a caller selecting ``selective`` would get the loud "needs
    its config" error -- so this pins the plumbing, not just the mode string.
    """
    model = parallelize_hf_transformers(
        _model(),
        cfg=ParallelConfig(),
        mesh=None,
        parallel_dims=None,
        activation_checkpoint="selective",
        selective_ac=SelectiveACConfig(),
    )

    assert all(isinstance(layer, CheckpointWrapper) for layer in model.layers)


# -- memory_budget ------------------------------------------------------------


def test_memory_budget_config_validates_range() -> None:
    """The upstream bound: finite and in [0, 1], default 0.5."""
    assert MemoryBudgetACConfig().memory_budget == 0.5
    MemoryBudgetACConfig(memory_budget=0.0)
    MemoryBudgetACConfig(memory_budget=1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        MemoryBudgetACConfig(memory_budget=-0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        MemoryBudgetACConfig(memory_budget=1.1)


def test_memory_budget_needs_its_config() -> None:
    with pytest.raises(ValueError, match="MemoryBudgetACConfig"):
        apply_ac(_model(), "memory_budget", compile_enabled=True)


def test_memory_budget_requires_compile() -> None:
    """Without compile the budget is a global nothing reads -- loud, not silent."""
    with pytest.raises(ValueError, match="requires compile"):
        apply_ac(
            _model(),
            "memory_budget",
            memory_budget=MemoryBudgetACConfig(),
            compile_enabled=False,
        )


def test_memory_budget_sets_the_functorch_global(monkeypatch) -> None:
    """The whole policy: one process-global the compile partitioner reads.

    The knob only exists on newer torch, so it is monkeypatched in here; the
    test then also pins that the model is handed back UNWRAPPED (this mode
    wraps nothing -- the partitioner does the work).
    """
    monkeypatch.setattr(
        torch._functorch.config, "activation_memory_budget", 1.0, raising=False
    )
    model = _model()

    assert (
        apply_ac(
            model,
            "memory_budget",
            memory_budget=MemoryBudgetACConfig(memory_budget=0.25),
            compile_enabled=True,
        )
        is model
    )

    assert torch._functorch.config.activation_memory_budget == 0.25
    assert not any(isinstance(layer, CheckpointWrapper) for layer in model.layers)


def test_memory_budget_refuses_a_torch_without_the_knob(monkeypatch) -> None:
    """On a torch whose functorch config has no ``activation_memory_budget``,
    setting it would be a silent no-op, so the mode refuses instead."""
    monkeypatch.delattr(
        torch._functorch.config, "activation_memory_budget", raising=False
    )
    with pytest.raises(NotImplementedError, match="activation_memory_budget"):
        apply_ac(
            _model(),
            "memory_budget",
            memory_budget=MemoryBudgetACConfig(),
            compile_enabled=True,
        )


def test_memory_budget_requires_compile_through_the_entry_point() -> None:
    """The entry point threads ``compile`` into the guard, so selecting the
    mode without compile fails at parallelize time, not at first backward."""
    with pytest.raises(ValueError, match="requires compile"):
        parallelize_hf_transformers(
            _model(),
            cfg=ParallelConfig(),
            mesh=None,
            parallel_dims=None,
            compile=False,
            activation_checkpoint="memory_budget",
            memory_budget_ac=MemoryBudgetACConfig(),
        )


def test_training_config_validates_memory_budget_and_region_modes() -> None:
    """Config-time fail fast, matching upstream's trainer validation."""
    TrainingConfig(activation_checkpoint_mode="memory_budget", compile=True)
    with pytest.raises(ValueError, match="requires training.compile"):
        TrainingConfig(activation_checkpoint_mode="memory_budget")
    with pytest.raises(NotImplementedError, match="torch_remat"):
        TrainingConfig(activation_checkpoint_mode="region")


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


def test_selective_ac_matches_uncheckpointed_bitwise() -> None:
    """Selective AC is a memory trade too: it must not move the dense numbers.

    The save set saves every second matmul and recomputes the rest, so the
    recompute replays matmuls the forward did -- ``preserve_rng_state`` plus
    deterministic CPU kernels keep that bitwise.
    """
    ref = _model()
    ref_logits = _loss_and_backward(ref)
    ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters()}

    model = apply_ac(_model(), "selective", selective=SelectiveACConfig())
    logits = _loss_and_backward(model)

    assert torch.equal(logits, ref_logits)
    for name, p in model.named_parameters():
        ref_name = name.replace("._checkpoint_wrapped_module", "")
        assert p.grad is not None, f"{name} got no gradient under selective AC"
        assert torch.equal(p.grad, ref_grads[ref_name]), f"{name} grad differs"


def test_backward_recomputes_each_layer_selectively() -> None:
    model = apply_ac(_model(), "selective", selective=SelectiveACConfig())
    assert _forward_counts(model) == [2] * _NUM_LAYERS


# -- selective: the save set and the fqn->shape expansion ----------------------


def test_default_save_ops_omits_topk() -> None:
    """The one deviation from upstream's set, and it is deliberate.

    HF's MoE routers normalize topk's output in place, which torch's selective
    checkpoint rejects ("Tensor cached ... has been mutated"), so saving topk
    would break every MoE model rather than protect it.
    """
    save_ops = _get_default_save_ops()

    assert torch.ops.aten.topk.default not in save_ops
    # Non-vacuity: the set is not just empty.
    assert torch.ops.aten.mm.default in save_ops
    assert torch.ops.aten._scaled_dot_product_attention_math.default in save_ops


def test_mm_recompute_shapes_uses_linear_in_out_order() -> None:
    """A Linear's weight is stored (out, in); the shapes set must hold (in, out).

    This is what makes the lookup match an ``aten.mm`` of the same GEMM, so the
    shapes are checked on a non-square projection -- on a square one a swapped
    order is invisible.
    """
    layer = _model().layers[0]

    # gate_proj is Linear(16 -> 32): weight (32, 16), so (in, out) = (16, 32).
    assert layer.mlp.gate_proj.weight.shape == (32, 16)
    assert _mm_recompute_shapes(layer, "layers.0", ["gate_proj"]) == {(16, 32)}
    assert _mm_recompute_shapes(layer, "layers.0", ["down_proj"]) == {(32, 16)}

    # A pattern matching a container (not an nn.Linear) is a loud error, so a
    # wrong fqn never passes for "matched nothing".
    with pytest.raises(ValueError, match="nn.Linear"):
        _mm_recompute_shapes(layer, "layers.0", ["layers.0"])


def test_selective_policy_recomputes_forced_shapes_and_alternates_the_rest() -> None:
    """The policy's two decisions: forced shapes, and the every-second matmul."""

    class Ctx:
        is_recompute = False

    mm = torch.ops.aten.mm.default
    lhs = torch.empty(4, 4)
    forced = torch.empty(4, 8)  # mm RHS is (in, out)
    other = torch.empty(4, 16)

    policy = _selective_policy({mm}, {(4, 8)})
    assert policy(Ctx(), mm, lhs, forced) is CheckpointPolicy.PREFER_RECOMPUTE

    # A different (in, out) still goes through the save-every-second dial:
    # the 1st is saved, the 2nd recomputed, the 3rd saved again.
    decisions = [policy(Ctx(), mm, lhs, other) for _ in range(3)]
    assert decisions == [
        CheckpointPolicy.MUST_SAVE,
        CheckpointPolicy.PREFER_RECOMPUTE,
        CheckpointPolicy.MUST_SAVE,
    ]


def test_selective_policy_normalizes_linear_weight_to_mm_order() -> None:
    """``aten.linear``'s args[1] is (out, in), so it needs transposing to match.

    Without it the forced-shape lookup would never fire on a Linear, which is
    the spelling every HF projection actually uses.
    """

    class Ctx:
        is_recompute = False

    linear = torch.ops.aten.linear.default
    # weight (out=8, in=4) is the same GEMM as an mm RHS of (in=4, out=8).
    policy = _selective_policy({linear}, {(4, 8)})
    assert (
        policy(Ctx(), linear, torch.empty(4, 4), torch.empty(8, 4))
        is CheckpointPolicy.PREFER_RECOMPUTE
    )


# -- selective on an MoE: the case the topk omission exists for ---------------

_MOE_SEQ = 32


def _moe_model(seed: int = 0) -> HFTransformerModel:
    """A tiny offline Qwen3MoE -- its router runs the topk the policy omits."""
    from transformers import AutoConfig

    torch.manual_seed(seed)
    return HFTransformerModel(
        AutoConfig.for_model(
            "qwen3_moe",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            router_aux_loss_coef=1e-3,
            max_position_embeddings=256,
            experts_implementation="eager",
        )
    )


def _moe_loss_and_backward(model: HFTransformerModel) -> torch.Tensor:
    ids = torch.randint(128, (_MOE_SEQ,), generator=torch.Generator().manual_seed(7))
    logits = model(ids, positions=torch.arange(_MOE_SEQ))
    logits.sum().backward()
    return logits.detach()


def test_selective_ac_matches_uncheckpointed_bitwise_on_moe() -> None:
    """Regression: saving topk made this raise, not merely differ.

    HF's router divides topk's output in place, so the cached tensor is
    mutated and torch aborts the recompute. This exercises the whole selective
    path over the MoE layer -- including the ``force_recompute_mm_shapes_by_fqns``
    expansion, which is empty by default here because HF's router is not an
    ``nn.Linear``.
    """
    ref = _moe_model()
    ref_logits = _moe_loss_and_backward(ref)
    ref_grads = {
        n.replace("._checkpoint_wrapped_module", ""): p.grad.clone()
        for n, p in ref.named_parameters()
    }

    model = apply_ac(_moe_model(), "selective", selective=SelectiveACConfig())
    logits = _moe_loss_and_backward(model)

    assert torch.equal(logits, ref_logits)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient under selective AC"
        ref_name = name.replace("._checkpoint_wrapped_module", "")
        assert torch.equal(p.grad, ref_grads[ref_name]), f"{name} grad differs"
