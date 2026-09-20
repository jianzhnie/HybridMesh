"""The EP swap: a swapped-in hpmesh MoE must reproduce the HF block it replaced.

Single-process, CPU. ``tests/ep_wiring_equivalence.py`` covers the multi-rank
all-to-all; what this suite pins is the swap itself: weight movement, router
parity (softmax scoring, optional top-k renormalization), the refusal paths,
and the load-balance aux loss the swapped router carries.

Comparisons run in float64. They are not exact: both HF and hpmesh compute
routing scores in fp32 (HF via ``softmax(dtype=float)``, hpmesh's
``RouterGateLinear`` by construction), but HF computes the gate GEMM in the
model dtype while hpmesh computes it in fp32 -- so the scores agree only to
fp32 rounding. 1e-6 separates that noise floor (~5e-8) from a wiring error
(O(1)).
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig

from hpmesh.models.common.aux_loss import AuxLoss
from hpmesh.models.common.moe import MoE
from hpmesh.models.hf_wrapper import HFTransformerModel
from hpmesh.parallel.ep import swap_hf_moe_blocks

TOL = 1e-6


@pytest.fixture(autouse=True)
def _aux_loss_state():
    """Snapshot and restore AuxLoss's class-level state around each test."""
    counts = dict(AuxLoss._group_counts)
    acc = dict(AuxLoss.group_acc)
    denominator = AuxLoss._step_denominator
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None
    yield
    AuxLoss._group_counts.clear()
    AuxLoss._group_counts.update(counts)
    AuxLoss.group_acc.clear()
    AuxLoss.group_acc.update(acc)
    AuxLoss._step_denominator = denominator


def _config(*, norm_topk_prob: bool, aux_coeff: float = 1e-3) -> AutoConfig:
    """A tiny offline Qwen3Moe: 2 layers, 8 experts, top-2 routing."""
    return AutoConfig.for_model(
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
        norm_topk_prob=norm_topk_prob,
        router_aux_loss_coef=aux_coeff,
        max_position_embeddings=256,
    )


def _model(config, *, seed: int = 0) -> HFTransformerModel:
    """Deterministically initialized tiny Qwen3Moe in float64."""
    torch.manual_seed(seed)
    return HFTransformerModel(config).to(torch.float64).eval()


def _data(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(128, (40,), generator=g)
    return ids, torch.arange(40)


@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_swapped_model_matches_hf_output(norm_topk_prob: bool) -> None:
    """Same weights, same tokens: the swap must not change the forward."""
    config = _config(norm_topk_prob=norm_topk_prob)
    ref = _model(config)
    swapped = _model(config)

    assert swap_hf_moe_blocks(swapped) == 2
    assert all(isinstance(layer.mlp, MoE) for layer in swapped.layers)

    ids, positions = _data()
    with torch.no_grad():
        expected = ref(ids, positions=positions)
        got = swapped(ids, positions=positions)
    torch.testing.assert_close(got, expected, rtol=TOL, atol=TOL)


def test_swap_moves_the_weights_verbatim() -> None:
    """w1/w3/w2 are gate_proj/up_proj/down_proj, elementwise (no transpose)."""
    config = _config(norm_topk_prob=True)
    ref = _model(config)
    swapped = _model(config)
    swap_hf_moe_blocks(swapped)

    hf_block = ref.layers[0].mlp
    moe = swapped.layers[0].mlp
    grouped = moe.routed_experts.inner_experts
    assert torch.equal(moe.router.gate.weight, hf_block.gate.weight)
    for e, hf_expert in enumerate(hf_block.experts):
        assert torch.equal(grouped.w1_EFD[e], hf_expert.gate_proj.weight)
        assert torch.equal(grouped.w3_EFD[e], hf_expert.up_proj.weight)
        assert torch.equal(grouped.w2_EDF[e], hf_expert.down_proj.weight)


def test_swap_preserves_eval_mode() -> None:
    """A fresh module defaults to training=True; the swap must not flip an
    eval-built model's blocks (the aux loss would fire without a denominator)."""
    swapped = _model(_config(norm_topk_prob=True))
    swap_hf_moe_blocks(swapped)
    assert not any(layer.mlp.training for layer in swapped.layers)

    ids, positions = _data()
    with torch.no_grad():
        swapped(ids, positions=positions)  # must not raise for a denominator


def test_swap_rejects_a_dense_model() -> None:
    """EP on a model with no MoE block is a config mistake; refuse loudly."""
    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = _model(config)
    with pytest.raises(TypeError, match="no HF MoE block"):
        swap_hf_moe_blocks(model)


def test_swap_rejects_a_model_without_layers() -> None:
    with pytest.raises(TypeError, match=r"\.layers"):
        swap_hf_moe_blocks(torch.nn.Linear(4, 4))


def test_aux_loss_is_injected_on_the_router_scores() -> None:
    """Training forward: the load-balance loss accumulates a metric and moves
    the router's gradient, while the forward output is bitwise unchanged."""
    ids, positions = _data()

    with_aux = _model(_config(norm_topk_prob=True, aux_coeff=1e-3))
    swap_hf_moe_blocks(with_aux)
    without_aux = _model(_config(norm_topk_prob=True, aux_coeff=0.0))
    swap_hf_moe_blocks(without_aux)
    assert without_aux.layers[0].mlp.router.aux_loss is None

    AuxLoss.set_step_denominator(torch.tensor(39.0))
    with_aux.train()
    without_aux.train()
    out_with = with_aux(ids, positions=positions)
    out_without = without_aux(ids, positions=positions)
    # Identity forward: the aux loss injects a gradient, not a value.
    assert torch.equal(out_with, out_without)
    out_with.sum().backward()
    out_without.sum().backward()

    router_with = with_aux.layers[0].mlp.router
    router_without = without_aux.layers[0].mlp.router
    assert router_with.aux_loss.instance_acc.item() > 0
    grad_diff = (
        (router_with.gate.weight.grad - router_without.gate.weight.grad)
        .abs()
        .max()
        .item()
    )
    assert grad_diff > 0


def test_aux_loss_requires_the_step_denominator() -> None:
    """Training forward without ``set_step_denominator`` must refuse."""
    model = _model(_config(norm_topk_prob=True))
    swap_hf_moe_blocks(model)
    model.train()
    ids, positions = _data()
    with pytest.raises(ValueError, match="set_step_denominator"):
        model(ids, positions=positions)
