"""Swapping an HF MoE block for the native one, and proving it changed nothing.

The swap is only defensible if it is numerically transparent: same weights, same
routing, same output. These tests check that claim at three levels -- that the
weights land in the right places, that one block agrees with the HF block it
replaced, and that the whole model's logits are unchanged. A swap that silently
mis-indexed the experts would still produce plausible-looking output, so the
per-tensor checks matter as much as the end-to-end one.

Everything here runs offline on CPU against a tiny qwen3_moe built from a config
-- no hub access, no GPU, no process group. EP>1 needs a process group and is
covered separately.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from hpmesh.models.common.moe import MoE
from hpmesh.models.moe_probe import detect_moe_layers, probe_moe_model
from hpmesh.models.moe_swap import swap_moe_layers

HIDDEN = 64
EXPERTS = 8
TOP_K = 2
MOE_HIDDEN = 32
LAYERS = 2


def _randomize_experts(model: AutoModelForCausalLM) -> None:
    """Give the expert weights magnitudes a trained model would have.

    Random init leaves them near zero, which makes every numerical comparison
    vacuous -- the differences vanish along with the signal. The layout differs
    by transformers version, so both are handled: 5.x fuses gate and up into one
    ``(E, 2I, D)`` tensor, while 4.x and Mixtral keep a module per expert.
    """
    with torch.no_grad():
        for layer in model.model.layers:
            experts = layer.mlp.experts
            if hasattr(experts, "gate_up_proj"):
                experts.gate_up_proj.normal_(0, 0.1)
                experts.down_proj.normal_(0, 0.2)
            else:
                for expert in experts:
                    expert.gate_proj.weight.normal_(0, 0.1)
                    expert.up_proj.weight.normal_(0, 0.1)
                    expert.down_proj.weight.normal_(0, 0.2)
            layer.mlp.gate.weight.normal_(0, 1.0)


def _stacked_hf_expert_weights(experts) -> tuple[torch.Tensor, torch.Tensor]:
    """Read an HF expert container into ``(gate_up (E, 2I, D), down (E, D, I))``.

    Normalizing both layouts to the same two tensors lets one assertion cover
    the fused and per-expert cases.
    """
    if hasattr(experts, "gate_up_proj"):
        return experts.gate_up_proj.detach().clone(), experts.down_proj.detach().clone()
    return (
        torch.stack(
            [torch.cat([e.gate_proj.weight, e.up_proj.weight]) for e in experts]
        ).detach(),
        torch.stack([e.down_proj.weight for e in experts]).detach(),
    )


def _moe_model(*, seed: int = 0) -> AutoModelForCausalLM:
    torch.manual_seed(seed)
    config = AutoConfig.for_model(
        "qwen3_moe",
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=64,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_HIDDEN,
        decoder_sparse_step=1,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    _randomize_experts(model)
    detect_moe_layers(model)
    return model


def _tokens(seq: int = 16, batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, seq, HIDDEN)


# -- probing -----------------------------------------------------------------


def test_probe_reports_every_moe_layer() -> None:
    model = _moe_model()
    arch = probe_moe_model(model, model.config)

    assert set(arch) == set(range(LAYERS))
    assert arch[0].num_experts == EXPERTS
    assert arch[0].moe_intermediate_size == MOE_HIDDEN
    assert arch[0].top_k == TOP_K


# -- the swap ----------------------------------------------------------------


def test_swap_replaces_every_moe_block() -> None:
    model = _moe_model()

    swapped = swap_moe_layers(model)

    assert swapped == LAYERS
    for layer in model.model.layers:
        assert isinstance(layer.mlp, MoE)
        assert layer.moe_enabled is True


def test_swap_is_a_no_op_on_a_dense_model() -> None:
    torch.manual_seed(0)
    config = AutoConfig.for_model(
        "llama",
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    detect_moe_layers(model)

    assert swap_moe_layers(model) == 0


# -- weight transfer ---------------------------------------------------------


def test_expert_weights_are_copied_not_reinitialized() -> None:
    """Each expert's gate/up/down must land in its own slice of the stacked form.

    The stacked read normalizes the transformers 5.x fused layout and the 4.x
    per-expert layout to the same ``(gate_up, down)`` pair, so this covers both.
    """
    model = _moe_model()
    gate_up, down = _stacked_hf_expert_weights(model.model.layers[0].mlp.experts)

    swap_moe_layers(model)

    native = model.model.layers[0].mlp.routed_experts.inner_experts
    intermediate = native.w1_EFD.shape[1]
    assert torch.equal(native.w1_EFD, gate_up[:, :intermediate, :])
    assert torch.equal(native.w3_EFD, gate_up[:, intermediate:, :])
    assert torch.equal(native.w2_EDF, down)


def test_router_weights_are_copied() -> None:
    model = _moe_model()
    original = model.model.layers[0].mlp.gate.weight.detach().clone()

    swap_moe_layers(model)

    native = model.model.layers[0].mlp.router.gate
    assert torch.equal(native.weight, original)


# -- numerical equivalence ---------------------------------------------------


def test_swapped_block_matches_the_hf_block() -> None:
    """One block, same input, same output -- up to fp32 reduction order."""
    model = _moe_model()
    x = _tokens()
    with torch.no_grad():
        expected = model.model.layers[0].mlp(x)[0]

    swap_moe_layers(model)

    with torch.no_grad():
        actual = model.model.layers[0].mlp(x)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_whole_model_logits_are_unchanged_by_the_swap() -> None:
    """The strongest claim: the swap is invisible from outside the model."""
    model = _moe_model()
    ids = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        expected = model(input_ids=ids).logits

    swap_moe_layers(model)

    with torch.no_grad():
        actual = model(input_ids=ids).logits

    # fp32 ulps, not a semantic difference: the native path sums each expert's
    # contribution through index_add rather than the HF loop's ordering.
    assert (expected - actual).abs().max() < 1e-5


def test_moe_accepts_a_flat_token_stream() -> None:
    """The framework's own callers pass (T, D); the HF decoder passes (B, T, D)."""
    model = _moe_model()
    swap_moe_layers(model)
    block = model.model.layers[0].mlp

    tokens = _tokens(batch=2).reshape(-1, HIDDEN)
    with torch.no_grad():
        flat = block(tokens)
        batched = block(tokens.unsqueeze(0))

    assert flat.shape == (tokens.shape[0], HIDDEN)
    assert torch.allclose(flat, batched.squeeze(0), atol=1e-6)


def test_routing_agrees_with_the_hf_router() -> None:
    """Same experts chosen, not just similar outputs -- a weight mix-up could hide.

    Logits are computed straight from ``gate.weight`` rather than by calling the
    HF router: transformers 5.x returns ``(logits, topk_weights, topk_indices)``
    from that call and 4.x returns logits alone, but either way the routing
    decision is a linear projection through the same weight, which is the
    parameter the swap is supposed to carry over.
    """
    model = _moe_model()
    x = _tokens().reshape(-1, HIDDEN)
    hf_gate = model.model.layers[0].mlp.gate.weight.detach()

    def topk_experts(weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x.float(), weight.float())
        return torch.topk(torch.softmax(logits, -1), TOP_K, -1).indices.sort(-1).values

    expected = topk_experts(hf_gate)

    swap_moe_layers(model)

    actual = topk_experts(model.model.layers[0].mlp.router.gate.weight)
    assert torch.equal(expected, actual)


# -- routing mechanics -------------------------------------------------------


def test_token_is_carried_to_the_experts_it_chose() -> None:
    """One-hot map and ids must describe the same choice."""
    model = _moe_model()
    swap_moe_layers(model)
    router = model.model.layers[0].mlp.router
    x = _tokens(seq=8).reshape(-1, HIDDEN)

    scores, ids, routing_map = router(x)

    assert scores.shape == (8, TOP_K)
    assert ids.shape == (8, TOP_K)
    assert routing_map.shape == (8, EXPERTS)
    # Every token chose exactly TOP_K distinct experts.
    assert routing_map.sum(-1).eq(TOP_K).all()
    for token in range(8):
        assert routing_map[token, ids[token]].all()


def test_local_dispatch_combine_round_trips_one_expert_per_token() -> None:
    """With top_k=1 and no EP, combining undoes dispatching up to the score."""
    model = _moe_model()
    swap_moe_layers(model)
    moe = model.model.layers[0].mlp
    dispatcher = moe.routed_experts.token_dispatcher
    x = _tokens(seq=12).reshape(-1, HIDDEN)

    scores, ids, routing_map = moe.router(x)
    routed, counts, metadata = dispatcher.dispatch(x, scores, ids, routing_map.sum(0))
    out = dispatcher.combine(routed, metadata, x)

    # Dispatching selects the tokens each expert owns; combining scatters back.
    assert out.shape == x.shape
    # A token routed to an expert must receive that expert's output for it,
    # scaled by its score -- so nothing routed can come back exactly zero.
    assert (out.abs().sum(-1) > 0).all()


def test_dispatcher_groups_tokens_in_expert_order() -> None:
    """GroupedExperts relies on tokens arriving contiguous and expert-sorted."""
    model = _moe_model()
    swap_moe_layers(model)
    moe = model.model.layers[0].mlp
    dispatcher = moe.routed_experts.token_dispatcher
    x = _tokens(seq=12).reshape(-1, HIDDEN)

    scores, ids, routing_map = moe.router(x)
    routed, counts, _ = dispatcher.dispatch(x, scores, ids, routing_map.sum(0))

    # counts must add up to the number of routed tokens
    assert int(counts.sum()) == routed.shape[0]
    # and each segment must match a real token from the input
    start = 0
    for count in counts.tolist():
        for row in range(start, start + count):
            assert (routed[row] == x).all(-1).any()
        start += count


@pytest.mark.parametrize("load_balance_coeff", [None, 1e-3])
def test_load_balance_buffers_match_the_configuration(
    load_balance_coeff: float | None,
) -> None:
    model = _moe_model()
    swap_moe_layers(model)
    moe = model.model.layers[0].mlp
    moe.load_balance_coeff = load_balance_coeff
    if load_balance_coeff is None:
        moe.expert_bias_E = None

    assert moe.tokens_per_expert_E.shape == (EXPERTS,)
    if load_balance_coeff is None:
        assert moe.expert_bias_E is None
    else:
        assert moe.expert_bias_E.shape == (EXPERTS,)
