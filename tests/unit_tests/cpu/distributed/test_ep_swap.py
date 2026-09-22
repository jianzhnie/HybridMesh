"""The EP swap: a swapped-in hpmesh MoE must reproduce the HF block it replaced.

Single-process, CPU. ``tests/ep_wiring_equivalence.py`` covers the multi-rank
all-to-all; what this suite pins is the swap itself: weight movement, router
parity (softmax scoring, optional top-k renormalization), the refusal paths,
and the load-balance aux loss the swapped router carries.

Comparisons run in float64 where the model allows it. They are not exact: both
HF and hpmesh compute routing scores in fp32 (HF via ``softmax(dtype=float)``,
hpmesh's ``RouterGateLinear`` by construction), but HF computes the gate GEMM in
the model dtype while hpmesh computes it in fp32 -- so the scores agree only to
fp32 rounding. 1e-6 separates that noise floor (~5e-8) from a wiring error
(O(1)).

Every MoE config here requests ``experts_implementation="eager"``. transformers
5.x defaults to ``"grouped_mm"``, which dispatches the experts to
``torch._grouped_mm`` -- bf16-only, and unavailable on CPU entirely. The
auto-fallback to ``"eager"`` that normally saves a CPU run does not fire, because
``_grouped_mm_can_dispatch`` checks the device and the pointer alignment but
never the dtype, so a float32/float64 model passes the gate at ``from_config``
and then dies inside the first forward. Asking for ``"eager"`` explicitly gets
HF's per-expert ``F.linear`` loop, which is the same arithmetic form hpmesh's
``GroupedExperts`` falls back to -- so the two sides are comparable op for op,
which is what makes a bitwise assertion meaningful rather than merely close.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig

from hpmesh.models.common.aux_loss import AuxLoss
from hpmesh.models.common.moe import MoE, RoutedExperts
from hpmesh.models.hf_wrapper import HFTransformerModel
from hpmesh.parallel.expert_parallel import swap_hf_moe_blocks
from hpmesh.parallel.expert_parallel.ep import _restore_fp32_state_buffers

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
        experts_implementation="eager",
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
    """Same weights, same tokens: the swap must not change the forward.

    A tolerance rather than equality, for the reason spelled out in the module
    docstring: the reference runs one fused 2F-wide GEMM per expert where the
    swap runs two F-wide ones, so the two sides differ in fp32 accumulation.
    ``test_swap_moves_the_weights_verbatim`` pins the weights themselves, which
    is where an actual wiring error would show.
    """
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
    """w1/w3 are the two halves of ``gate_up_proj``; w2 is ``down_proj``.

    Elementwise, with no transpose: transformers 5.x splits each expert's fused
    projection with ``linear(x, gate_up_proj[e]).chunk(2, dim=-1)``, so the gate
    is the first half of the output dim and the up projection the second. That
    the two halves are not swapped is a property nothing else here would catch,
    since both project into the same ``(F, D)`` shape.
    """
    config = _config(norm_topk_prob=True)
    ref = _model(config)
    swapped = _model(config)
    swap_hf_moe_blocks(swapped)

    hf_block = ref.layers[0].mlp
    moe = swapped.layers[0].mlp
    grouped = moe.routed_experts.inner_experts
    assert torch.equal(moe.router.gate.weight, hf_block.gate.weight)
    gate_up = hf_block.experts.gate_up_proj
    hidden = gate_up.shape[1] // 2
    assert torch.equal(grouped.w1_EFD, gate_up[:, :hidden])
    assert torch.equal(grouped.w3_EFD, gate_up[:, hidden:])
    assert torch.equal(grouped.w2_EDF, hf_block.experts.down_proj)


def test_the_gate_and_up_halves_are_not_interchangeable() -> None:
    """Pins the split against HF itself rather than against our reader.

    The copy above is an identity check -- it re-derives the same halves the
    swap derived, so a reader that had gate and up swapped would satisfy it.
    This one rebuilds an expert's output from the *swapped* tensors using HF's
    own two lines and requires it to equal HF's own output, which holds only
    for the true split.
    """
    config = _config(norm_topk_prob=True)
    ref = _model(config)
    swapped = _model(config)
    swap_hf_moe_blocks(swapped)

    hf_experts = ref.layers[0].mlp.experts
    grouped = swapped.layers[0].mlp.routed_experts.inner_experts
    act = ref.layers[0].mlp.experts.act_fn

    x = torch.randn(6, config.hidden_size, dtype=torch.float64)
    with torch.no_grad():
        for e in range(config.num_experts):
            gate, up = torch.nn.functional.linear(x, hf_experts.gate_up_proj[e]).chunk(
                2, dim=-1
            )
            expected = torch.nn.functional.linear(
                act(gate) * up, hf_experts.down_proj[e]
            )
            got = torch.nn.functional.linear(
                act(x @ grouped.w1_EFD[e].T) * (x @ grouped.w3_EFD[e].T),
                grouped.w2_EDF[e],
            )
            assert torch.equal(got, expected), f"expert {e} is mis-split"


def test_swap_sets_the_fsdp_moe_flags() -> None:
    """FSDP's MoE branch keys off ``layer.moe_enabled`` / ``layer.moe``.

    Without them the expert weights are sharded as dense parameters over the
    dense DP mesh, mixing ranks of different EP coordinates into one FSDP
    group. ``moe`` must be a plain attribute, not a registered submodule: it
    aliases ``layer.mlp``, and registering it would double every expert weight
    in the state_dict.
    """
    swapped = _model(_config(norm_topk_prob=True))
    swap_hf_moe_blocks(swapped)

    for layer in swapped.layers:
        assert layer.moe_enabled is True
        assert layer.moe is layer.mlp
        assert "moe" not in layer._modules
        assert "moe" not in dict(layer.named_modules())


def test_swap_preserves_eval_mode() -> None:
    """A fresh module defaults to training=True; the swap must not flip an
    eval-built model's blocks (the aux loss would fire without a denominator)."""
    swapped = _model(_config(norm_topk_prob=True))
    swap_hf_moe_blocks(swapped)
    assert not any(layer.mlp.training for layer in swapped.layers)

    ids, positions = _data()
    with torch.no_grad():
        swapped(ids, positions=positions)  # must not raise for a denominator


def test_a_router_with_bias_is_refused() -> None:
    """``RouterGateLinear`` has no bias slot; dropping one would change scores.

    No supported family has a biased router, so the guard fires only on a
    family the probe does not know -- pinned here so it cannot rot into a
    silent drop.
    """
    model = _model(_config(norm_topk_prob=True))
    gate = model.layers[0].mlp.gate
    gate.bias = torch.nn.Parameter(
        torch.zeros(gate.weight.shape[0], dtype=gate.weight.dtype)
    )

    with pytest.raises(NotImplementedError, match="router bias"):
        swap_hf_moe_blocks(model)


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


# -- DeepSeek-V3: a different block shape --------------------------------------
#
# DeepSeek-V3 differs from Qwen3Moe in two ways the swap has to absorb: the
# router is a bespoke module rather than an ``nn.Linear``, and it carries an
# ``e_score_correction_bias`` buffer. It also has *no* ``router_aux_loss_coef``
# in its HF config, unlike Qwen3Moe.
#
# transformers 5.x moved the routing attributes (``top_k``, ``n_group``,
# ``topk_group``, ``norm_topk_prob``, ``routed_scaling_factor``) onto the *block*
# for every family; in 4.x DeepSeek kept them on the router. The probe reads
# both locations, and ``test_deepseek_v3_routing_attributes_live_on_the_block``
# pins the fact so a future move shows up as a failure rather than as the probe
# quietly falling back.
#
# These run in float32 rather than float64: HF's own ``DeepseekV3MoE.moe``
# allocates its accumulator as ``torch.zeros_like(hidden_states,
# dtype=topk_weights.dtype)``, and ``topk_weights`` is fp32 on every path, so a
# float64 DeepSeek fails in *its own* forward (index_add_ dtype error) before
# any swapped code runs.


def _deepseek_config(**overrides) -> AutoConfig:
    """A tiny offline DeepSeek-V3: 3 layers, first dense, 8 experts, 2 groups."""
    settings = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        max_position_embeddings=256,
        experts_implementation="eager",
    )
    settings.update(overrides)
    return AutoConfig.for_model("deepseek_v3", **settings)


def _deepseek_model(**overrides) -> HFTransformerModel:
    torch.manual_seed(0)
    return HFTransformerModel(_deepseek_config(**overrides)).float().eval()


def test_deepseek_v3_routing_attributes_are_probeable() -> None:
    """Dependency versions may place routing metadata on block or router."""
    model = _deepseek_model()
    block, router = model.layers[1].mlp, model.layers[1].mlp.gate
    owners = (block, router, model.model.config)
    for aliases in (
        ("top_k", "num_experts_per_tok"),
        ("n_group", "num_group"),
        ("topk_group",),
        ("norm_topk_prob",),
    ):
        assert any(
            getattr(owner, attr, None) is not None
            for owner in owners
            for attr in aliases
        ), f"{aliases} are absent from all supported owners"


@pytest.mark.parametrize(
    "overrides",
    [
        {"norm_topk_prob": True, "routed_scaling_factor": 2.5},
        {"norm_topk_prob": False, "routed_scaling_factor": 2.5},
        {"norm_topk_prob": True, "routed_scaling_factor": 1.0},
        {"scoring_func": "softmax", "norm_topk_prob": True},
    ],
)
def test_deepseek_v3_swap_matches_hf_output(overrides: dict) -> None:
    """Same weights, same tokens: the swap must not change DeepSeek's forward.

    Not bitwise, and the reason is a *shape* difference rather than a wrong
    weight. transformers 5.x fuses an expert's gate and up projections into one
    ``(E, 2F, D)`` tensor and applies them in a single ``F.linear`` of output
    width 2F, splitting afterwards; hpmesh keeps them apart and runs two GEMMs
    of width F. The operands are the same numbers -- ``GateUpProj[:, :F]`` is
    literally ``w1_EFD`` -- but fp32 accumulates over D in a different
    association, so the two agree only to rounding. In float64 the difference is
    exactly 0, which is what identifies it as accumulation order rather than an
    arithmetic error.

    The bar is therefore ``TOL`` at both ends: a tolerance that admits the
    rounding but not a wiring error, since 2e-7 against outputs of magnitude
    0.45 still separates a wrong half or a wrong expert (O(1)) from this noise.
    """
    config = _deepseek_config(**overrides)
    torch.manual_seed(0)
    ref = HFTransformerModel(config).float().eval()
    torch.manual_seed(0)
    swapped = HFTransformerModel(config).float().eval()

    # first_k_dense_replace=1, so only the two sparse layers swap.
    assert swap_hf_moe_blocks(swapped) == 2
    assert not isinstance(swapped.layers[0].mlp, MoE)
    assert all(isinstance(layer.mlp, MoE) for layer in swapped.layers[1:])

    ids, positions = _data()
    with torch.no_grad():
        torch.testing.assert_close(
            swapped(ids, positions=positions),
            ref(ids, positions=positions),
            rtol=TOL,
            atol=TOL,
        )


def test_deepseek_v3_router_settings_reach_the_hpmesh_router() -> None:
    """The probed routing config, as the swapped router ends up holding it.

    Every one of these is read from a *different* place in the HF model (the
    block for the numbers, a buffer on the router for the scoring function), so
    this is the end-to-end check that the probe assembles them correctly.
    """
    model = _deepseek_model()

    swap_hf_moe_blocks(model)
    router = model.layers[1].mlp.router
    assert router.score_func == "sigmoid"
    assert router.route_norm is True
    assert router.route_scale == 2.5
    assert router.num_expert_groups == 2
    assert router.num_limited_groups == 1


def test_deepseek_v3_shared_experts_survive_the_swap() -> None:
    """DeepSeek has an (ungated) shared expert; the swap must carry it over.

    The spelling is ``shared_experts``, plural. Qwen3Moe's blocks have neither,
    so a probe that looked only for the singular would drop this one in silence
    and change the forward by a whole FFN.
    """
    model = _deepseek_model()
    assert model.layers[1].mlp.shared_experts is not None

    swap_hf_moe_blocks(model)
    assert model.layers[1].mlp.shared_experts is not None


def test_deepseek_v3_gets_the_load_balancing_bias() -> None:
    """The probed ``e_score_correction_bias`` turns the bias path on -- and only there.

    Qwen3Moe has no such buffer, so the coefficient is decided per block from
    the probed router rather than globally. Otherwise every model would get a
    bias, and a frozen zero one would land in every checkpoint.
    """
    model = _deepseek_model()
    swap_hf_moe_blocks(model)

    sparse = [layer.mlp for layer in model.layers if isinstance(layer.mlp, MoE)]
    assert all(moe.load_balance_coeff is not None for moe in sparse)
    assert all(moe.expert_bias_E is not None for moe in sparse)

    qwen = _model(_config(norm_topk_prob=True))
    swap_hf_moe_blocks(qwen)
    assert all(layer.mlp.load_balance_coeff is None for layer in qwen.layers)


def test_the_bias_is_copied_out_of_the_hf_router() -> None:
    """The bias is optimization state, so a swap must not silently reset it."""
    model = _deepseek_model()
    hf_router = model.layers[1].mlp.gate
    with torch.no_grad():
        hf_router.e_score_correction_bias.copy_(
            torch.arange(8, dtype=torch.float32) / 100
        )

    swap_hf_moe_blocks(model)
    assert torch.equal(
        model.layers[1].mlp.expert_bias_E, hf_router.e_score_correction_bias
    )


def test_deepseek_v3_gets_no_aux_loss_from_its_own_config() -> None:
    """Its HF config has no ``router_aux_loss_coef``, so the loss is absent.

    A real gap rather than an oversight in the swap: DeepSeek-V3's design calls
    for a sequence-wise balance loss, and its config simply has no field to
    carry the coefficient. ``router_aux_loss_coef`` on the swap is the knob that
    supplies one; without it the router carries no aux loss at all.
    """
    model = _deepseek_model()
    assert getattr(model.model.config, "router_aux_loss_coef", None) is None

    swap_hf_moe_blocks(model)
    assert model.layers[1].mlp.router.aux_loss is None

    explicit = _deepseek_model()
    swap_hf_moe_blocks(explicit, router_aux_loss_coef=1e-3)
    assert explicit.layers[1].mlp.router.aux_loss is not None


def test_the_swap_refuses_to_guess_a_shared_expert_gate() -> None:
    """A gated shared expert is the one shared-expert shape MoE cannot express.

    ``MoE.shared_experts`` is additive; a ``shared_expert_gate`` multiplies.
    Dropping the gate would change the forward by a whole FFN's worth of
    scaling, so refusing is the point. No architecture in the supported set hits
    it (Qwen3.5 does), but the guard is pinned so it cannot rot unnoticed.
    """
    model = _deepseek_model()
    model.layers[1].mlp.shared_expert_gate = torch.nn.Linear(64, 1)

    with pytest.raises(NotImplementedError, match="shared_expert_gate"):
        swap_hf_moe_blocks(model)


# -- the other families: where the block lives, and what the experts are called -


def _plain_settings(**overrides) -> dict:
    """DeepSeek-shaped settings, minus ``model_type`` (see ``_plain_model``)."""
    settings = _deepseek_config(**overrides).to_dict()
    settings.pop("model_type", None)
    return settings


def _plain_model(family: str, **overrides) -> HFTransformerModel:
    """A tiny model built straight from ``family``'s own config.

    ``to_dict()`` carries ``model_type``, which collides with ``for_model``'s
    first positional parameter -- so it is dropped before forwarding.
    """
    cfg = AutoConfig.for_model(family, **_plain_settings(**overrides))
    torch.manual_seed(0)
    return HFTransformerModel(cfg).float().eval()


def _mixtral_model(**overrides) -> HFTransformerModel:
    settings = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        experts_implementation="eager",
    )
    settings.update(overrides)
    torch.manual_seed(0)
    return (
        HFTransformerModel(AutoConfig.for_model("mixtral", **settings)).float().eval()
    )


@pytest.mark.parametrize("family", ["glm4_moe", "deepseek_v2"])
def test_the_deepseek_router_shape_swaps_exactly(family: str) -> None:
    """GLM4 and DeepSeek-V2 share V3's block shape but not all of its rules.

    GLM4's routing is V3's with the bias rename; DeepSeek-V2's *default*
    ``topk_method`` ("greedy") routes freely over all experts even though its
    block carries ``num_group``/``topk_group``. Reading those grouping
    attributes without checking the method confines every token to a group HF
    never imposes -- a routing change, so it moves the output far more than the
    fused-versus-split GEMM rounding this asserts within (module docstring).
    """
    cfg = AutoConfig.for_model(family, **_plain_settings())
    torch.manual_seed(0)
    ref = HFTransformerModel(cfg).float().eval()
    torch.manual_seed(0)
    swapped = HFTransformerModel(cfg).float().eval()

    assert swap_hf_moe_blocks(swapped) == 2

    ids, positions = _data()
    with torch.no_grad():
        torch.testing.assert_close(
            swapped(ids, positions=positions),
            ref(ids, positions=positions),
            rtol=TOL,
            atol=TOL,
        )


def test_deepseek_v2_greedy_leaves_the_grouping_off() -> None:
    """The attributes are present but unused; the swap must not switch them on.

    This is the shape of the bug the test above catches: the probe reads
    ``num_group`` off the block and gets a number that looks meaningful, while
    HF's ``greedy`` path never consults it.
    """
    model = _plain_model("deepseek_v2")
    block, gate = model.layers[1].mlp, model.layers[1].mlp.gate
    assert (
        getattr(block, "topk_method", None)
        or getattr(gate, "topk_method", None)
    ) == "greedy"
    assert (
        getattr(block, "num_group", None) or getattr(gate, "num_group", None)
    ) == 2  # present, unused

    swap_hf_moe_blocks(model)
    router = model.layers[1].mlp.router
    assert router.num_expert_groups is None
    assert router.num_limited_groups is None


def test_deepseek_v2_group_limited_greedy_is_refused() -> None:
    """Its rule is a group ``max``; this swap implements V3's top-2 sum.

    Both are "group-limited", so nothing in the config distinguishes them and
    the wrong rule would just pick different experts -- a silent numeric drift.
    """
    model = _plain_model("deepseek_v2", topk_method="group_limited_greedy")

    with pytest.raises(NotImplementedError, match="single best expert"):
        swap_hf_moe_blocks(model)


def test_deepseek_v2_does_not_renormalize() -> None:
    """Its routing ignores ``norm_topk_prob`` entirely.

    transformers 5.x moved ``topk_method`` off the gate and onto the block, and
    dropped ``norm_topk_prob`` from DeepSeek-V2 altogether -- the routing never
    read it. A guard that looked for the declared field therefore stops working
    on the upgrade *while still looking correct*, and the swap silently starts
    renormalizing weights HF leaves raw. The marker has to be ``topk_method``,
    wherever the version puts it.
    """
    model = _plain_model("deepseek_v2")
    block, gate = model.layers[1].mlp, model.layers[1].mlp.gate
    assert (
        getattr(block, "topk_method", None)
        or getattr(gate, "topk_method", None)
    ) == "greedy"

    swap_hf_moe_blocks(model)
    assert model.layers[1].mlp.router.route_norm is False


def test_mixtral_swaps_in_place() -> None:
    """Mixtral's block lives on ``mlp`` like every other family in 5.x.

    It used to be ``block_sparse_moe``; a swap that still looked there would
    find no MoE at all on a Mixtral and refuse the model outright.
    """
    model = _mixtral_model()
    assert type(model.layers[0].mlp).__name__ == "MixtralSparseMoeBlock"

    assert swap_hf_moe_blocks(model) == 2

    for layer in model.layers:
        assert isinstance(layer.mlp, MoE)
        assert not hasattr(layer, "block_sparse_moe")


def test_mixtral_matches_hf_output() -> None:
    """The one family whose per-expert names match ``GroupedExperts``'.

    Mixtral's ``w1``/``w2``/``w3`` are the least likely to be mis-mapped by a
    reader -- and the most dangerous to mis-map, because ``w2`` (the ``(F, D)``
    down projection) fits ``w3_EFD``'s ``(F, D)`` slot exactly. A forward
    comparison against HF is what catches that, shapes being no help at all.

    A tolerance rather than equality: same fused-versus-split GEMM difference as
    every other family (module docstring). ``2F == D`` here, so an argument from
    the shapes being degenerate does not apply -- the accumulation order still
    differs, because HF splits *after* the 2F-wide GEMM and hpmesh never forms
    it.
    """
    cfg = AutoConfig.for_model(
        "mixtral",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        experts_implementation="eager",
    )
    torch.manual_seed(0)
    ref = HFTransformerModel(cfg).float().eval()
    torch.manual_seed(0)
    swapped = HFTransformerModel(cfg).float().eval()

    swap_hf_moe_blocks(swapped)
    ids, positions = _data()
    with torch.no_grad():
        torch.testing.assert_close(
            swapped(ids, positions=positions),
            ref(ids, positions=positions),
            rtol=TOL,
            atol=TOL,
        )


def test_mixtral_returns_a_bare_tensor() -> None:
    """transformers 5.x dropped the ``(hidden, router_logits)`` tuple.

    Its decoder layer used to unpack two values; the swap used to wrap the
    block to keep supplying them. Both are gone, so the block must be a plain
    ``MoE`` -- a surviving wrapper would fail the other way round.
    """
    model = _mixtral_model()
    swap_hf_moe_blocks(model)

    ids, positions = _data()
    with torch.no_grad():
        out = model(ids, positions=positions)
    assert out.shape == (40, 128)


def test_gpt_oss_is_refused() -> None:
    """Its experts are transposed and carry biases; ``GroupedExperts`` has neither.

    ``gate_up_proj`` is ``(E, D, 2F)`` rather than ``(E, 2F, D)``, so the shared
    split convention would slice the *token* dim and then copy a plausibly-shaped
    wrong tensor. Silently converting it is therefore not an option, and the
    refusal has to reach the caller as a refusal rather than as a "not a MoE
    block" skip that leaves the model running with unswapped experts.

    Built from ``GptOssExperts`` directly rather than as a whole model. A
    ``GptOssForCausalLM`` cannot be instantiated on CPU at all -- HF's
    ``_supports_sdpa`` guard rejects the eager attention this build falls back
    to -- and the layout is the thing under test, not the model around it. The
    block is hand-assembled because ``swap_hf_moe_blocks`` walks ``.layers``,
    which a bare experts module does not have.
    """
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    config = AutoConfig.for_model(
        "gpt_oss",
        hidden_size=64,
        intermediate_size=128,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    block = torch.nn.Module()
    block.experts = GptOssExperts(config)
    assert block.experts.gate_up_proj.shape[1] == 64  # (E, D, 2F), not (E, 2F, D)

    holder = torch.nn.Module()
    holder.layers = torch.nn.ModuleList([torch.nn.Module()])
    holder.layers[0].mlp = block

    with pytest.raises(NotImplementedError, match="bias vectors"):
        swap_hf_moe_blocks(holder)


def test_a_block_with_the_wrong_expert_layout_is_rejected() -> None:
    """A misshapen ``gate_up_proj`` must not be sliced on a guess.

    Refusing beats guessing: the shape check is what stands between a bad
    layout and a quiet half-copy, and it has to fail as a shape error rather
    than as an index error somewhere deeper.
    """
    model = _deepseek_model()
    with torch.no_grad():
        model.layers[1].mlp.experts.gate_up_proj = torch.nn.Parameter(
            torch.zeros(8, 7, 64)  # an odd output dim: not 2F
        )

    with pytest.raises(ValueError, match="unrecognized expert weight shapes"):
        swap_hf_moe_blocks(model)


def test_the_swap_keeps_the_token_count_buffer_in_fp32() -> None:
    """The swapped MoE's state buffers must survive ``.to(dtype=...)`` unscathed.

    ``Module.to`` converts every floating-point buffer along with the
    parameters, and ``_convert_block`` casts the new MoE to match the HF block's
    dtype. In a bf16 run that silently demoted the load-balancing buffers:
    ``tokens_per_expert_E`` is a token *count*, which bf16 cannot hold exactly
    past 256 -- 1001 becomes 1000 -- and ``expert_bias_E`` is an additive
    correction a bf16 round per step would erode. Neither has a gradient, so the
    cast bought nothing. hpmesh is fp32-only today, which is exactly why this
    needs a test: the bug is invisible until a bf16 path exists, and then it is
    silent.

    Qwen3Moe carries no ``e_score_correction_bias``, so only the count buffer is
    present here; ``_restore_fp32_state_buffers`` covers both and
    ``test_the_fp32_restore_covers_the_bias_buffer`` pins the other.
    """
    swapped = _model(_config(norm_topk_prob=True))

    assert swap_hf_moe_blocks(swapped) == 2

    moe = swapped.layers[0].mlp
    assert isinstance(moe, MoE)
    assert moe.tokens_per_expert_E.dtype == torch.float32

    # And the conversion really was asked for: the parameters did follow the
    # block, so this is not passing because nothing was cast at all.
    assert moe.router.gate.weight.dtype == torch.float64  # nosec

    # A count bf16 would have mangled round-trips exactly.
    moe.tokens_per_expert_E += 1001
    assert moe.tokens_per_expert_E[0].item() == 1001.0


def test_the_fp32_restore_covers_the_bias_buffer() -> None:
    """``_restore_fp32_state_buffers`` is what the check above relies on.

    Built directly rather than through the swap, because the families that carry
    ``expert_bias_E`` are a larger fixture than this property needs. The point is
    narrow and worth isolating: yes to float buffers, no to a plain
    ``Module.to``.
    """
    from hpmesh.models.common.grouped_experts import GroupedExperts

    grouped = GroupedExperts(dim=16, hidden_dim=32, num_experts=256)
    moe = MoE(
        num_experts=256,
        routed_experts=RoutedExperts(grouped, None),
        router=torch.nn.Linear(16, 256, bias=False),
        load_balance_coeff=1e-3,
    )
    assert moe.expert_bias_E is not None, "no bias registered -- vacuous"

    # Without the restore, this is what the swap used to do.
    moe.to(dtype=torch.bfloat16)
    assert moe.expert_bias_E.dtype == torch.bfloat16  # the bug, demonstrated

    moe.to(dtype=torch.bfloat16)  # re-cast so the fix under test starts from it
    _restore_fp32_state_buffers(moe)

    assert moe.expert_bias_E.dtype == torch.float32
    assert moe.tokens_per_expert_E.dtype == torch.float32
    assert moe.router.weight.dtype == torch.bfloat16  # params still follow

    # Exactness, which is the whole reason the dtype matters.
    # 0.012345 as bf16 is 0.012329; as fp32 it is exact to float32 precision.
    moe.expert_bias_E[0] = 0.012345
    moe.tokens_per_expert_E += 1001
    assert (
        moe.expert_bias_E[0].item()
        == torch.tensor(0.012345, dtype=torch.float32).item()
    )
    assert moe.tokens_per_expert_E[0].item() == 1001.0
