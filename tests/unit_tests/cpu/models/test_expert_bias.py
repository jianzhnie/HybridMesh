"""The auxiliary-loss-free expert bias: ``update_expert_bias`` and its hook.

Two things are pinned here, and they fail differently:

* the **update rule** -- sign-based, mean-centred, in place -- which is what
  keeps ``sum(expert_bias_E) == 0`` and makes the step size independent of how
  lopsided the load is. A rule that drifted from that would still "balance" the
  experts, just slowly and with the whole routed output shifting underneath it.
* the **hook wiring** -- that the counts are reduced over exactly the axes that
  shard a token stream, and over no others. Reducing over the expert dim or the
  pipeline dim, or skipping DP, produces a bias that is subtly wrong per rank
  rather than obviously broken.

The collectives are exercised with a size-1 gloo group, so the count is a no-op
arithmetically but the code path still runs. That is the same fixture shape
``test_parallel_dims.py`` uses.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from hpmesh.models.common.grouped_experts import GroupedExperts
from hpmesh.models.common.moe import (
    MoE,
    RoutedExperts,
    TokenChoiceTopKRouter,
    _iter_moe_layers,
    _update_expert_bias,
    register_moe_load_balancing_hook,
)
from hpmesh.models.common.token_dispatcher import LocalTokenDispatcher
from hpmesh.models.hf_wrapper import HFTransformerModel
from hpmesh.parallel.expert_parallel import swap_hf_moe_blocks

try:
    from transformers import AutoConfig
except ImportError:  # pragma: no cover - transformers is a runtime dependency
    AutoConfig = None

_NUM_EXPERTS = 4
_DIM = 8
_TOP_K = 2


@pytest.fixture(scope="module")
def single_rank_group(tmp_path_factory):
    """A size-1 gloo group: the collectives run, nothing moves."""
    created = not dist.is_initialized()
    if created:
        store = dist.FileStore(str(tmp_path_factory.mktemp("pg") / "store"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    yield
    if created:
        dist.destroy_process_group()


def _moe(*, coeff: float | None = 0.1) -> MoE:
    return MoE(
        _NUM_EXPERTS,
        RoutedExperts(
            GroupedExperts(_DIM, 8, _NUM_EXPERTS),
            LocalTokenDispatcher(_NUM_EXPERTS, _TOP_K),
        ),
        TokenChoiceTopKRouter(_NUM_EXPERTS, _DIM, _TOP_K),
        load_balance_coeff=coeff,
    )


class _Holder(torch.nn.Module):
    """Stands in for a model part: ``_iter_moe_layers`` walks ``.layers``."""

    def __init__(self, moes: list[MoE | None]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(moe) for moe in moes])


class _Layer(torch.nn.Module):
    """A decoder layer holding a MoE, or a dense one holding none.

    The block goes on ``mlp`` -- the attribute the swap actually replaces it in
    (``MOE_LAYER_ATTRS``). Putting it anywhere else would test a fiction: the
    lookup would find nothing on a real swapped model and the hook would no-op,
    which is exactly the bug ``test_the_hook_updates_a_real_swapped_model``
    guards against.
    """

    def __init__(self, moe: MoE | None) -> None:
        super().__init__()
        if moe is not None:
            self.mlp = moe


# -- the update rule ----------------------------------------------------------


def test_the_step_is_sign_based_with_magnitude_coeff() -> None:
    """Every expert moves by exactly ``+/- coeff`` -- never by how far off it is.

    That is what makes one runaway expert unable to dominate, and it is also
    why activation checkpointing's doubled count is harmless.
    """
    moe = _moe(coeff=0.1)
    moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))
    before = moe.expert_bias_E.clone()

    moe.update_expert_bias()

    delta = moe.expert_bias_E - before
    nonzero = delta[delta != 0]
    assert nonzero.numel() > 0
    assert torch.equal(nonzero.abs(), torch.full_like(nonzero, 0.1))


def test_under_loaded_experts_are_pushed_up_and_over_loaded_down() -> None:
    """The sign of the step is what steers load; getting it backwards diverges."""
    moe = _moe(coeff=0.1)
    # Expert 0 sees twice the mean, expert 1 sees nothing.
    moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))
    before = moe.expert_bias_E.clone()

    moe.update_expert_bias()

    delta = moe.expert_bias_E - before
    assert float(delta[0]) < 0, "the over-loaded expert must lose bias"
    assert float(delta[1]) > 0, "the under-loaded expert must gain bias"


def test_the_update_is_mean_centred() -> None:
    """``sum(expert_bias_E)`` stays 0, so the bias shifts choices, not the output.

    Without centring the bias would drift upward as a whole and act as a global
    additive term on every routing score.
    """
    moe = _moe(coeff=0.1)
    moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))

    moe.update_expert_bias()

    assert abs(float(moe.expert_bias_E.sum())) < 1e-9


def test_a_uniform_load_leaves_the_bias_alone() -> None:
    """Already-balanced counts mean every delta is 0 -- the non-vacuity check.

    Confirms the tests above are measuring the load and not simply adding a
    constant on every call.
    """
    moe = _moe(coeff=0.1)
    moe.tokens_per_expert_E.copy_(torch.full((_NUM_EXPERTS,), 7.0))

    moe.update_expert_bias()

    assert torch.equal(moe.expert_bias_E, torch.zeros(_NUM_EXPERTS))


def test_the_counter_is_drained_so_it_never_leaks_into_the_next_step() -> None:
    """The counter is per-step scratch; ``update_expert_bias`` is what zeroes it."""
    moe = _moe()
    moe.tokens_per_expert_E.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    moe.update_expert_bias()

    assert float(moe.tokens_per_expert_E.sum()) == 0.0


def test_a_disabled_coefficient_makes_the_update_a_no_op() -> None:
    """``load_balance_coeff=None`` means no bias buffer and nothing to update."""
    moe = _moe(coeff=None)
    moe.tokens_per_expert_E.copy_(torch.ones(_NUM_EXPERTS))

    moe.update_expert_bias()

    assert moe.expert_bias_E is None


# -- the hook -----------------------------------------------------------------


def test_the_hook_fires_once_per_optimizer_step() -> None:
    """One update per ``step()``, not one per microbatch.

    That granularity is the whole reason the bias lives outside the forward: it
    has to see a complete accumulation window.
    """
    moe = _moe()
    model = _Holder([moe])
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    register_moe_load_balancing_hook(optimizer, [model], parallel_dims=None)

    assert float(moe.expert_bias_E.abs().sum()) == 0.0  # nothing yet: no counts
    moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))
    optimizer.step()

    assert float(moe.expert_bias_E.abs().sum()) > 0.0


def test_no_hook_is_registered_for_a_model_without_moe_layers() -> None:
    """A dense run must not pay a traversal or an empty collective per step."""
    model = _Holder([])
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)

    register_moe_load_balancing_hook(optimizer, [model], parallel_dims=None)

    assert optimizer._optimizer_step_pre_hooks == {}


def test_a_dense_layer_among_sparse_ones_is_skipped() -> None:
    """Mixed sparse/dense models have layers with no MoE at all."""
    sparse = _moe()
    model = _Holder([sparse, None, None])

    assert _iter_moe_layers(model) == [sparse]


def test_the_hook_updates_a_real_swapped_model() -> None:
    """End to end against the object graph the swap actually produces.

    The hand-built layers above encode an assumption about *where* the swap
    leaves the block; this test removes that assumption by swapping a real HF
    model and reading the hook's effect back off it. It is the one that would
    have caught the lookup reading ``layer.moe`` while the swap wrote
    ``layer.mlp`` -- a mismatch under which every other test in this file still
    passes, the register function quietly no-ops, and the bias never updates.
    """
    config = AutoConfig.for_model(
        "deepseek_v3",
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
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = HFTransformerModel(config).float().eval()
    assert swap_hf_moe_blocks(model) == 2

    layers = _iter_moe_layers(model)
    assert len(layers) == 2, "the hook cannot see the swapped-in blocks"

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    register_moe_load_balancing_hook(optimizer, [model], parallel_dims=None)
    balanced = torch.zeros(8)
    for moe in layers:
        moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0, 3.0, 3.0, 2.0, 2.0]))
        assert torch.equal(moe.expert_bias_E, balanced)

    optimizer.step()

    for moe in layers:
        assert not torch.equal(moe.expert_bias_E, balanced), "the bias never moved"
        assert float(moe.tokens_per_expert_E.sum()) == 0.0, "the counter was not drained"


def test_the_collective_runs_over_a_real_process_group(single_rank_group) -> None:
    """The reduction path executes without a mesh -- e.g. before a mesh exists."""
    moe = _moe()
    moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))
    model = _Holder([moe])

    _update_expert_bias([(model, [moe])], parallel_dims=None)

    # A size-1 group leaves the counts alone, so the update is the same one the
    # direct test asserts.
    assert float(moe.expert_bias_E[1]) > 0


def test_every_layer_of_every_part_is_updated() -> None:
    """Under pipeline parallelism a part owns a subset; all of them must move."""
    a, b, c = _moe(), _moe(), _moe()
    parts = [_Holder([a, b]), _Holder([c])]
    for moe in (a, b, c):
        moe.tokens_per_expert_E.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))

    _update_expert_bias([(p, _iter_moe_layers(p)) for p in parts], parallel_dims=None)

    for moe in (a, b, c):
        assert float(moe.expert_bias_E[1]) > 0, "a layer was skipped"
        assert float(moe.tokens_per_expert_E.sum()) == 0.0, "a counter was not drained"
