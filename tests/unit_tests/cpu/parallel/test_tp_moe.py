"""Tests for MoE-under-TP: plan specs, weight sharding, and the fail-fast matrix.

The boundary collectives (``_TPMoeSequenceBoundary``'s all-gather /
reduce-scatter pair) need a real process group, and the fused path needs CUDA;
neither runs in this environment. What is pinned here is everything around
them: the plan-spec resolution, the expert-weight shard layout, the
single-process partial-sum equivalence that the reduce-scatter completes (the
same math, checked without a group), state-dict FQN stability, the tp=1
no-op, and every loud-raise of the combination matrix.

The multi-rank forward/backward equivalence against an unsharded reference is
NOT covered here -- it needs torch >= 2.12 with the distributed stack this
environment lacks; re-run on a multi-GPU host before trusting the numerics.
"""

from __future__ import annotations

from tests.caps import require_env

require_env('spmd_types')


import pytest
import torch
import torch.nn as nn

from hpmesh.errors import UnsupportedCombinationError
from hpmesh.parallel.tensor_parallel import apply_tp
from hpmesh.parallel.tensor_parallel.tp import (
    _MOE_PLAN_SPECS,
    _resolve_plan,
    _shard_experts_for_tp,
)
from hpmesh.trainer import ParallelConfig

MOE_PLAN = {
    "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    "layers.*.mlp.experts": "moe_tp_experts",
}


class _Experts(nn.Module):
    """The transformers 5.x fused-experts shape: (E, 2F, D) and (E, D, F)."""

    def __init__(self, num_experts: int, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * hidden, dim))
        self.down_proj = nn.Parameter(torch.randn(num_experts, dim, hidden))

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for e in range(self.gate_up_proj.shape[0]):
            mask = routing_weights[:, e] != 0
            if not mask.any():
                continue
            gate, up = torch.nn.functional.linear(
                x[mask], self.gate_up_proj[e]
            ).chunk(2, dim=-1)
            out[mask] = out[mask] + torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up, self.down_proj[e]
            ) * routing_weights[mask, e : e + 1]
        return out


class _MoeBlock(nn.Module):
    """A minimal HF-style MoE block: router + fused experts, top-k routing.

    Carries the attributes the swap probe reads (``gate`` weight, ``top_k``)
    so ``_is_hf_moe_block`` recognizes it.
    """

    def __init__(self, num_experts: int, dim: int, hidden: int, top_k: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = _Experts(num_experts, dim, hidden)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lead = hidden_states.shape[:-1]
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        scores = torch.softmax(self.gate(x).float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        routing = torch.zeros_like(scores).scatter_(-1, topk_ids, topk_weights)
        out = self.experts(x, routing)
        return out.reshape(*lead, -1)


class _MoeModel(nn.Module):
    def __init__(self, num_experts: int = 4, dim: int = 16, hidden: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.ModuleDict({"mlp": _MoeBlock(num_experts, dim, hidden, top_k=2)})]
        )
        self._tp_plan = MOE_PLAN


class _FakeMeshAxis:
    """The ``mesh["tp"]`` surface apply_tp touches, with no process group."""

    device_type = "cpu"

    def __init__(self, size: int, rank: int) -> None:
        self._size = size
        self._rank = rank

    def get_group(self):
        return None

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


class _FakeMesh:
    def __init__(self, size: int, rank: int) -> None:
        self._axis = _FakeMeshAxis(size, rank)

    def __getitem__(self, name: str):
        assert name == "tp"
        return self._axis


def _applied_model(tp_size: int, tp_rank: int, seed: int = 0) -> _MoeModel:
    torch.manual_seed(seed)
    model = _MoeModel()
    cfg = ParallelConfig(tensor_parallel_size=tp_size)
    apply_tp(model, mesh=_FakeMesh(tp_size, tp_rank), cfg=cfg)
    return model


# -- plan resolution ----------------------------------------------------------


def test_moe_plan_specs_resolve_instead_of_raising() -> None:
    plan = _resolve_plan(_MoeModel(), None)
    assert set(plan) == set(MOE_PLAN)
    # The MoE spec strings resolve to None: nothing for the per-Linear engine
    # to swap; the structural MoE path realizes them. "rowwise" on the packed
    # down_proj keeps its dense meaning -- harmless, since down_proj is a
    # stacked parameter, not an nn.Linear.
    assert plan["layers.*.mlp.experts.gate_up_proj"] is None
    assert plan["layers.*.mlp.experts"] is None
    assert plan["layers.*.mlp.experts.down_proj"].kind == "rowwise"


def test_ep_plan_strings_still_raise_on_the_tp_path() -> None:
    class M(nn.Module):
        _tp_plan = {"layers.*.mlp.experts.gate_up_proj": "grouped_gemm"}

    with pytest.raises(ValueError, match="Unsupported TP plan entry"):
        _resolve_plan(M(), None)


# -- weight sharding ----------------------------------------------------------


def test_expert_shards_reconstruct_the_full_weights() -> None:
    torch.manual_seed(0)
    block = _MoeBlock(num_experts=4, dim=16, hidden=8, top_k=2)
    gate_up_full = block.experts.gate_up_proj.detach().clone()
    down_full = block.experts.down_proj.detach().clone()

    shards = []
    for rank in range(2):
        torch.manual_seed(0)
        b = _MoeBlock(num_experts=4, dim=16, hidden=8, top_k=2)
        ids = _shard_experts_for_tp(b, tp_size=2, tp_rank=rank)
        assert id(b.experts.gate_up_proj) in ids
        assert id(b.experts.down_proj) in ids
        # The router is never sharded.
        assert id(b.gate.weight) not in ids
        shards.append(
            (b.experts.gate_up_proj.detach(), b.experts.down_proj.detach())
        )

    # gate_up: (E, 2F, D), gate half and up half sharded separately on dim 1,
    # so each rank's stored (E, F, D) shard is cat(gate_shard, up_shard).
    f = down_full.shape[2]
    rec_gate = torch.cat([g[:, : f // 2] for g, _ in shards], dim=1)
    rec_up = torch.cat([g[:, f // 2 :] for g, _ in shards], dim=1)
    assert torch.equal(rec_gate, gate_up_full[:, :f])
    assert torch.equal(rec_up, gate_up_full[:, f:])
    # down: (E, D, F) sharded on dim 2.
    rec = torch.cat([d for _, d in shards], dim=2)
    assert torch.equal(rec, down_full)


def test_partial_expert_outputs_sum_to_the_unsharded_reference() -> None:
    """The reduce-scatter's sum, checked without a process group.

    Each "rank" runs the block with its F-shard of the experts on the full
    token stream; the partial outputs summed across ranks must equal the
    unsharded block's output. This is the arithmetic the boundary
    reduce-scatter completes -- the collective itself is environment-uncovered.
    """
    torch.manual_seed(0)
    ref = _MoeBlock(num_experts=4, dim=16, hidden=8, top_k=2)
    x = torch.randn(3, 5, 16)
    with torch.no_grad():
        expected = ref(x)

    partial = None
    for rank in range(2):
        torch.manual_seed(0)
        block = _MoeBlock(num_experts=4, dim=16, hidden=8, top_k=2)
        _shard_experts_for_tp(block, tp_size=2, tp_rank=rank)
        with torch.no_grad():
            contribution = block(x)
        partial = contribution if partial is None else partial + contribution
    # Not bit-exact: the sharded path splits each expert GEMM into per-rank
    # halves whose partial sums are added in a different order. fp32 rounding
    # is the only difference.
    assert torch.allclose(partial, expected, atol=1e-4)


def test_shard_experts_raises_when_f_is_not_divisible() -> None:
    block = _MoeBlock(num_experts=4, dim=16, hidden=6, top_k=2)
    with pytest.raises(ValueError, match="not divisible"):
        _shard_experts_for_tp(block, tp_size=4, tp_rank=0)


# -- engine behavior ----------------------------------------------------------


def test_apply_tp_shards_the_block_and_keeps_state_dict_fqns() -> None:
    torch.manual_seed(0)
    model = _MoeModel()
    fqns_before = set(model.state_dict())
    router_before = model.layers[0]["mlp"].gate.weight.detach().clone()

    cfg = ParallelConfig(tensor_parallel_size=2)
    apply_tp(model, mesh=_FakeMesh(2, 0), cfg=cfg)

    # FQNs are unchanged; only the expert shapes shrank.
    assert set(model.state_dict()) == fqns_before
    block = model.layers[0]["mlp"]
    assert block.experts.gate_up_proj.shape == (4, 8, 16)  # 2F/tp on dim 1
    assert block.experts.down_proj.shape == (4, 16, 4)  # F/tp on dim 2
    # Router replicated, bit-identical.
    assert torch.equal(block.gate.weight, router_before)
    # The boundary mixin and the grad-allreduce exclusion marker are installed.
    assert block._tp_seq_group is not None or hasattr(block, "_tp_seq_group")
    assert block._tp_sharded_param_ids == frozenset(
        {id(block.experts.gate_up_proj), id(block.experts.down_proj)}
    )


def test_apply_tp_is_idempotent_on_a_moe_block() -> None:
    model = _applied_model(tp_size=2, tp_rank=0)
    cfg = ParallelConfig(tensor_parallel_size=2)
    apply_tp(model, mesh=_FakeMesh(2, 0), cfg=cfg)  # second pass
    block = model.layers[0]["mlp"]
    # Not re-sharded: the shapes are still the tp=2 shard, not tp=4.
    assert block.experts.gate_up_proj.shape == (4, 8, 16)


def test_apply_tp_raises_when_moe_specs_match_no_block() -> None:
    class Dense(nn.Module):
        _tp_plan = dict(MOE_PLAN)

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

    cfg = ParallelConfig(tensor_parallel_size=2)
    with pytest.raises(UnsupportedCombinationError, match="no HF MoE block"):
        apply_tp(Dense(), mesh=object(), cfg=cfg)


def test_apply_tp_raises_on_a_shared_expert_block() -> None:
    model = _MoeModel()
    block = model.layers[0]["mlp"]
    block.shared_expert = nn.Linear(16, 16)  # DeepSeek-style shared FFN
    cfg = ParallelConfig(tensor_parallel_size=2)
    with pytest.raises(NotImplementedError, match="shared expert"):
        apply_tp(model, mesh=_FakeMesh(2, 0), cfg=cfg)


# -- the combination matrix ---------------------------------------------------


def test_tp_and_ep_together_are_now_allowed() -> None:
    # tp x ep, upstream-aligned: TP shards the dense parts, EP owns the
    # routed experts, the router stays replicated.
    ParallelConfig(tensor_parallel_size=2, expert_parallel_size=2)


def test_tp_ep_cp_together_fail_fast_at_the_config() -> None:
    with pytest.raises(NotImplementedError, match="tp x ep x cp"):
        ParallelConfig(
            tensor_parallel_size=2,
            expert_parallel_size=2,
            context_parallel_size=2,
        )


def test_ep_alone_and_tp_alone_are_still_valid_configs() -> None:
    ParallelConfig(expert_parallel_size=2)
    ParallelConfig(tensor_parallel_size=2)


def test_apply_tp_defers_the_moe_blocks_to_ep_when_ep_is_on() -> None:
    """With ep>1, apply_tp must leave HF MoE blocks untouched for the swap."""
    torch.manual_seed(0)
    model = _MoeModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = ParallelConfig(tensor_parallel_size=2, expert_parallel_size=2)
    apply_tp(model, mesh=_FakeMesh(2, 0), cfg=cfg)

    block = model.layers[0]["mlp"]
    # No F-sharding, no boundary mixin, no exclusion marker.
    assert block.experts.gate_up_proj.shape == (4, 16, 16)
    assert block.experts.down_proj.shape == (4, 16, 8)
    assert not hasattr(block, "_tp_sharded_param_ids")
    assert "_tp_moe_boundary" not in block.__dict__
    assert not type(block).__name__.startswith("TPMoe")
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_swap_refuses_a_shared_expert_block_under_tp_x_ep() -> None:
    from hpmesh.parallel.expert_parallel.swap import swap_hf_moe_blocks

    model = _MoeModel()
    model.layers[0]["mlp"].shared_expert = nn.Linear(16, 16)
    with pytest.raises(NotImplementedError, match="shared"):
        swap_hf_moe_blocks(model, ep_group=None, tp_enabled=True)


def test_tp_sharded_param_ids_covers_dense_tp_and_ep_experts_not_router() -> None:
    from hpmesh.models.common.grouped_experts import GroupedExperts
    from hpmesh.parallel.tensor_parallel.tp import ColwiseLinear
    from hpmesh.trainer.trainer import _tp_sharded_param_ids

    grouped = GroupedExperts(dim=8, hidden_dim=4, num_experts=2)
    router = nn.Linear(8, 2, bias=False)
    dense_tp = ColwiseLinear(torch.randn(8, 8), tp_size=2, tp_rank=0, group=None)
    part = nn.ModuleDict({"ge": grouped, "gate": router, "proj": dense_tp})

    ids = _tp_sharded_param_ids([part])
    assert id(dense_tp.weight) in ids  # dense TP shard
    for p in grouped.parameters():
        assert id(p) in ids  # EP expert slice: grad complete per rank
    assert id(router.weight) not in ids  # replicated: must still be summed


def test_tp_one_leaves_a_moe_model_bit_identical() -> None:
    torch.manual_seed(0)
    model = _MoeModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = ParallelConfig()  # tp = 1
    assert apply_tp(model, mesh=None, cfg=cfg) is model
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
    assert "moe_tp_experts" in _MOE_PLAN_SPECS
