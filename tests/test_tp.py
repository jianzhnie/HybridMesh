"""Tests for the tensor-parallel declaration layer and engine wiring.

The collectives themselves (``AllGatherLinear`` / ``LinearReduceScatter``) run
only on CUDA symmetric memory, so a full forward is not testable here. What IS
testable on CPU is everything around them: the declaration, the weight layout
each kind produces, plan resolution, and that ``apply_tp`` leaves a model alone
when TP is off.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hpmesh.parallel.tp import (
    ColwiseLinear,
    RowwiseLinear,
    ShardingConfig,
    _match,
    _resolve_plan,
    apply_tp,
    colwise,
    rowwise,
)
from hpmesh.trainer import HybridMeshConfig


def test_declaration_factories_pick_the_right_realizer() -> None:
    assert colwise() == ShardingConfig(kind="colwise", implementation=ColwiseLinear)
    assert rowwise() == ShardingConfig(kind="rowwise", implementation=RowwiseLinear)


def test_colwise_layout_is_transposed_then_cut_on_output() -> None:
    # nn.Linear weight is [out, in]; colwise stores [in, out/tp].
    W = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)  # out=4, in=8
    mod = ColwiseLinear(W, tp_size=2, tp_rank=0, group=None)
    assert mod.weight.shape == (8, 2)
    assert torch.equal(mod.weight, W.t().contiguous()[:, 0:2])
    # rank 1 holds the other half
    other = ColwiseLinear(W, tp_size=2, tp_rank=1, group=None)
    assert torch.equal(other.weight, W.t().contiguous()[:, 2:4])


def test_rowwise_layout_is_cut_on_input_features() -> None:
    W = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    mod = RowwiseLinear(W, tp_size=2, tp_rank=1, group=None)
    assert mod.weight.shape == (4, 4)
    assert torch.equal(mod.weight, W[:, 4:8])


def test_sharded_weights_reconstruct_the_original() -> None:
    W = torch.randn(6, 12)
    for cls in (ColwiseLinear, RowwiseLinear):
        shards = [cls(W, tp_size=3, tp_rank=r, group=None).weight for r in range(3)]
        if cls is RowwiseLinear:
            rec = torch.cat(shards, dim=1)
        else:
            rec = torch.cat(shards, dim=1).t().contiguous()
        assert torch.equal(rec, W)


def test_plan_resolution_from_hf_string_map() -> None:
    class M(nn.Module):
        _tp_plan = {
            "layers.*.q_proj": "colwise",
            "layers.*.o_proj": "rowwise",
        }

    plan = _resolve_plan(M(), None)
    assert {k: v.kind for k, v in plan.items()} == {
        "layers.*.q_proj": "colwise",
        "layers.*.o_proj": "rowwise",
    }
    assert _match(plan, "layers.3.q_proj").kind == "colwise"
    assert _match(plan, "layers.0.o_proj").kind == "rowwise"
    assert _match(plan, "layers.0.up_proj") is None


def test_apply_tp_is_a_noop_when_tp_is_one() -> None:
    model = nn.Linear(4, 4)
    cfg = HybridMeshConfig()  # tp defaults to 1
    assert apply_tp(model, mesh=None, cfg=cfg) is model
    assert isinstance(model, nn.Linear)  # not swapped
