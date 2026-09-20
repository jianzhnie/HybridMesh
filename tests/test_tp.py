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

from hpmesh.parallel.tensor_parallel.tp import (
    ColwiseLinear,
    RowwiseLinear,
    ShardingConfig,
    _match,
    _resolve_plan,
    apply_tp,
    colwise,
    rowwise,
)
from hpmesh.trainer import ParallelConfig


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


def test_plan_resolution_prefers_the_tp_plan_property_over_the_attribute() -> None:
    """A wrapper that re-parents the model must win over the raw attribute.

    ``HFTransformerModel`` holds the HF model under ``self.model``, so its module
    paths carry a ``model.`` prefix the raw HF plan does not. Reading ``_tp_plan``
    off such a wrapper yields patterns that match nothing.
    """

    class Wrapper(nn.Module):
        _tp_plan = {"layers.*.q_proj": "colwise"}

        @property
        def tp_plan(self) -> dict[str, str]:
            return {"model.layers.*.q_proj": "colwise"}

    plan = _resolve_plan(Wrapper(), None)
    assert set(plan) == {"model.layers.*.q_proj"}


def test_a_wrapper_tp_plan_matches_the_modules_it_exposes() -> None:
    """The regression this locks in: TP matched 0 of 15 projections.

    Two things had to line up and neither did. The plan lives on the inner HF
    model, not the wrapper, so ``_resolve_plan`` found nothing; and even once
    found, HF's patterns are spelled relative to the HF model while the
    wrapper's ``named_modules`` paths sit under ``model.``. ``apply_tp`` then
    matched nothing and left the model replicated -- a TP run that silently
    trains a non-sharded model and looks like a success.

    Checked on the real wrapper so the path spelling is the real one.
    """
    from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config

    config = build_model_config(
        "llama",
        seq_len=32,
        arch_overrides={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )
    model = HFTransformerModel(config)
    plan = _resolve_plan(model, None)

    assert plan, "the wrapper exposed no usable TP plan"

    matched = [
        path
        for path, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and _match(plan, path) is not None
    ]
    # 7 projections per layer x 2 layers. ``lm_head`` is deliberately absent:
    # HF's plan does not shard it, which is why the vocab-parallel loss path
    # stays unreachable until a Shard(0) head plan lands.
    assert len(matched) == 14
    assert not any(path.endswith("lm_head") for path in matched)


def test_apply_tp_is_a_noop_when_tp_is_one() -> None:
    model = nn.Linear(4, 4)
    cfg = ParallelConfig()  # tp defaults to 1
    assert apply_tp(model, mesh=None, cfg=cfg) is model
    assert isinstance(model, nn.Linear)  # not swapped
