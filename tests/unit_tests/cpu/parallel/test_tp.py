"""Tests for the tensor-parallel declaration layer and engine wiring.

The fused collectives (``AllGatherLinear`` / ``LinearReduceScatter``) run only
on CUDA symmetric memory; the functional-collective fallback that lets TP run
on CPU needs a real process group, so the end-to-end forward/backward
equivalence lives in ``tests/integration_tests/tp_equivalence.py`` (torchrun).
What is testable single-process here is everything around the collectives: the
declaration, the weight layout each kind produces, plan resolution, the
loud-raise guards, and that ``apply_tp`` leaves a model alone when TP is off.
"""

from __future__ import annotations

from tests.caps import require_env

require_env('spmd_types')


import pytest
import torch
import torch.nn as nn

from hpmesh.parallel.tensor_parallel import apply_tp
from hpmesh.parallel.tensor_parallel.tp import (
    ColwiseLinear,
    ColwiseLinearNoGather,
    RowwiseLinear,
    ShardingConfig,
    _match,
    _resolve_plan,
    colwise,
    rowwise,
)
from hpmesh.trainer import ParallelConfig


def test_declaration_factories_pick_the_right_realizer() -> None:
    assert colwise() == ShardingConfig(kind="colwise", implementation=ColwiseLinear)
    assert rowwise() == ShardingConfig(kind="rowwise", implementation=RowwiseLinear)


def test_colwise_layout_is_cut_on_output_features() -> None:
    # nn.Linear weight is [out, in]; colwise keeps that layout and cuts dim 0,
    # storing [out/tp, in] -- the w_shard_n = [N/R, K] contract AllGatherLinear
    # documents (the op self-transposes inside the GEMM). An earlier version
    # transposed first and cut the last dim, feeding the op [K, N/R] and
    # producing garbage on the fused path.
    W = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)  # out=4, in=8
    mod = ColwiseLinear(W, tp_size=2, tp_rank=0, group=None)
    assert mod.weight.shape == (2, 8)
    assert torch.equal(mod.weight, W[0:2])
    # rank 1 holds the other half
    other = ColwiseLinear(W, tp_size=2, tp_rank=1, group=None)
    assert torch.equal(other.weight, W[2:4])


def test_rowwise_layout_is_cut_on_input_features() -> None:
    W = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    mod = RowwiseLinear(W, tp_size=2, tp_rank=1, group=None)
    assert mod.weight.shape == (4, 4)
    assert torch.equal(mod.weight, W[:, 4:8])


def test_sharded_weights_reconstruct_the_original() -> None:
    W = torch.randn(6, 12)
    for cls in (ColwiseLinear, ColwiseLinearNoGather, RowwiseLinear):
        shards = [cls(W, tp_size=3, tp_rank=r, group=None).weight for r in range(3)]
        if cls is RowwiseLinear:
            rec = torch.cat(shards, dim=1)
        else:
            rec = torch.cat(shards, dim=0)
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
    from hpmesh.models.hf_factory import build_model_config
    from hpmesh.models.hf_wrapper import HFTransformerModel

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


def test_qwen3_plan_resolves_rather_than_raising_on_its_qk_norms() -> None:
    """The plan an hpmesh-supported family actually ships must resolve.

    Qwen3 -- the architecture the repo's own example trains -- marks
    ``q_norm`` / ``k_norm`` with HF's ``replicated_with_grad_allreduce``, a spec
    ``_resolve_plan`` had no branch for. Every Qwen3 TP run therefore died in
    ``apply_tp`` before touching a weight, and no test caught it because the
    only plan under test was a hand-written ``{colwise, rowwise}`` map.

    The two branches mean different things and both matter here: a norm entry
    resolves to ``None`` (left whole on every rank, its gradient summed by
    ``Trainer._allreduce_replicated_tp_grads``), while the projections still
    resolve to real realizers.
    """
    from hpmesh.models.hf_factory import build_model_config
    from hpmesh.models.hf_wrapper import HFTransformerModel

    config = build_model_config(
        "qwen3",
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

    # The plan is the real one, so the entry that used to raise is present.
    assert "replicated_with_grad_allreduce" in model.tp_plan.values()

    plan = _resolve_plan(model, None)  # must not raise

    norms = [p for p in plan if p.endswith(("q_norm", "k_norm"))]
    assert norms, "q_norm/k_norm are not in the plan -- this test is vacuous"
    assert all(plan[p] is None for p in norms), "a norm must not be sharded"

    matched = [
        path
        for path, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and _match(plan, path) is not None
    ]
    # 7 projections per layer x 2; the norms are neither nn.Linear nor sharded.
    assert len(matched) == 14
    # And the norms really are parameters of this model, so leaving them
    # unsharded is a decision about a module that exists.
    norm_params = [
        n
        for n, _ in model.named_parameters()
        if n.endswith(("q_norm.weight", "k_norm.weight"))
    ]
    assert norm_params


def test_apply_tp_is_a_noop_when_tp_is_one() -> None:
    model = nn.Linear(4, 4)
    cfg = ParallelConfig()  # tp defaults to 1
    assert apply_tp(model, mesh=None, cfg=cfg) is model
    assert isinstance(model, nn.Linear)  # not swapped


def test_apply_tp_raises_when_the_model_has_no_plan() -> None:
    """TP with no plan must refuse, not silently run replicated.

    The raise fires before the mesh is touched (plan validation comes first),
    so a sentinel stands in for the mesh here -- reaching for it would prove
    the guard did not fire.
    """
    cfg = ParallelConfig(tensor_parallel_size=2)
    with pytest.raises(ValueError, match="no TP plan"):
        apply_tp(nn.Linear(4, 4), mesh=object(), cfg=cfg)


def test_apply_tp_raises_when_the_plan_matches_no_module() -> None:
    """A plan that matches nothing is the same silent-replication failure."""

    class Model(nn.Module):
        _tp_plan = {"*.q_proj": "colwise"}

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

    cfg = ParallelConfig(tensor_parallel_size=2)
    with pytest.raises(ValueError, match="matched no nn.Linear"):
        apply_tp(Model(), mesh=object(), cfg=cfg)
