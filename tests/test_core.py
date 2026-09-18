"""Unit tests for the parts that run without a GPU / process group.

Covers the behaviors verified during prototyping: the world_size constraint,
deterministic synthetic data, DP batch slicing, and the model bundle. The config
is the grouped HybridMeshConfig; the flat view (cfg.dp, cfg.steps, ...) is what
the trainer/mesh/bundle read.
"""

from __future__ import annotations

import pytest
import torch

from hpmesh.bundle import build_bundle
from hpmesh.mesh import build_parallel_dims
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.trainer import HybridMeshConfig, ParallelArguments, TrainingArguments
from hpmesh.trainer.trainer import Trainer


def _cfg(**parallel_kw) -> HybridMeshConfig:
    return HybridMeshConfig(parallel=ParallelArguments(**parallel_kw))


def test_derive_dp_derives_from_world_size() -> None:
    cfg = _cfg(data_parallel_shard_degree=-1)
    assert cfg.derive_dp(world_size=8) == 8
    assert cfg.derive_dp(world_size=4) == 4


def test_derive_dp_rejects_inconsistent_degrees() -> None:
    cfg = _cfg(data_parallel_shard_degree=1)
    with pytest.raises(ValueError):
        cfg.derive_dp(world_size=2)


def test_derive_dp_rejects_indivisible_world() -> None:
    cfg = _cfg(data_parallel_shard_degree=-1, tensor_parallel_degree=3)
    with pytest.raises(ValueError):
        cfg.derive_dp(world_size=8)  # 8 % 3 != 0


def test_derive_dp_narrows_by_the_non_dp_degrees() -> None:
    # tp=2 consumes half the ranks; the rest are data-parallel.
    cfg = _cfg(data_parallel_shard_degree=-1, tensor_parallel_degree=2)
    assert cfg.derive_dp(world_size=8) == 4


def test_build_parallel_dims_resolves_against_world_size() -> None:
    # Single process -> no process group and no parallelism to describe.
    assert build_parallel_dims(HybridMeshConfig(), world_size=1) is None

    cfg = _cfg(data_parallel_shard_degree=-1, tensor_parallel_degree=2)
    pd = build_parallel_dims(cfg, world_size=8)
    assert isinstance(pd, ParallelDims)
    # tp=2 over 8 ranks leaves 4 for data parallelism; dp_shard=-1 resolves here.
    assert (pd.tp, pd.dp_shard) == (2, 4)


def test_derive_dp_matches_parallel_dims_resolution() -> None:
    # The config helper and the torchtitan class must agree, or the trainer and
    # the mesh would disagree about how many ranks go to data parallelism.
    cfg = _cfg(data_parallel_shard_degree=-1, tensor_parallel_degree=2)
    pd = build_parallel_dims(cfg, world_size=8)
    assert cfg.derive_dp(world_size=8) == pd.dp_shard


def test_cp_must_divide_seq_len() -> None:
    with pytest.raises(ValueError):
        HybridMeshConfig(
            parallel=ParallelArguments(context_parallel_degree=3),
            training=TrainingArguments(max_seq_len=64),
        )


def _bare_trainer(cfg: HybridMeshConfig) -> Trainer:
    """A Trainer with __init__ bypassed, for testing pure data helpers."""
    t = Trainer.__new__(Trainer)
    t.cfg = cfg
    return t


def test_synthetic_batch_is_deterministic() -> None:
    cfg = HybridMeshConfig(
        training=TrainingArguments(global_batch_size=8, max_seq_len=16, seed=42)
    )
    t = _bare_trainer(cfg)
    b1 = t._make_batch(step=0)
    b2 = t._make_batch(step=0)
    assert torch.equal(b1.input_ids, b2.input_ids)
    assert b1.input_ids.shape == (8, 16)


def test_dp_slice_partitions_global_batch() -> None:
    # Simulate 2 DP ranks without a process group by driving the slice math directly.
    cfg = HybridMeshConfig(
        training=TrainingArguments(global_batch_size=8, max_seq_len=16, seed=42)
    )
    t = _bare_trainer(cfg)
    batch = t._make_batch(step=0)
    per = cfg.global_batch_size // 2
    r0 = batch.input_ids[0:per]
    r1 = batch.input_ids[per : 2 * per]
    assert torch.equal(torch.cat([r0, r1]), batch.input_ids)


def test_build_bundle_offline_llama() -> None:
    cfg = HybridMeshConfig(training=TrainingArguments(seed=42))
    torch.manual_seed(cfg.seed)
    bundle = build_bundle(cfg, device=torch.device("cpu"))
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    loss = bundle.model(ids, ids.clone())
    assert loss.ndim == 0  # scalar loss
    assert float(loss) > 0
