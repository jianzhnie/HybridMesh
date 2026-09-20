"""Context-parallel attention: the single-rank guards.

The redistributions themselves need two ranks and live in
``tests/cp_equivalence.py``. What can be checked without a mesh is the refusal
path -- and that path carries the whole safety argument for this module. Every
other hpmesh component degrades gracefully when a parallelism axis is off; a CP
redistribution cannot, because skipping it does not skip work, it changes the
answer. A rank that never gathers K/V attends its query shard against itself.

So the contract these tests pin is narrow and deliberate: with CP off, refuse.
"""

from __future__ import annotations

import pytest
import spmd_types as spmd
import torch

from hpmesh.parallel.context_parallel import (
    HEAD_DIM,
    TOKEN_DIM,
    KVAllGatherContextParallel,
    UlyssesContextParallel,
    apply_cp,
    cp_group,
    cp_redistribute,
)
from hpmesh.trainer import ParallelConfig
from hpmesh.utils.spmd_context import set_current_spmd_mesh


def _x() -> torch.Tensor:
    return torch.randn(4, 2, 8)


def test_no_cp_mesh_means_no_group() -> None:
    with set_current_spmd_mesh(None):
        assert cp_group() is None


def test_redistribute_refuses_without_a_cp_mesh() -> None:
    """Silently returning ``x`` here would train on wrong attention."""
    with set_current_spmd_mesh(None):
        with pytest.raises(RuntimeError, match="multi-rank CP mesh"):
            cp_redistribute(_x(), src=spmd.S(TOKEN_DIM), dst=spmd.R)


def test_redistribute_refuses_for_the_head_conversion_too() -> None:
    """Both directions of the all-to-all are guarded, not just the first."""
    with set_current_spmd_mesh(None):
        with pytest.raises(RuntimeError, match="multi-rank CP mesh"):
            cp_redistribute(_x(), src=spmd.S(HEAD_DIM), dst=spmd.S(TOKEN_DIM))


def test_kv_all_gather_refuses_without_a_cp_mesh() -> None:
    kv = KVAllGatherContextParallel()
    with set_current_spmd_mesh(None):
        with pytest.raises(RuntimeError):
            kv(_x(), _x(), _x())


def test_ulysses_shard_refuses_without_a_cp_mesh() -> None:
    ulysses = UlyssesContextParallel()
    with set_current_spmd_mesh(None):
        with pytest.raises(RuntimeError):
            ulysses.shard(_x(), _x(), _x())


def test_ulysses_unshard_refuses_without_a_cp_mesh() -> None:
    ulysses = UlyssesContextParallel()
    with set_current_spmd_mesh(None):
        with pytest.raises(RuntimeError):
            ulysses.unshard(_x())


def test_default_reduce_dtype_is_float32() -> None:
    """Upstream's default; bf16 is available but must be opted into."""
    assert KVAllGatherContextParallel().reduce_dtype == torch.float32
    assert KVAllGatherContextParallel(reduce_dtype=torch.bfloat16).reduce_dtype == (
        torch.bfloat16
    )


def test_apply_cp_is_a_no_op_when_cp_is_off() -> None:
    """CP=1 must hand the model back untouched rather than raise."""
    model = torch.nn.Linear(4, 4)
    cfg = ParallelConfig(context_parallel_size=1)

    assert apply_cp(model, None, cfg) is model


def test_apply_cp_requires_the_flex_backend() -> None:
    """CP expresses the sharded mask as a BlockMask, which only flex consumes.

    A plain module has no HF config at all, so it fails the backend check --
    the same check that stops a CPU-run model (whose wrapper selected 'sdpa')
    from silently computing attention over its own shard only.
    """

    cfg = ParallelConfig(context_parallel_size=2)

    with pytest.raises(RuntimeError, match="attention backend"):
        apply_cp(torch.nn.Linear(4, 4), None, cfg)
