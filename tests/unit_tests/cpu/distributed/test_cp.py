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


# -- ulysses head divisibility -------------------------------------------------
#
# The check lives behind mesh validation, so it needs a CP axis of size 2 --
# which a single process cannot build for real. The "fake" backend stands the
# mesh up without collectives, and the ulysses guards raise before any
# collective would run.


class _StubModel(torch.nn.Module):
    """Just enough of HFTransformerModel for apply_cp's validation path."""

    def __init__(self, num_attention_heads: int, num_key_value_heads: int) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.model = SimpleNamespace(
            config=SimpleNamespace(
                _attn_implementation="flex_torchtitan",
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
        )
        self.layers = torch.nn.ModuleList()

    def set_cp_mesh(self, cp_mesh, load_balancer=None) -> None:
        pass


@pytest.fixture
def tp_cp_mesh():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    store = FakeStore()
    dist.init_process_group("fake", store=store, rank=0, world_size=4)
    try:
        yield init_device_mesh("cpu", (2, 2), mesh_dim_names=("tp", "cp"))
    finally:
        dist.destroy_process_group()


def _ulysses_cfg(tp: int) -> ParallelConfig:
    return ParallelConfig(
        tensor_parallel_size=tp,
        context_parallel_size=2,
        context_parallel_strategy="ulysses",
        context_parallel_load_balancer=None,
        backend="gloo",
    )


def test_ulysses_heads_must_divide_tp_times_cp(tp_cp_mesh) -> None:
    """TP shards heads first: 4 heads over tp=2 leaves 2 local, which cp=2
    divides -- but the same 4 heads over tp=1 must still pass, and 2 heads
    over tp=2 (1 local) must refuse even though 2 % cp == 0."""
    model = _StubModel(num_attention_heads=4, num_key_value_heads=4)
    apply_cp(model, tp_cp_mesh, _ulysses_cfg(tp=2))

    model = _StubModel(num_attention_heads=2, num_key_value_heads=2)
    with pytest.raises(ValueError, match=r"tp\*cp"):
        apply_cp(model, tp_cp_mesh, _ulysses_cfg(tp=2))


def test_ulysses_kv_heads_use_the_local_count_too(tp_cp_mesh) -> None:
    """num_key_value_heads falls under the same local-head rule."""
    model = _StubModel(num_attention_heads=8, num_key_value_heads=2)
    with pytest.raises(ValueError, match="num_key_value_heads"):
        apply_cp(model, tp_cp_mesh, _ulysses_cfg(tp=2))


# -- the ulysses full-length mask ----------------------------------------------


def test_full_length_mask_tracks_batch_invariant_mode(monkeypatch) -> None:
    """The kernel's rebuilt mask must take the wrapper's separate_full_blocks
    choice, or ulysses decomposes the mask differently from every other path.

    Spied at the ``create_block_mask`` call site; a size-1 fake CP group is
    enough because mask building runs before any collective.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    from hpmesh.parallel.context_parallel.cp_kernel import CPFlexKernel
    from hpmesh.utils.batch_invariant import set_batch_invariant_mode

    store = FakeStore()
    dist.init_process_group("fake", store=store, rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("cp",))
        kernel = CPFlexKernel(cp_mesh=mesh["cp"], strategy="ulysses")

        import torch.nn.attention.flex_attention as flex

        calls = []
        real_create_block_mask = flex.create_block_mask

        def _spy(*args, **kwargs):
            calls.append(kwargs)
            return real_create_block_mask(*args, **kwargs)

        monkeypatch.setattr(flex, "create_block_mask", _spy)

        q = torch.randn(1, 1, 256, 8, dtype=torch.float64)
        set_batch_invariant_mode(False)
        try:
            kernel._full_length_causal_mask(q)
            set_batch_invariant_mode(True)
            kernel._full_length_causal_mask(q)
        finally:
            set_batch_invariant_mode(False)

        # Two builds (the mode is part of the cache key), each mirroring the
        # wrapper's ``separate_full_blocks=not is_in_batch_invariant_mode()``.
        assert [c["separate_full_blocks"] for c in calls] == [True, False]
    finally:
        dist.destroy_process_group()
