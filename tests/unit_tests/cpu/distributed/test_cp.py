"""Context-parallel attention: the single-rank guards and the kernel's setup.

The redistributions themselves need two ranks and live in
``tests/cp_equivalence.py`` and ``cp_wiring_equivalence.py``. What can be checked
without a mesh is where the module is *explicit*: the strategy dispatch, the
refusals that carry the whole safety argument, and the ulysses head-divisibility
check. The refusal matters more here than anywhere else in hpmesh -- every other
component degrades gracefully when a parallelism axis is off, but a CP
redistribution cannot, because skipping it does not skip work, it changes the
answer.

``apply_cp``'s argument validation runs before any collective, so most of the
kernel-facing checks below stand a size-1 "fake" process group up instead of a
real two-rank one; a genuinely multi-rank redistribution is the equivalence
harnesses' job.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from hpmesh.parallel.context_parallel import (
    CPFlexKernel,
    apply_cp,
)
from hpmesh.trainer import ParallelConfig


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


# -- the kernel's strategy dispatch -------------------------------------------


@pytest.fixture
def fake_cp_mesh():
    """A size-1 CP axis over the 'fake' backend -- enough to construct a kernel.

    ``CPFlexKernel.__init__`` captures the process group and the set of allowed
    strategies before any collective runs, so building one needs a mesh but not
    a second rank.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    store = FakeStore()
    dist.init_process_group("fake", store=store, rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("cp",))["cp"]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strategy", ["kv_allgather", "ulysses"])
def test_the_known_strategies_are_accepted(fake_cp_mesh, strategy: str) -> None:
    """Both strategies the package advertises must actually construct.

    Non-vacuity: this is the accepting half of the pair below, so a kernel that
    rejected everything cannot pass both.
    """
    kernel = CPFlexKernel(cp_mesh=fake_cp_mesh, strategy=strategy)

    assert kernel.strategy == strategy


def test_an_unknown_strategy_is_refused(fake_cp_mesh) -> None:
    """A typo'd strategy must fail at attach time, not silently pick one path.

    ``forward`` branches on the name, so an unrecognized value would fall
    through to the KV all-gather branch and train a *different* CP formulation
    than the one asked for, with the forward still producing plausible numbers.
    """
    with pytest.raises(NotImplementedError, match="not wired"):
        CPFlexKernel(cp_mesh=fake_cp_mesh, strategy="ring")


def test_the_default_strategy_is_kv_allgather(fake_cp_mesh) -> None:
    """The default has to match ``ParallelConfig``'s, or a caller that relies on
    both defaults gets whichever the other object happened to pick."""
    assert ParallelConfig().context_parallel_strategy == "kv_allgather"
    assert CPFlexKernel(cp_mesh=fake_cp_mesh).strategy == "kv_allgather"


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

        # Two builds: mode remains part of the cache key on every supported
        # torch. Newer torch releases expose ``separate_full_blocks`` and must
        # receive the wrapper's matching choice; torch 2.10 removed the knob,
        # so the compatibility path cannot pass it.
        assert len(calls) == 2
        if "separate_full_blocks" in inspect.signature(
            real_create_block_mask
        ).parameters:
            assert [c["separate_full_blocks"] for c in calls] == [True, False]
        else:
            assert all("separate_full_blocks" not in c for c in calls)
    finally:
        dist.destroy_process_group()
