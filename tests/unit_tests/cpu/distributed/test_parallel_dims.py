"""``ParallelDims`` and the config layer that resolves it: guards, not happy path.

Two things live here. The first is the derivation chain a config walks before a
process group exists -- ``ParallelConfig.derive_dp`` -> ``build_parallel_dims``
-> ``ParallelDims`` -- where a mis-sized launch has to fail with a config-level
message rather than deep inside mesh construction. The second is the part a
successful run never executes: the assignment guards that reject an inconsistent
size, and the two mesh lookups that fail *because* an axis is disabled. Those are
the paths that turn a misconfigured run into a stack trace instead of a silent
wrong-size process group.

``build_mesh`` is what needs a real process group, so the mesh tests run behind a
single-rank gloo group -- the same fixture shape as ``test_pipeline.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hpmesh.mesh import build_parallel_dims
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.trainer import HybridMeshConfig, ParallelConfig


def _config(**parallel_kw) -> HybridMeshConfig:
    return HybridMeshConfig(parallel=ParallelConfig(**parallel_kw))


@pytest.fixture(scope="module")
def single_rank_group(tmp_path_factory):
    """A size-1 gloo group: enough for ``init_device_mesh``, no p2p traffic."""
    created = not dist.is_initialized()
    if created:
        store = dist.FileStore(str(tmp_path_factory.mktemp("pg") / "store"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    yield
    if created:
        dist.destroy_process_group()


def _dims(world_size: int = 8, **overrides) -> ParallelDims:
    fields = {
        "dp_replicate": 1,
        "dp_shard": 8,
        "cp": 1,
        "tp": 1,
        "pp": 1,
        "ep": 1,
        "world_size": world_size,
        **overrides,
    }
    return ParallelDims(**fields)


# -- assignment guards -------------------------------------------------------


def test_a_product_that_misses_the_world_size_is_rejected() -> None:
    """The one arithmetic mistake every parallelism config makes at some point."""
    with pytest.raises(AssertionError, match="Invalid parallel dims"):
        _dims(world_size=8, dp_replicate=2, dp_shard=2, tp=2)  # 2*2*2 = 8, ok
        _dims(world_size=7, dp_replicate=2, dp_shard=2, tp=2)  # ...but the world is 7


def test_ep_must_divide_the_sparse_region() -> None:
    """EP shards the experts over dp_shard * cp * tp; a remainder is unwired."""
    # dp_shard * cp * tp == 8, so ep=3 cannot tile it.
    with pytest.raises(ValueError, match=r"must divide"):
        _dims(world_size=8, dp_shard=8, ep=3)


def test_ep_that_divides_is_accepted() -> None:
    """The non-vacuity check for the guard above: ep=4 divides 8."""
    dims = _dims(world_size=8, dp_shard=8, ep=4)
    assert dims.ep == 4
    # The sparse region EP tiles over: dp_shard * cp * tp.
    assert dims.dp_shard * dims.cp * dims.tp == 8


def test_a_non_positive_size_is_rejected() -> None:
    """A zero size would divide by zero deep in the mesh builder."""
    with pytest.raises(AssertionError):
        _dims(world_size=8, dp_shard=0)


# -- mesh resolution ---------------------------------------------------------


def test_a_disabled_axis_resolves_to_none_rather_than_a_size_one_mesh(
    single_rank_group,
) -> None:
    """``get_optional_mesh`` is how components ask "is this parallelism on?".

    Returning a size-1 mesh instead of ``None`` would make every
    ``if mesh is None`` branch in ``apply_*`` dead, and the collectives would
    run over a degenerate group.
    """
    dims = _dims(world_size=1, dp_shard=1, tp=1)
    dims.build_mesh()

    assert dims.get_optional_mesh("tp") is None
    assert dims.get_optional_mesh("cp") is None
    # dp_shard is deliberately always alive (fully_shard installs the
    # MixedPrecisionPolicy through it), so it is the one axis that is not None.
    assert dims.get_optional_mesh("dp_shard") is not None


def test_get_mesh_raises_for_a_disabled_axis_but_names_the_reason(
    single_rank_group,
) -> None:
    """ "Not available" must distinguish "off" from "misspelled"."""
    dims = _dims(world_size=1, dp_shard=1)
    dims.build_mesh()

    with pytest.raises(ValueError, match="is not available"):
        dims.get_mesh("tp")


def test_an_unknown_axis_name_lists_the_valid_ones(single_rank_group) -> None:
    """A typo should be self-correcting, not a bare IndexError."""
    dims = _dims(world_size=1, dp_shard=1)
    dims.build_mesh()

    with pytest.raises(ValueError, match="Invalid mesh dim"):
        dims.get_optional_mesh("tensor_parallel")


def test_resolving_to_a_mesh_that_covers_the_world_is_the_single_axis_case(
    single_rank_group,
) -> None:
    """``dp_shard`` on a 1-rank world: the axis is live at size 1 by design."""
    dims = _dims(world_size=1, dp_shard=1)
    dims.build_mesh()
    mesh = dims.get_optional_mesh("dp_shard")
    assert mesh is not None
    assert mesh.size() == 1


def test_a_multi_axis_request_returns_one_mesh_from_the_cache(
    single_rank_group,
) -> None:
    """Multi-axis lookups are cached by name-tuple, so identity must hold."""
    dims = _dims(world_size=1, dp_shard=1)
    dims.build_mesh()
    init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))
    # A second call must hand back the same object, not rebuild it.
    first = dims.get_optional_mesh(["dp_shard"])
    second = dims.get_optional_mesh(["dp_shard"])
    assert first is second


# -- config resolution: derive_dp and build_parallel_dims ---------------------


def test_derive_dp_derives_from_world_size() -> None:
    """``-1`` means "use whatever is left", not "1"."""
    cfg = _config(data_parallel_shard_size=-1)
    assert cfg.derive_dp(world_size=8) == 8
    assert cfg.derive_dp(world_size=4) == 4


def test_derive_dp_rejects_inconsistent_sizes() -> None:
    """A pinned dp_shard that does not multiply out to world_size is a launch bug."""
    cfg = _config(data_parallel_shard_size=1)
    with pytest.raises(ValueError):
        cfg.derive_dp(world_size=2)


def test_derive_dp_rejects_indivisible_world() -> None:
    cfg = _config(data_parallel_shard_size=-1, tensor_parallel_size=3)
    with pytest.raises(ValueError):
        cfg.derive_dp(world_size=8)  # 8 % 3 != 0


def test_derive_dp_narrows_by_the_non_dp_sizes() -> None:
    """tp=2 consumes half the ranks; the rest are data-parallel."""
    cfg = _config(data_parallel_shard_size=-1, tensor_parallel_size=2)
    assert cfg.derive_dp(world_size=8) == 4


def test_build_parallel_dims_resolves_against_world_size() -> None:
    """Single process -> no process group, so there is no ``ParallelDims`` at all.

    ``None`` is the sentinel every downstream ``parallel_dims is None`` guard
    keys on, so returning a degenerate object here would make them all dead.
    """
    assert build_parallel_dims(HybridMeshConfig(), world_size=1) is None

    cfg = _config(data_parallel_shard_size=-1, tensor_parallel_size=2)
    pd = build_parallel_dims(cfg, world_size=8)
    assert isinstance(pd, ParallelDims)
    # tp=2 over 8 ranks leaves 4 for data parallelism; dp_shard=-1 resolves here.
    assert (pd.tp, pd.dp_shard) == (2, 4)


def test_derive_dp_matches_parallel_dims_resolution() -> None:
    """The config helper and the torchtitan-shaped class must agree.

    They compute the same number by different routes -- the config by dividing
    world_size itself, ``ParallelDims`` inside ``_validate`` -- so a drift
    between them would make the trainer and the mesh disagree about how many
    ranks go to data parallelism, silently.
    """
    cfg = _config(data_parallel_shard_size=-1, tensor_parallel_size=2)
    pd = build_parallel_dims(cfg, world_size=8)
    assert cfg.derive_dp(world_size=8) == pd.dp_shard


def test_cp_only_still_needs_a_loss_reduction() -> None:
    """cp > 1 with dp = 1 shards the sequence but leaves the DP axis empty.

    The loss is summed over each rank's own *slice* of the sequence, so with
    only CP on, a dp-only reduction would be over a size-1 group: every rank
    would report its own shard's loss as the whole batch's. The trainer
    therefore gates the loss reduce-group on ``dp_cp_enabled`` (dp *or* cp)
    rather than on how dense the mesh is.

    Only the flags are asserted here -- the group sizes they select need a live
    process group (``get_optional_mesh`` builds meshes). The sizes themselves
    are pinned by the ``expected_sizes`` table in ``parallel_dims.py``, which
    is what makes the property sufficient: ``loss`` is defined there as
    ``dp_replicate * dp_shard * cp``, so choosing it is choosing a group that
    spans the cp axis. The end-to-end version runs under torchrun in
    ``tests/integration_tests/cp_wiring_equivalence.py``.
    """
    cfg = _config(data_parallel_shard_size=1, context_parallel_size=2)
    pd = build_parallel_dims(cfg, world_size=2)
    assert isinstance(pd, ParallelDims)

    assert pd.cp_enabled
    assert not pd.dp_enabled
    # The property the trainer gates on: either axis alone is enough. Gating on
    # cp alone (or on dp alone) is the bug this pins.
    assert pd.dp_cp_enabled


# -- tensor-parallel declaration layer ---------------------------------------


def test_sharding_a_weight_that_does_not_divide_is_rejected() -> None:
    """A ragged split would silently give ranks different-size shards."""
    from hpmesh.parallel.tensor_parallel.tp import _shard_weight

    with pytest.raises(ValueError, match="not divisible by"):
        _shard_weight(torch.zeros(5, 4), 0, tp_size=2, tp_rank=0)


def test_sharding_a_weight_keeps_only_this_ranks_slice() -> None:
    """The non-vacuity check: rank 1 of 2 gets the second half along dim 0."""
    from hpmesh.parallel.tensor_parallel.tp import _shard_weight

    weight = torch.arange(8).reshape(4, 2)
    assert _shard_weight(weight, 0, tp_size=2, tp_rank=1).tolist() == [[4, 5], [6, 7]]


def test_an_unknown_shard_kind_is_rejected() -> None:
    """``kind`` selects the collective the wrapper installs; a typo is fatal."""
    import torch.nn as nn

    from hpmesh.parallel.tensor_parallel.tp import ShardingConfig

    with pytest.raises(ValueError, match="Unknown shard kind"):
        ShardingConfig(kind="diagonal", implementation=nn.Linear)
