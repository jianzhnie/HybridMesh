"""``ParallelDims`` assignment and mesh resolution: the guards, not the happy path.

The derivation arithmetic (``dp == world_size / (cp * tp * pp)`` and friends)
already has coverage in ``test_core.py``. What is pinned here is the part a
successful run never executes: the checks that reject an inconsistent
assignment, and the two mesh lookups that fail *because* an axis is disabled.
Those are the paths that turn a misconfigured run into a stack trace instead of
a silent wrong-size process group, and nothing was executing them.

``build_mesh`` is what needs a real process group, so the mesh tests run behind a
single-rank gloo group -- the same fixture shape as ``test_pipeline.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hpmesh.parallel.parallel_dims import ParallelDims


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


def test_a_non_positive_degree_is_rejected() -> None:
    """A zero degree would divide by zero deep in the mesh builder."""
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


# -- pipeline stage arithmetic ------------------------------------------------


def test_the_smallest_legal_stage_count_leaves_one_effective_layer_each() -> None:
    """``num_stages == num_effective_layers`` is the boundary, and it must work.

    One stage below this is rejected by the ``num_stages > num_effective_layers``
    guard, so this is the tightest split the arithmetic has to survive.
    """
    from hpmesh.parallel.pipeline_parallel.pipeline import (
        generate_llm_fqn_per_model_part,
    )

    stages = generate_llm_fqn_per_model_part(
        num_stages=4, num_layers=2, input_weight=1, output_weight=1
    )
    assert len(stages) == 4


def test_more_stages_than_effective_layers_is_rejected() -> None:
    """The guard that makes the ``layers_per_stage == 0`` branch unreachable."""
    from hpmesh.parallel.pipeline_parallel.pipeline import (
        generate_llm_fqn_per_model_part,
    )

    with pytest.raises(ValueError, match="cannot be greater than effective"):
        generate_llm_fqn_per_model_part(
            num_stages=8, num_layers=2, input_weight=1, output_weight=1
        )


def test_a_weighted_module_must_fit_inside_one_stage() -> None:
    """A stage holding only the embedding is a stage with no transformer layers."""
    from hpmesh.parallel.pipeline_parallel.pipeline import (
        generate_llm_fqn_per_model_part,
    )

    with pytest.raises(ValueError, match="input_weight .* exceeds minimum"):
        generate_llm_fqn_per_model_part(
            num_stages=3, num_layers=1, input_weight=5, output_weight=1
        )
    with pytest.raises(ValueError, match="output_weight .* exceeds minimum"):
        generate_llm_fqn_per_model_part(
            num_stages=3, num_layers=1, input_weight=1, output_weight=5
        )


def test_a_zero_stage_count_is_rejected() -> None:
    from hpmesh.parallel.pipeline_parallel.pipeline import (
        generate_llm_fqn_per_model_part,
    )

    with pytest.raises(ValueError, match="at least 1"):
        generate_llm_fqn_per_model_part(num_stages=0, num_layers=2)
