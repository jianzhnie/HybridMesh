"""EFSDP expert-placement boundary: Shard(0) vs Shard(1) on the expert dim.

``apply_fsdp_to_decoder`` picks ``Shard(1)`` (shard the expert *features*)
over ``Shard(0)`` (shard the experts) when ``efsdp * ep > num_experts`` -- the
FSDP degree over the sparse region exceeds the expert count, so sharding the
expert axis would pad. Upstream compares against the TOTAL expert count; the
EP swap builds ``GroupedExperts`` per rank, so its ``num_experts`` is the local
shard (total / ep), and reading it instead makes the condition
``efsdp * ep**2 > total`` -- ep times stricter, silently picking Shard(1)
where Shard(0) belongs (numerically correct, but the wrong placement).

The decision is evaluated here through the real ``shard_placement_fn`` the
function builds, captured via a ``fully_shard`` spy. The per-param mesh path
itself cannot run on one rank: size-1 gloo meshes share a single process
group, which FSDP2 rejects for mixed mesh infos -- so the spy strips the
callback before delegating, and the callback is exercised directly.

Boundary pinned (efsdp axis size 1 on a single-rank mesh, so
``efsdp_ep_size == ep_size``):

* ``ep * efsdp == total`` -> Shard(0) (equality shards the experts);
* ``ep * efsdp < total``  -> Shard(0);
* ``ep * efsdp > total``  -> Shard(1) (padding avoidance).
"""

from __future__ import annotations

from tests.caps import require_env

require_env('dtensor', 'spmd_types')


import math

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

import hpmesh.parallel.fully_shard.fsdp as fsdp_module
from hpmesh.models.common.grouped_experts import GroupedExperts
from hpmesh.models.common.moe import MoE, RoutedExperts, TokenChoiceTopKRouter
from hpmesh.models.common.token_dispatcher import LocalTokenDispatcher
from hpmesh.parallel.fully_shard.fsdp import (
    _fsdp_shard_degree,
    apply_fsdp_to_decoder,
    enable_fsdp_symm_mem,
)

_DIM = 8
_HIDDEN = 16


@pytest.fixture(scope="module")
def single_rank_group(tmp_path_factory):
    """A size-1 gloo group: enough for ``init_device_mesh``, no collectives."""
    created = not dist.is_initialized()
    if created:
        store = dist.FileStore(str(tmp_path_factory.mktemp("pg") / "store"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    yield
    if created:
        dist.destroy_process_group()


class _SparseBlock(nn.Module):
    """The shape the swap leaves behind: MoE under ``mlp``, flagged for FSDP."""

    def __init__(self, total: int, local: int) -> None:
        super().__init__()
        self.attn = nn.Linear(_DIM, _DIM)
        moe = MoE(
            num_experts=total,
            routed_experts=RoutedExperts(
                GroupedExperts(_DIM, _HIDDEN, local),
                LocalTokenDispatcher(total, 1),
            ),
            router=TokenChoiceTopKRouter(total, _DIM, top_k=1),
            load_balance_coeff=None,
        )
        self.mlp = moe
        self.moe_enabled = True
        # Not registered: the same module already lives under ``mlp``, exactly
        # like the swap's ``object.__setattr__`` (double registration would
        # duplicate every expert weight in the state dict).
        object.__setattr__(self, "moe", moe)


class _ToyModel(nn.Module):
    """The five attributes ``apply_fsdp_to_decoder`` reads off a model."""

    def __init__(self, total: int, local: int) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(32, _DIM)
        self.layers = nn.ModuleList([_SparseBlock(total, local)])
        self.norm = nn.LayerNorm(_DIM)
        self.lm_head = nn.Linear(_DIM, 32, bias=False)
        self.enable_weight_tying = False


def _expert_placement(
    monkeypatch: pytest.MonkeyPatch, *, total: int, local: int, ep_size: int
) -> Shard:
    """The placement the MoE branch chooses for the expert weights."""
    model = _ToyModel(total, local)
    grouped = model.layers[0].mlp.routed_experts.inner_experts
    # Non-vacuity: the grouped weights really do hold only the local shard --
    # the count the buggy comparison read.
    assert grouped.num_experts == local < total
    # Captured BEFORE wrapping: ``fully_shard`` replaces the parameter objects,
    # and the placement fn matches by identity against the pre-wrap set.
    expert_param = next(iter(grouped.parameters()))

    captured: dict[str, object] = {}
    real_fully_shard = fsdp_module.fully_shard

    def _spy(module, **kwargs):
        placement_fn = kwargs.pop("shard_placement_fn", None)
        if placement_fn is not None:
            captured["fn"] = placement_fn
        return real_fully_shard(module, **kwargs)

    monkeypatch.setattr(fsdp_module, "fully_shard", _spy)

    dp_mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))
    edp_mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("efsdp",))
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        torch.float32,
        torch.float32,
        pp_enabled=False,
        ep_size=ep_size,
        edp_mesh=edp_mesh,
    )

    assert "fn" in captured, "the MoE branch never built a placement function"
    return captured["fn"](expert_param).placement


def test_equal_degree_and_experts_shards_the_expert_axis(
    single_rank_group, monkeypatch
) -> None:
    """``efsdp * ep == total``: Shard(0) exactly fits; the shard-count read
    (local = total / ep) would turn ``>`` true and mis-pick Shard(1)."""
    placement = _expert_placement(monkeypatch, total=2, local=1, ep_size=2)
    assert placement == Shard(0)


def test_degree_below_experts_shards_the_expert_axis(
    single_rank_group, monkeypatch
) -> None:
    """``efsdp * ep < total``: Shard(0); the buggy read flips this to Shard(1)
    as soon as ``efsdp * ep`` clears total / ep."""
    placement = _expert_placement(monkeypatch, total=8, local=2, ep_size=4)
    assert placement == Shard(0)


def test_degree_above_experts_shards_the_feature_axis(
    single_rank_group, monkeypatch
) -> None:
    """``efsdp * ep > total``: Shard(1) avoids padding the expert axis. Not a
    legal EP configuration (ep must divide total); it pins the far side of
    the comparison, where the buggy read agrees."""
    placement = _expert_placement(monkeypatch, total=2, local=1, ep_size=4)
    assert placement == Shard(1)


# -- FSDP shard degree over the dense mesh -------------------------------------


class _MeshStub:
    """Stand-in for the rebuilt FSDP mesh; multi-axis real meshes need a
    multi-rank process group, which a CPU test does not have."""

    def __init__(self, axes: dict[str, int]) -> None:
        self._axes = axes

    @property
    def mesh_dim_names(self) -> tuple[str, ...]:
        return tuple(self._axes)

    def size(self) -> int:
        return math.prod(self._axes.values())

    def __getitem__(self, name: str) -> _MeshStub:
        return _MeshStub({name: self._axes[name]})


def test_shard_degree_is_the_shard_axis_when_no_replication() -> None:
    assert _fsdp_shard_degree(_MeshStub({"dp_shard": 4})) == 4


def test_shard_degree_folds_cp_into_the_shard_axis() -> None:
    """``resolve_fsdp_mesh`` flattens (dp_shard, cp) into ``dp_shard_cp``."""
    assert _fsdp_shard_degree(_MeshStub({"dp_shard_cp": 8})) == 8


def test_shard_degree_excludes_dp_replicate() -> None:
    """HSDP: dp_replicate=2, dp_shard=4, num_experts=4 must compare 4, not 8 --
    counting the replicate axis mis-picks Shard(1) for the experts."""
    assert _fsdp_shard_degree(_MeshStub({"dp_replicate": 2, "dp_shard": 4})) == 4
    assert _fsdp_shard_degree(_MeshStub({"dp_replicate": 2, "dp_shard_cp": 8})) == 8


# -- symmetric-memory scope -----------------------------------------------------


def test_symm_mem_none_scope_is_a_noop() -> None:
    enable_fsdp_symm_mem(nn.Linear(4, 4), None)


def test_symm_mem_rejects_an_unknown_scope() -> None:
    with pytest.raises(ValueError, match="scope"):
        enable_fsdp_symm_mem(nn.Linear(4, 4), "sparse")
