"""EP dispatch backend selection: gating, config validation, torchao adapter.

``torchao``/``deep_ep``/``hybridep`` are not installed in this environment, so
the optional-dependency seam is exercised against a fake ``torchao`` module
injected into ``sys.modules`` -- the fake reproduces the interface hpmesh calls
(``torchao.prototype.moe_training.ep.permute.permute_and_pad``, sentinel-row
padding semantics), not the real kernel's output. Numerical parity with real
torchao is unverified until the dependency lands and these tests are re-run
against it; the gating tests (no package installed) run for real.
"""

from tests.caps import require_env

require_env('spmd_types')


import sys
import types

import pytest
import torch

from hpmesh.config import ParallelConfig
from hpmesh.models.common.token_dispatcher import (
    TORCHAO_INSTALL_HINT,
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
    TorchAOTokenDispatcher,
)
from hpmesh.parallel.expert_parallel.swap import swap_hf_moe_blocks

# --------------------------------------------------------------------------
# config gating
# --------------------------------------------------------------------------


def test_default_dispatcher_is_alltoall():
    cfg = ParallelConfig()
    assert cfg.ep_token_dispatcher == "alltoall"
    assert cfg.ep_torchao_pad_multiple == 16


def test_unknown_dispatcher_value_rejected():
    with pytest.raises(ValueError, match="ep_token_dispatcher"):
        ParallelConfig(ep_token_dispatcher="fastep")


@pytest.mark.parametrize("backend", ["deepep", "hybridep"])
def test_cuda_only_backends_refused_with_unlock_conditions(backend):
    with pytest.raises(NotImplementedError, match="registered gap") as excinfo:
        ParallelConfig(expert_parallel_size=2, ep_token_dispatcher=backend)
    assert "alltoall" in str(excinfo.value)


def test_non_default_dispatcher_requires_ep():
    with pytest.raises(NotImplementedError, match="expert_parallel_size=1"):
        ParallelConfig(expert_parallel_size=1, ep_token_dispatcher="torchao")


def test_torchao_dispatcher_accepted_with_ep():
    cfg = ParallelConfig(expert_parallel_size=2, ep_token_dispatcher="torchao")
    assert cfg.ep_token_dispatcher == "torchao"


def test_pad_multiple_must_be_positive():
    with pytest.raises(ValueError, match="ep_torchao_pad_multiple"):
        ParallelConfig(
            expert_parallel_size=2,
            ep_token_dispatcher="torchao",
            ep_torchao_pad_multiple=0,
        )


# --------------------------------------------------------------------------
# torchao adapter: gating and fake-package interface
# --------------------------------------------------------------------------


def _fake_permute_and_pad(x, counts, ep_size, e, pad_multiple):
    """CPU stand-in for torchao's permute_and_pad (EP=1 semantics).

    Reproduces the contract the dispatcher relies on: tokens are already
    expert-sorted for a single rank, each expert's group is padded up to a
    multiple of ``pad_multiple`` with rows taken from a zero sentinel row
    appended at index R, and ``permuted_indices`` maps every padded output
    row back to its source row (padding rows point at the sentinel).
    """
    assert ep_size == 1
    counts = counts.tolist()
    assert len(counts) == e
    R, D = x.shape
    x_ext = torch.cat([x, x.new_zeros(1, D)], dim=0)
    indices = []
    pos = 0
    for c in counts:
        indices.extend(range(pos, pos + c))
        indices.extend([R] * (-c % pad_multiple))
        pos += c
    assert pos == R
    padded_counts = torch.tensor(
        [c + (-c % pad_multiple) for c in counts], dtype=torch.long
    )
    indices = torch.tensor(indices, dtype=torch.long)
    return x_ext.shape, x_ext[indices], indices, padded_counts, None


@pytest.fixture
def fake_torchao(monkeypatch):
    """Inject a fake ``torchao.prototype.moe_training.ep.permute`` chain."""
    for name in (
        "torchao",
        "torchao.prototype",
        "torchao.prototype.moe_training",
        "torchao.prototype.moe_training.ep",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    permute_mod = types.ModuleType("torchao.prototype.moe_training.ep.permute")
    permute_mod.permute_and_pad = _fake_permute_and_pad
    monkeypatch.setitem(
        sys.modules, "torchao.prototype.moe_training.ep.permute", permute_mod
    )
    return permute_mod


def test_torchao_dispatcher_without_the_package_raises_an_install_hint(monkeypatch):
    """Real environment check: ``torchao`` is not installed here."""
    monkeypatch.delitem(sys.modules, "torchao", raising=False)
    with pytest.raises(ImportError, match="pip install torchao"):
        TorchAOTokenDispatcher(num_experts=4, top_k=2, pad_multiple=16)
    assert "ep_token_dispatcher='torchao'" in TORCHAO_INSTALL_HINT


def test_torchao_dispatcher_rejects_bad_pad_multiple(fake_torchao):
    with pytest.raises(ValueError, match="pad_multiple"):
        TorchAOTokenDispatcher(num_experts=4, top_k=2, pad_multiple=0)


def _routing(num_tokens=6, dim=4, num_experts=4, top_k=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x_TD = torch.randn(num_tokens, dim, generator=g)
    topk_expert_ids_TK = torch.randint(0, num_experts, (num_tokens, top_k), generator=g)
    topk_scores_TK = torch.rand(num_tokens, top_k, generator=g)
    num_local_tokens_per_expert_E = torch.zeros(num_experts, dtype=torch.long)
    num_local_tokens_per_expert_E.scatter_add_(
        0,
        topk_expert_ids_TK.reshape(-1),
        torch.ones(num_tokens * top_k, dtype=torch.long),
    )
    return x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E


def test_torchao_ep1_dispatch_pads_groups_to_multiple(fake_torchao):
    x_TD, scores, ids, counts = _routing()
    dispatcher = TorchAOTokenDispatcher(num_experts=4, top_k=2, pad_multiple=4)

    routed, padded_counts, metadata = dispatcher.dispatch(x_TD, scores, ids, counts)

    assert padded_counts.tolist() == [
        int(c) + (-int(c) % 4) for c in counts.tolist()
    ]
    assert routed.shape[0] == int(padded_counts.sum())
    assert routed.shape[1] == x_TD.shape[1]
    # No EP group was wired: the all-to-all path must not have run.
    assert dispatcher.ep_group is None
    assert metadata.input_splits == [] and metadata.output_splits == []


def test_torchao_ep1_combine_matches_local_dispatcher(fake_torchao):
    x_TD, scores, ids, counts = _routing()
    torchao_dispatcher = TorchAOTokenDispatcher(num_experts=4, top_k=2, pad_multiple=4)
    local_dispatcher = LocalTokenDispatcher(num_experts=4, top_k=2)

    routed_padded, _, metadata = torchao_dispatcher.dispatch(x_TD, scores, ids, counts)
    routed_local, _, local_metadata = local_dispatcher.dispatch(
        x_TD, scores, ids, counts
    )

    # Any elementwise expert "computation" works; the padding rows ride along
    # and must be dropped by combine without disturbing the real rows.
    out_torchao = torchao_dispatcher.combine(routed_padded * 2.0, metadata, x_TD)
    out_local = local_dispatcher.combine(routed_local * 2.0, local_metadata, x_TD)
    torch.testing.assert_close(out_torchao, out_local)


def test_torchao_is_an_alltoall_dispatcher(fake_torchao):
    assert issubclass(TorchAOTokenDispatcher, AllToAllTokenDispatcher)


# --------------------------------------------------------------------------
# swap-level defensive gating (callers that bypass ParallelConfig)
# --------------------------------------------------------------------------


def test_swap_refuses_registered_gap_backends():
    # Backend gating runs before any model probing, so a stub model suffices.
    model = types.SimpleNamespace(layers=[])
    with pytest.raises(NotImplementedError, match="registered gap"):
        swap_hf_moe_blocks(model, token_dispatcher="deepep")


def test_swap_refuses_unknown_backend():
    model = types.SimpleNamespace(layers=[])
    with pytest.raises(ValueError, match="unknown ep_token_dispatcher"):
        swap_hf_moe_blocks(model, token_dispatcher="fastep")
