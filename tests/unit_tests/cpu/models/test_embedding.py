"""Vocab-parallel embedding: global ``padding_idx`` gradient suppression.

Upstream #4637 (``Fix padding gradients in vocab-parallel embeddings``):
``padding_idx`` is a *global* row id, so a shard that does not own that row must
not hand it to the local ``F.embedding`` -- doing so either crashes (the id is
beyond the shard's row count) or silently zeroes the gradient of an unrelated
row. Only the owning shard passes ``padding_idx - offset``.

Single-rank harness: the TP path only needs ``dist.get_world_size`` /
``dist.get_rank`` on a process-group handle, so a 1-rank fake pg plus patched
counts stands in for a real TP group.
"""

from tests.caps import require_env

require_env('spmd_types')


import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

from hpmesh.models.common import embedding as embedding_mod
from hpmesh.models.common.embedding import Embedding


def _single_rank_fake_pg():
    store = FakeStore()
    dist.init_process_group("fake", store=store, rank=0, world_size=1)
    return dist.new_group()


def test_owning_shard_freezes_only_its_local_padding_row(monkeypatch):
    group = _single_rank_fake_pg()
    try:
        monkeypatch.setattr(embedding_mod, "spmd_mesh_group", lambda axis: group)
        # tp group of size 1: offset 0, the shard owns every row, so a global
        # padding_idx of 7 maps to local row 7.
        monkeypatch.setattr(embedding_mod.dist, "get_world_size", lambda g: 1)
        monkeypatch.setattr(embedding_mod.dist, "get_rank", lambda g: 0)

        emb = Embedding(num_embeddings=8, embedding_dim=4, padding_idx=7)
        emb.weight = torch.nn.Parameter(
            torch.randn(8, 4, dtype=torch.float64), requires_grad=True
        )
        input_ids = torch.tensor([[3, 7, 1]])
        out = emb(input_ids)
        out.sum().backward()
        grad = emb.weight.grad
        # The padding row's gradient is suppressed ...
        assert grad[7].abs().sum() == 0
        # ... while the other rows still learn (non-vacuity).
        assert grad[3].abs().sum() > 0
        assert grad[1].abs().sum() > 0
    finally:
        dist.destroy_process_group()


def test_non_owning_shard_passes_no_local_padding_idx(monkeypatch):
    group = _single_rank_fake_pg()
    try:
        monkeypatch.setattr(embedding_mod, "spmd_mesh_group", lambda axis: group)
        # tp group of size 2, this rank is 1: vocab 8 -> chunk 4, offset 4,
        # local rows [4, 8). Global padding_idx 3 lives on rank 0.
        monkeypatch.setattr(embedding_mod.dist, "get_world_size", lambda g: 2)
        monkeypatch.setattr(embedding_mod.dist, "get_rank", lambda g: 1)

        emb = Embedding(num_embeddings=8, embedding_dim=4, padding_idx=3)
        emb.weight = torch.nn.Parameter(
            torch.randn(4, 4, dtype=torch.float64), requires_grad=True
        )
        # Before #4637 this raised "Padding_idx must be within
        # num_embeddings" (3 is outside a 4-row shard ... only because the
        # *global* index leaked in); here it must run and keep every local
        # row trainable.
        input_ids = torch.tensor([[6, 7]])
        out = emb(input_ids)
        out.sum().backward()
        grad = emb.weight.grad
        assert grad[2].abs().sum() > 0  # local row for global id 6
        assert grad[3].abs().sum() > 0  # local row for global id 7
    finally:
        dist.destroy_process_group()
