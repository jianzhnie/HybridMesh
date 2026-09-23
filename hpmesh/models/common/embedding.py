"""Vocab-parallel embedding.

Vendored from torchtitan ``models/common/embedding.py``. The only change is that
the nested ``Config`` is gone: hpmesh constructs the module directly, so there is
nothing to build from. The forward is unchanged.

Why it exists at all: when the TP axis shards the embedding weight on its vocab
dim (``Shard(0)`` on ``tok_embeddings``), HF's plain ``nn.Embedding.forward``
would index into a *local* weight with *global* token ids. This override applies
the vocab offset ``tp_rank * chunk_size`` and masks the ids outside the local
range, so each rank gathers its own shard and the partial results sum to the full
embedding (the mask zeroes the ranks that hold no matching row). On a mesh with no
TP group it falls back to plain ``F.embedding``.

TODO(pianpwk): rename to VocabParallelEmbedding
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ...utils.spmd_context import spmd_mesh_group

__all__ = ["Embedding"]


class Embedding(nn.Embedding):
    """nn.Embedding with optional local vocab-parallel execution.

    NOTE: currently unused. It pairs with a TP-sharded embedding weight (the
    ``Shard(0)`` on ``tok_embeddings``), but the TP path is not wired up, so
    nothing constructs it and the plain HF embedding is what runs. Whoever wires
    TP needs to swap it in, or the local weight will be indexed with global token
    ids.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run vocab-parallel embedding when the active mesh has a TP group."""
        tp_group = spmd_mesh_group("tp")
        if tp_group is None:
            return F.embedding(
                input,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        tp_size = dist.get_world_size(tp_group)
        chunk_size = (self.num_embeddings + tp_size - 1) // tp_size
        offset = dist.get_rank(tp_group) * chunk_size
        mask = (input >= offset) & (input < offset + self.weight.shape[0])
        local_input = (input - offset).clamp(0, self.weight.shape[0] - 1)
        # padding_idx is a global row id; only the shard that owns that row may
        # hand it to the local F.embedding. Any other shard would either crash
        # (the id is beyond the shard's row count) or silently zero the
        # gradient of an unrelated local row.
        local_padding_idx = None
        if (
            self.padding_idx is not None
            and offset <= self.padding_idx < offset + self.weight.shape[0]
        ):
            local_padding_idx = self.padding_idx - offset
        out = F.embedding(
            local_input,
            self.weight,
            local_padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return out * mask.unsqueeze(-1).to(out.dtype)
