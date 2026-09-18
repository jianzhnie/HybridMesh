"""Fused QKV projection and the head-split helper.

Vendored from torchtitan ``models/common/attention.py``. Only these two pieces
came across: the rest of that module is a family of attention implementations
that would duplicate the HF models hpmesh wraps. What was removed here:

* the ``Module`` protocol and its ``Config`` -- ``QKVLinear`` takes its sizes as
  keyword args, and ``linear`` is a plain ``nn.Linear`` (a ``nn.Linear`` plus a
  ``Config`` adds nothing, so hpmesh uses the module directly).
* the ``spmd.local_map`` / ``spmd.assert_type`` decorations and the
  ``spmd.local()`` blocks -- type-checker annotations with no runtime effect.
  ``local_head_split`` documented a sharding contract; that is now prose rather
  than an assertion, since ``spmd_types`` is not wired into hpmesh's forward
  path.

Why fuse Q, K and V into one projection: three separate GEMMs of the same input
become one, which cuts kernel launches and gives the TP sharding a single tensor
to split. The ``R`` dimension below is ``heads_per_kv + 2`` -- one K head and one
V head per KV group alongside its Q heads -- so the fused layout stays grouped by
KV head.

Why the checkpoint hooks: external checkpoints (HF among them) store separate
``wq``/``wk``/``wv`` tensors, so the fused parameter has to present itself under
those names on save and accept them on load. Without this an HP-format
checkpoint would not load at all.

Shape legend, scoped to this file: ``T`` = tokens, ``D`` = model dimension,
``H`` = head dimension, ``R`` = heads-per-KV-group + 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate

__all__ = ["QKVLinear", "local_head_split"]


def local_head_split(
    t: torch.Tensor,
    head_dim: int,
    *,
    dp_shard_dim: int = 0,
) -> torch.Tensor:
    """Split the last dimension into heads, keeping it sharded on the local axis.

    A thin reshape used where a tensor's trailing dim is ``n_heads * head_dim``:
    the split is local by construction, so no communication is involved on
    either sharded axis (``dp_shard_dim`` keeps its shard, and the last dim
    stays sharded on ``tp``).
    """
    return t.view(*t.shape[:-1], -1, head_dim)


class QKVLinear(nn.Module):
    """One fused QKV projection, split along ``R`` on output.

    Args:
        head_dim: size of one attention head.
        n_heads: number of query heads.
        n_kv_heads: number of key/value heads (GQA). Must divide ``n_heads``.
        linear: the fused projection, sized
            ``n_kv_heads * r * head_dim`` x ``dim``. Passed in rather than
            built here so the caller controls its dtype, bias and sharding.

    Raises:
        ValueError: if ``n_kv_heads`` does not divide ``n_heads``, which would
            leave a KV group with a fractional number of query heads.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        linear: nn.Module,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by "
                f"n_kv_heads ({n_kv_heads}) for fused QKV"
            )
        self.wqkv = linear
        self.heads_per_kv = n_heads // n_kv_heads
        self.r_dim = self.heads_per_kv + 2
        # Registering on the instance keeps the weight/bias handling identical
        # for both, and avoids re-reading the parameters on every call.
        self.register_state_dict_post_hook(self._split_qkv_on_save)
        self.register_load_state_dict_pre_hook(self._merge_qkv_on_load)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ``x`` and return ``(q, k, v)``, each ``(T, n_heads, H)``."""
        num_tokens = x.shape[0]
        # Fused QKV: one matmul, then reshape and split along R.
        # [T, n_kv_heads * R * head_dim] -> [T, n_kv_heads, R, head_dim].
        # -1 rather than n_kv_heads so a TP-sharded output still reshapes.
        qkv = self.wqkv(x)
        qkv = qkv.view(num_tokens, -1, self.r_dim, self.head_dim)

        local_num_tokens = qkv.shape[0]
        xq, xk, xv = torch.split(qkv, [self.heads_per_kv, 1, 1], dim=-2)
        # split leaves xk/xv as strided views into the fused buffer; attention
        # and KV-cache kernels read raw memory assuming a contiguous head-major
        # layout, so materialize all three here.
        return (
            xq.reshape(local_num_tokens, -1, self.head_dim).contiguous(),
            xk.reshape(local_num_tokens, -1, self.head_dim).contiguous(),
            xv.reshape(local_num_tokens, -1, self.head_dim).contiguous(),
        )

    @staticmethod
    def _split_qkv_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Split fused ``wqkv`` into stock ``wq``/``wk``/``wv`` (weight and bias)."""
        hd, hpk, r = module.head_dim, module.heads_per_kv, module.r_dim

        for param, ndim in (("weight", 4), ("bias", 3)):
            key = f"{prefix}wqkv.{param}"
            if key not in state_dict:
                continue
            tensor = state_dict.pop(key)
            # Gather to Replicate so the n_kv-leading reshape is local (dim 0
            # unsharded) when a Shard(0) split would not divide n_kv_heads
            # (e.g. dp_shard=8, n_kv_heads=4); stays a DTensor for the copy.
            if isinstance(tensor, DTensor):
                tensor = tensor.redistribute(
                    tensor.device_mesh, [Replicate()] * tensor.device_mesh.ndim
                )
            n_kv = tensor.shape[0] // (r * hd)
            tail = (tensor.shape[1],) if ndim == 4 else ()
            w = tensor.reshape(n_kv, r, hd, *tail)
            state_dict[f"{prefix}wq.{param}"] = (
                w[:, :hpk].reshape(-1, *tail).contiguous()
            )
            state_dict[f"{prefix}wk.{param}"] = (
                w[:, hpk].reshape(-1, *tail).contiguous()
            )
            state_dict[f"{prefix}wv.{param}"] = (
                w[:, hpk + 1].reshape(-1, *tail).contiguous()
            )

    @staticmethod
    def _merge_qkv_on_load(module, state_dict, prefix, *args) -> None:
        """Merge stock ``wq``/``wk``/``wv`` back into fused ``wqkv``."""
        hd, hpk = module.head_dim, module.heads_per_kv

        for param, ndim in (("weight", 4), ("bias", 3)):
            keys = [f"{prefix}{w}.{param}" for w in ("wq", "wk", "wv")]
            if not all(k in state_dict for k in keys):
                continue
            wq, wk, wv = (state_dict.pop(k) for k in keys)
            # TODO: check whether this all-gather can be avoided.
            # Gather to Replicate so the n_kv reshape is local; stays a DTensor
            # so the fused result can be copied into the sharded wqkv param.
            if isinstance(wq, DTensor):
                wq, wk, wv = (
                    t.redistribute(t.device_mesh, [Replicate()] * t.device_mesh.ndim)
                    for t in (wq, wk, wv)
                )
            n_kv = wk.shape[0] // hd
            tail = (wq.shape[1],) if ndim == 4 else ()
            q = wq.reshape(n_kv, hpk, hd, *tail)
            k = wk.reshape(n_kv, 1, hd, *tail)
            v = wv.reshape(n_kv, 1, hd, *tail)
            state_dict[f"{prefix}wqkv.{param}"] = torch.cat([q, k, v], dim=1).reshape(
                -1, *tail
            )
