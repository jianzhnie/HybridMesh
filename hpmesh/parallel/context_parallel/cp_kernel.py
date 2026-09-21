"""The CP flex kernel: redistribute q/k/v across the CP group, then run flex.

``apply_cp`` attaches one of these to every decoder layer's attention module
as ``_titan_flex_kernel``; ``hf_wrapper._flex_attention_hf`` then routes the
layer's attention call through it. q/k/v arrive HF-shaped --
``(batch, heads, seq, dim)`` -- with the sequence already sharded along dim 2
by ``shard_batch_for_cp``. Two strategies redistribute them:

* ``"kv_allgather"`` all-gathers K/V along the sequence dim, so flex runs the
  local query shard against the full-length keys. Q and the output stay
  token-sharded.
* ``"ulysses"`` all-to-all's every input from a token shard into a head shard
  -- ``(b, h, s/cp, d) -> (b, h/cp, s, d)`` -- runs flex on the full sequence
  with ``heads / cp`` heads, then all-to-all's the output back. Compute and
  memory both shard, at the cost of two all-to-alls per layer.

Neither redistribution depends on an ambient SPMD context. The kv_allgather
gather is torch's own ``flex_cp_allgather`` custom op (the one torchtitan's
HF-backend ``_wrap_flex_kernel_cp`` uses), held by process-group name; the
ulysses all-to-alls drive ``all_to_all_single`` on the group itself. Both are
captured at attach time.

Masks under ulysses: attention runs on the FULL sequence, so the Q-sharded
BlockMask the wrapper hands every CP kernel does not apply -- its Q axis is the
local shard's length. The wrapper cannot be told which strategy attached (its
only CP channel is ``set_cp_mesh``), so the kernel rebuilds the full-length
causal mask itself. That rebuild is exact because the wrapper's internal CP
mask is causal-only; packed (``block_causal``) runs are refused at attach time
in ``apply_cp``, where the model config is still visible.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from hpmesh.utils.batch_invariant import is_in_batch_invariant_mode

__all__ = ["CPFlexKernel"]

# q/k/v arrive HF-shaped: (batch, heads, seq, dim). The sequence axis the CP
# input sharding split is dim 2 of this layout.
_SEQ_DIM = 2
_HEAD_DIM = 1

_KNOWN_STRATEGIES = ("kv_allgather", "ulysses")


def _cp_all_to_all(
    x_BHSD: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """One CP all-to-all: split ``scatter_dim`` across the group, join ``gather_dim``.

    Chunk ``r`` of the scatter dim goes to rank ``r``; the pieces that come
    back are concatenated along the gather dim in rank order. With
    ``scatter_dim=1, gather_dim=2`` this is the ulysses token-to-head swap
    ``(b, h, s/cp, d) -> (b, h/cp, s, d)``; swapping the two dims is the exact
    inverse, which is also each direction's backward.
    """
    cp = group.size()
    shape = x_BHSD.shape
    if shape[scatter_dim] % cp != 0:
        raise ValueError(
            f"Ulysses all-to-all scatters dim {scatter_dim} of size "
            f"{shape[scatter_dim]} across cp={cp} ranks, which does not divide "
            "evenly. For the head axis this means the model's head count must "
            "be a multiple of the CP degree."
        )
    split = (
        shape[:scatter_dim] + (cp, shape[scatter_dim] // cp) + shape[scatter_dim + 1 :]
    )
    # Chunk r of the scatter dim is bound for rank r, so it leads.
    x = x_BHSD.reshape(split).movedim(scatter_dim, 0).contiguous()
    y = torch.empty_like(x)
    dist.all_to_all_single(y, x, group=group)
    # y[r] is rank r's piece; merge the group axis into the gather dim so the
    # pieces concatenate in rank order.
    y = y.movedim(0, gather_dim)
    merged = (
        y.shape[:gather_dim]
        + (y.shape[gather_dim] * y.shape[gather_dim + 1],)
        + y.shape[gather_dim + 2 :]
    )
    return y.reshape(merged)


class _SeqToHead(torch.autograd.Function):
    """``(b, h, s/cp, d) -> (b, h/cp, s, d)``; the backward is the inverse swap."""

    @staticmethod
    def forward(ctx, x_BHSD, group):
        ctx.group = group
        return _cp_all_to_all(x_BHSD, group, scatter_dim=_HEAD_DIM, gather_dim=_SEQ_DIM)

    @staticmethod
    def backward(ctx, grad_BHSD):
        return (
            _cp_all_to_all(
                grad_BHSD, ctx.group, scatter_dim=_SEQ_DIM, gather_dim=_HEAD_DIM
            ),
            None,
        )


class _HeadToSeq(torch.autograd.Function):
    """``(b, h/cp, s, d) -> (b, h, s/cp, d)``; the backward is the inverse swap."""

    @staticmethod
    def forward(ctx, x_BHSD, group):
        ctx.group = group
        return _cp_all_to_all(x_BHSD, group, scatter_dim=_SEQ_DIM, gather_dim=_HEAD_DIM)

    @staticmethod
    def backward(ctx, grad_BHSD):
        return (
            _cp_all_to_all(
                grad_BHSD, ctx.group, scatter_dim=_HEAD_DIM, gather_dim=_SEQ_DIM
            ),
            None,
        )


class CPFlexKernel(nn.Module):
    """Flex attention with q/k/v redistributed across the CP group.

    Args:
        cp_mesh: the CP axis mesh. Its process group is captured at
            construction, so attaching the kernel is also when a missing or
            mis-sized CP axis fails.
        strategy: ``"kv_allgather"`` (all-gather K/V, Q stays token-sharded) or
            ``"ulysses"`` (all-to-all onto the head axis, attention runs
            full-length with ``heads / cp`` heads per rank).
    """

    def __init__(self, *, cp_mesh: DeviceMesh, strategy: str = "kv_allgather") -> None:
        super().__init__()
        if strategy not in _KNOWN_STRATEGIES:
            raise NotImplementedError(
                f"CP strategy {strategy!r} is not wired; supported strategies "
                f"are {_KNOWN_STRATEGIES}."
            )
        self.strategy = strategy
        self._cp_group = cp_mesh.get_group()
        if strategy == "kv_allgather":
            try:
                from torch.distributed.tensor.experimental._context_parallel._cp_custom_ops import (  # noqa: E501
                    flex_cp_allgather,
                )
            except ImportError as e:  # pragma: no cover - torch version guard
                raise ImportError(
                    "CPFlexKernel relies on torch's private ``flex_cp_allgather`` "
                    "(torch.distributed.tensor.experimental._context_parallel."
                    "_cp_custom_ops), which this torch build does not provide. It "
                    "is present in torch 2.6+; upgrade torch, or set cp=1."
                ) from e
            self._flex_cp_allgather = flex_cp_allgather
            self._cp_pg_name = c10d._get_process_group_name(cp_mesh.get_group())
        # Ulysses rebuilds the full-length causal mask per forward (see the
        # module docstring); cache it per (length, device, batch-invariant
        # mode). The cache lives on the kernel instance -- one per attention
        # layer -- so the rebuild happens once per length per layer, not once
        # per forward.
        self._full_masks: dict = {}

    def forward(self, query, key, value, *, module, block_mask=None, **kwargs):
        """Redistribute q/k/v per the strategy, run flex, redistribute back.

        For ``kv_allgather``, ``block_mask`` is the Q-sharded BlockMask (local
        Q, full KV) that pairs with the gathered K/V; for ``ulysses`` the
        incoming mask is replaced by a full-length causal one (see
        :meth:`_forward_ulysses`). Returns just the attention output tensor --
        ``hf_wrapper._flex_attention_hf`` appends the ``None`` LSE itself.
        """
        if self.strategy == "ulysses":
            return self._forward_ulysses(query, key, value, module=module, **kwargs)
        key, value = self._flex_cp_allgather(
            key.contiguous(), value.contiguous(), _SEQ_DIM, self._cp_pg_name
        )
        if query.is_cuda:
            from transformers.integrations.flex_attention import (
                flex_attention_forward,
            )

            out, _ = flex_attention_forward(
                module, query, key, value, block_mask, **kwargs
            )
            return out
        # CPU: transformers routes flex through torch.compile, whose inductor
        # flex lowering has no CPU target (this is why the wrapper picks sdpa
        # off CUDA). The eager fallback computes the same math unfused; it
        # exists so CP is exercisable on CPU-only machines, e.g. the gloo
        # equivalence tests.
        from torch.nn.attention.flex_attention import flex_attention

        return flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=kwargs.get("scaling"),
            enable_gqa=True,
        ).transpose(1, 2)  # HF's interface contract is (batch, seq, heads, dim)

    def _forward_ulysses(self, query, key, value, *, module, **kwargs):
        """Swap the token shard for a head shard, attend full-length, swap back.

        The incoming ``block_mask`` is deliberately dropped: the wrapper
        Q-shards the mask for every CP strategy (it cannot be told which one
        attached), while ulysses attends full-length queries. The full-length
        causal mask is rebuilt here instead -- exact because the wrapper's
        internal CP mask is causal-only. Packed sequences are refused at
        attach time in ``apply_cp``.
        """
        q = _SeqToHead.apply(query.contiguous(), self._cp_group)
        k = _SeqToHead.apply(key.contiguous(), self._cp_group)
        v = _SeqToHead.apply(value.contiguous(), self._cp_group)
        block_mask = self._full_length_causal_mask(q)
        if query.is_cuda:
            from transformers.integrations.flex_attention import (
                flex_attention_forward,
            )

            out, _ = flex_attention_forward(module, q, k, v, block_mask, **kwargs)
            out = out.transpose(1, 2)  # HF returns (batch, seq, heads, dim)
        else:
            # Same CPU eager fallback as the kv_allgather path above.
            from torch.nn.attention.flex_attention import flex_attention

            out = flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=kwargs.get("scaling"),
                enable_gqa=True,
            )  # already (batch, heads, seq, dim)
        out = _HeadToSeq.apply(out.contiguous(), self._cp_group)
        return out.transpose(1, 2)  # HF's interface contract is (b, s/cp, h, d)

    def _full_length_causal_mask(self, q_BHSD: torch.Tensor):
        """The full-sequence causal BlockMask, built once per length, device,
        and batch-invariant mode.

        Under ulysses, q/k/v arrive at flex with the full sequence, so the mask
        is the unsharded causal one -- the same mask the wrapper builds
        internally before Q-sharding it for kv_allgather. ``separate_full_blocks``
        tracks the wrapper's batch-invariant-mode choice, so the ulysses
        decomposition matches every other path's numerics.
        """
        from torch.nn.attention.flex_attention import create_block_mask

        def _causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        seq_len = q_BHSD.shape[_SEQ_DIM]
        key = (seq_len, q_BHSD.device, is_in_batch_invariant_mode())
        mask = self._full_masks.get(key)
        if mask is None:
            mask = create_block_mask(
                _causal,
                1,
                None,
                seq_len,
                seq_len,
                device=q_BHSD.device,
                BLOCK_SIZE=128,
                separate_full_blocks=not is_in_batch_invariant_mode(),
            )
            self._full_masks[key] = mask
        return mask
