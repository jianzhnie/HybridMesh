"""The CP flex kernel: all-gather K/V across the CP group, then run flex.

``apply_cp_ep`` attaches one of these to every decoder layer's attention module
as ``_titan_flex_kernel``; ``hf_wrapper._flex_attention_hf`` then routes the
layer's attention call through it. q/k/v arrive HF-shaped --
``(batch, heads, seq, dim)`` -- with the sequence already sharded along dim 2
by ``shard_batch_for_cp``. Attention needs full-length K/V, so the kernel
all-gathers them across the CP group before running flex; Q and the output
stay token-sharded, which is what makes this the KV-all-gather strategy.

The gather is torch's own ``flex_cp_allgather`` custom op (the one torchtitan's
HF-backend ``_wrap_flex_kernel_cp`` uses): forward all-gathers along the
sequence dim, backward reduce-scatters each rank's gradient slice back. The
kernel holds its CP process group by name, resolved at attach time, so the
forward does not depend on any ambient SPMD context.

Only the KV-all-gather strategy is wired; ``strategy`` reserves the slot for
Ulysses (all-to-all onto the head axis), whose primitives already exist in
``parallel/cp_ep.py``.
"""

from __future__ import annotations

import torch.distributed.distributed_c10d as c10d
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

__all__ = ["CPFlexKernel"]

# q/k/v arrive HF-shaped: (batch, heads, seq, dim). The sequence axis the CP
# input sharding split is dim 2 of this layout.
_SEQ_DIM = 2


class CPFlexKernel(nn.Module):
    """Flex attention with K/V all-gathered across the CP group.

    Args:
        cp_mesh: the CP axis mesh. Its process group is captured by name at
            construction, so attaching the kernel is also when a missing or
            mis-sized CP axis fails.
        strategy: ``"kv_allgather"`` only; anything else reserves the Ulysses
            slot and raises.
    """

    def __init__(self, *, cp_mesh: DeviceMesh, strategy: str = "kv_allgather") -> None:
        super().__init__()
        if strategy != "kv_allgather":
            raise NotImplementedError(
                f"CP strategy {strategy!r} is not wired; only 'kv_allgather' "
                "(all-gather K/V, Q stays token-sharded) is. The Ulysses "
                "all-to-all primitives exist in parallel/cp_ep.py."
            )
        self.strategy = strategy
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

    def forward(self, query, key, value, *, module, block_mask=None, **kwargs):
        """All-gather K/V along the sequence dim, then run flex attention.

        ``block_mask`` is the Q-sharded BlockMask (local Q, full KV) that pairs
        with the gathered K/V. Returns just the attention output tensor --
        ``hf_wrapper._flex_attention_hf`` appends the ``None`` LSE itself.
        """
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
