"""Flex-attention kernel as a plain module carrying its own SPMD layout.

Under TP the sharding engine wraps this forward in a *local SPMD region*: q/k/v
arrive as local tensors annotated with head-sharded types, the flex HOP runs, and
the output is annotated to match. Declaring the layout on the module (rather than
hand-rolling a wrapper) is what lets the kernel be sharded the same way as the
rest of the network.

torchtitan makes this a torchtitan ``Module`` subclass so the kernel can carry
its sharding config through that protocol. hpmesh has no module protocol, so
this is a plain ``nn.Module`` holding the sharding config the engine reads.

The HF attention module and the BlockMask ride as passthrough keyword args. They
are not tensors, so the SPMD wrapper leaves them untouched.
"""

from __future__ import annotations

import torch.nn as nn
from transformers.integrations.flex_attention import flex_attention_forward

__all__ = ["HFFlexKernel"]


class HFFlexKernel(nn.Module):
    """Runs HF's flex attention over head-sharded q/k/v.

    Attributes:
        _sharding_config: The local SPMD layout the TP engine applies around
            ``forward``. Named with a leading underscore to match the convention
            every other module follows, so one lookup works for all of them.
    """

    def __init__(self, *, sharding_config) -> None:
        super().__init__()
        self._sharding_config = sharding_config

    def forward(self, query, key, value, *, module, block_mask=None, **kwargs):
        # flex_attention_forward returns (output, lse); output is already
        # transposed to (b, seq, heads, dim). Return just the tensor -- the
        # local SPMD region has a single tensor output.
        out, _ = flex_attention_forward(module, query, key, value, block_mask, **kwargs)
        return out
