"""A fixed-dtype forward matmul for the decoder's output projection.

Semantics vendored from torchtitan's ``CastLinear``: the vocabulary logits are
scored in a configured dtype regardless of the dtype the rest of the model
runs in, because bf16 logits lose the precision the loss (and RL logprob/KL
math downstream of it) needs. Upstream swaps the lm_head's config node inside
its model-config tree; hpmesh has no config tree, so the swap happens at model
build time in ``models/hf_wrapper.py`` and this file carries only the module
and the in-place swap helper.

The class is an ``nn.Linear`` subclass rather than a wrapper around one on
purpose: a wrapper would register the original projection as a child and
prefix every state-dict key with the wrapper's name, breaking checkpoint
compatibility with an unwrapped run. Keeping ``weight``/``bias`` as direct
parameters of the replacement module is what leaves the FQNs -- and, because
the swap rebinds the same ``Parameter`` objects, an embedding/lm_head weight
tie -- exactly as they were.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TORCH_DTYPE_MAP", "CastLinear", "to_cast_linear"]

# Keys spell dtypes the way the training config does; the set matches
# torchtitan's TORCH_DTYPE_MAP.
TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class CastLinear(nn.Linear):
    """An ``nn.Linear`` whose forward matmul runs in ``compute_dtype``.

    Inputs, weight, and bias are cast to ``compute_dtype`` before ``F.linear``
    and the output is returned in that dtype. The stored parameters keep their
    original dtype -- autograd casts the incoming gradients back through the
    forward casts, so the weight gradient lands in the parameter's own dtype
    and the optimizer never sees an upcast copy. The cast cannot be cached
    across steps because the optimizer updates the weight every step.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        compute_dtype: torch.dtype,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.compute_dtype = compute_dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        return F.linear(
            input.to(self.compute_dtype), self.weight.to(self.compute_dtype), bias
        )


def to_cast_linear(linear: nn.Module, compute_dtype: torch.dtype) -> CastLinear:
    """Build a ``CastLinear`` that reuses ``linear``'s parameter objects.

    Rebinding rather than copying is the point: a tied lm_head shares one
    ``Parameter`` with the embedding, and only reusing the object keeps the tie
    (and the optimizer's parameter identity) intact. The geometry is read off
    the module being replaced, so the swap cannot silently change shapes.
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(
            f"lm_head cast expects an nn.Linear to replace, got "
            f"{type(linear).__name__}. A model whose output projection is not "
            "a plain Linear needs its own cast path."
        )
    cast = CastLinear(
        linear.in_features,
        linear.out_features,
        compute_dtype=compute_dtype,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    cast.weight = linear.weight
    cast.bias = linear.bias
    return cast
