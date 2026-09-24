"""The projections a ``models/common`` model applies, where they differ from ``nn``.

Vendored from torchtitan ``models/common/linear.py``. Upstream ``Linear`` is a
diamond subclass of ``nn.Linear`` + ``Module`` so that a config can build it;
that class is not carried over -- a model that wants a plain projection uses
``nn.Linear`` / the ``parallel.tensor_parallel`` wrappers directly, which is
what ``dist_gemm`` and ``tp`` already do.

What is kept is the two classes whose forward is NOT ``nn.Linear``:

* :class:`RouterGateLinear` -- the router scores, pinned to fp32.
* :class:`PartialBiasRowwiseLinear` -- a bias that is TP-partial in forward.

Shape suffix legend, scoped to this file: T = tokens, D = model dimension,
E = number of experts.
"""

from __future__ import annotations

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from hpmesh.accelerator.spmd_context import spmd_mesh_group

__all__ = [
    "PartialBiasRowwiseLinear",
    "RouterGateLinear",
]


@spmd.register_local_autograd_function
class _RouterGateLinearFunction(torch.autograd.Function):
    """Router projection with FP32 output and backward GEMMs."""

    @staticmethod
    def forward(ctx, input_TD: torch.Tensor, weight_ED: torch.Tensor) -> torch.Tensor:
        use_cuda_bf16_forward = (
            input_TD.device.type == "cuda"
            and input_TD.dtype is torch.bfloat16
            and weight_ED.dtype is torch.bfloat16
        )
        if use_cuda_bf16_forward:
            input_forward_TD = input_TD
            weight_forward_ED = weight_ED
            # CUDA supports BF16 matmul with FP32 accumulation and output via
            # out_dtype. The portable path below promotes the operands because
            # this mixed input/output dtype is not supported by all devices.
            output_TE = torch.mm(
                input_forward_TD, weight_forward_ED.T, out_dtype=torch.float32
            )
        else:
            input_forward_TD = input_TD.float()
            weight_forward_ED = weight_ED.float()
            output_TE = torch.mm(input_forward_TD, weight_forward_ED.T)

        ctx.save_for_backward(input_forward_TD, weight_forward_ED)
        ctx.input_dtype = input_TD.dtype
        ctx.weight_dtype = weight_ED.dtype
        return output_TE

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TE: torch.Tensor):
        input_forward_TD, weight_forward_ED = ctx.saved_tensors
        grad_output_fp32_TE = grad_output_TE.float()

        grad_input_TD = None
        if ctx.needs_input_grad[0]:
            grad_input_TD = torch.mm(grad_output_fp32_TE, weight_forward_ED.float()).to(
                ctx.input_dtype
            )

        grad_weight_ED = None
        if ctx.needs_input_grad[1]:
            grad_weight_ED = torch.mm(
                grad_output_fp32_TE.T, input_forward_TD.float()
            ).to(ctx.weight_dtype)

        return grad_input_TD, grad_weight_ED


class RouterGateLinear(nn.Linear):
    """The router's projection: one score per expert, always in fp32.

    A plain ``nn.Linear`` would return the input dtype, and a bf16 score can
    reorder a top-k on close calls, so both the forward GEMM and the backward
    GEMMs run in fp32 whatever the model's dtype is. The custom Function (rather
    than casting the operands inline) is what pins the backward as well; on CUDA
    with bf16 operands the forward keeps bf16 inputs and asks for an fp32
    accumulation, while every other device promotes the operands first.

    Subclasses ``nn.Linear`` rather than a bare ``nn.Module`` so the module
    signature is the one the rest of the stack expects: ``weight`` of shape
    ``(E, D)``, and a ``bias`` attribute that exists (as ``None``) even when the
    projection is built without one.

    Args:
        dim: model dimension (D).
        num_experts: number of experts (E).
        bias: add a bias to the scores. Defaults to False, matching the
            bias-free projection the HF MoE blocks ship.
    """

    def __init__(self, dim: int, num_experts: int, bias: bool = False) -> None:
        super().__init__(dim, num_experts, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_TE = _RouterGateLinearFunction.apply(input, self.weight)
        if self.bias is not None:
            output_TE = output_TE + self.bias.float()
        return output_TE


class PartialBiasRowwiseLinear(nn.Linear):
    """Rowwise linear whose invariant bias becomes TP-partial in forward.

    A rowwise projection shards the input features, so each rank holds a partial
    sum until the reduce-scatter. A bias added before that reduction must be
    partial too -- added on every rank it would be counted ``tp`` times, so it is
    converted to a partial value and left for the collective to sum.

    No bias means nothing to redistribute, which is why the bias is a
    construction requirement rather than an option.

    Args:
        in_features: input feature count (the weight's second dim).
        out_features: output feature count (the weight's first dim).
        bias: must be true; there is nothing to redistribute without it.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        if not bias:
            raise ValueError("PartialBiasRowwiseLinear requires bias=True")
        super().__init__(in_features, out_features, bias=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        assert bias is not None
        tp_group = spmd_mesh_group("tp")
        if tp_group is not None:
            bias = spmd.convert(
                bias,
                tp_group,
                src=spmd.I,
                dst=spmd.P,
                expert_mode=True,
            )
        return F.linear(input, self.weight, bias)
