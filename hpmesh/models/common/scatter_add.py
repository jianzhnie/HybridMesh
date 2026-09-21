"""Deterministic ``scatter_add`` for the MoE combine step.

Vendored from torchtitan ``ops/scatter_add.py``, ``custom_op`` wrapper and all.
The wrapper is what makes the determinism requirement survive ``torch.compile``:
inductor reads the global deterministic-algorithms flag at compile time, so an
eager-side toggle around a plain ``scatter_add`` never enters the graph. As a
registered op, the toggle lives inside the op body, the fake impl keeps FX
tracing working, and the custom autograd node replaces the rule a raw
``Tensor.scatter_add`` would have carried.

Why the determinism toggle: on CUDA, ``scatter_add`` resolves duplicate indices
with atomics, whose order varies run to run. The combine step scatters every
token back into its original position, so a token routed to k experts hits the
same index k times -- exactly the duplicate-index case. Forcing deterministic
algorithms picks the ordered kernel instead, which is what makes a run
reproducible.
"""

from __future__ import annotations

import torch

__all__ = ["deterministic_scatter_add"]


@torch.library.custom_op("hpmesh::deterministic_scatter_add", mutates_args=())
def deterministic_scatter_add(
    out: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    """``out.scatter_add(dim=0, index, src)``, forced to a deterministic kernel.

    The toggle is global, so it is restored (including the warn-only flag)
    rather than simply turned off -- the caller may have had it set.
    """
    prev_enabled = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        return out.scatter_add(dim=0, index=index, src=src)
    finally:
        torch.use_deterministic_algorithms(prev_enabled, warn_only=prev_warn_only)


@deterministic_scatter_add.register_fake
def _(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(out)


def _backward(
    ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None, torch.Tensor]:
    (index,) = ctx.saved_tensors
    grad_src = torch.gather(grad_output, dim=0, index=index)
    return grad_output, None, grad_src


def _setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: torch.Tensor,
) -> None:
    _out, index, _src = inputs
    ctx.save_for_backward(index)


deterministic_scatter_add.register_autograd(
    _backward,
    setup_context=_setup_context,
)
