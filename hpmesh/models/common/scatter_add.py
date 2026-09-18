"""Deterministic ``scatter_add`` for the MoE combine step.

Vendored from torchtitan ``ops/scatter_add.py``, minus the
``torch.library.custom_op`` registration. That wrapper exists so the op has a
fake implementation for FX tracing and a custom autograd node; here it runs
eager, and ``Tensor.scatter_add`` already carries a correct autograd rule, so the
plain call keeps the backward for free.

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
