"""The deterministic scatter_add: eager semantics, autograd, and compile.

The op is a ``torch.library.custom_op`` so the determinism requirement travels
into a compiled graph -- inductor reads the deterministic-algorithms flag at
compile time, so an eager-side toggle around a plain ``scatter_add`` would be
invisible there. These tests pin the eager behavior unchanged and the op's
traceability (a fullgraph compile must not break on it).
"""

from __future__ import annotations

from tests.caps import require_env

require_env('spmd_types')


import torch

from hpmesh.models.common.scatter_add import deterministic_scatter_add


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    # Duplicate indices in both rows: the case the determinism is for.
    index = torch.tensor([[0, 2, 0], [1, 1, 3]])
    src = torch.randn(2, 3, generator=g, dtype=torch.float64)
    out = torch.randn(4, 3, generator=g, dtype=torch.float64)
    return out, index, src


def test_matches_plain_scatter_add() -> None:
    out, index, src = _inputs()
    got = deterministic_scatter_add(out, index, src)
    expected = out.scatter_add(dim=0, index=index, src=src)
    assert torch.equal(got, expected)


def test_does_not_mutate_the_out_argument() -> None:
    out, index, src = _inputs()
    snapshot = out.clone()
    deterministic_scatter_add(out, index, src)
    assert torch.equal(out, snapshot)


def test_backward_matches_plain_scatter_add() -> None:
    """grad_out passes through; grad_src gathers along the index."""
    out, index, src = _inputs()

    out1 = out.clone().requires_grad_(True)
    src1 = src.clone().requires_grad_(True)
    deterministic_scatter_add(out1, index, src1).sum().backward()

    out2 = out.clone().requires_grad_(True)
    src2 = src.clone().requires_grad_(True)
    out2.scatter_add(dim=0, index=index, src=src2).sum().backward()

    assert torch.equal(out1.grad, out2.grad)
    assert torch.equal(src1.grad, src2.grad)
    assert index.requires_grad is False


def test_determinism_flag_is_restored() -> None:
    """The toggle is global; the caller's setting must survive the call."""
    prev = torch.are_deterministic_algorithms_enabled()
    out, index, src = _inputs()
    deterministic_scatter_add(out, index, src)
    assert torch.are_deterministic_algorithms_enabled() is prev


def test_works_inside_torch_compile_without_a_graph_break() -> None:
    """``fullgraph=True`` raises on any graph break, so passing pins the op's
    traceability; the compiled result must match eager bit for bit (the op is
    opaque to inductor, so nothing around it can be reassociated)."""
    out, index, src = _inputs()

    compiled = torch.compile(deterministic_scatter_add, fullgraph=True)
    got = compiled(out, index, src)
    expected = deterministic_scatter_add(out, index, src)
    assert torch.equal(got, expected)
