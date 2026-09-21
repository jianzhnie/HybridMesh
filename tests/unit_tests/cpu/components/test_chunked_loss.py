"""Chunked lm_head + cross-entropy: pinned against the full-logits path.

The chunked path exists to bound peak memory, but its contract is numerical:
the loss value and every gradient must match ``lm_head`` over the whole
sequence followed by one sum-reduced cross-entropy. These tests compare
against exactly that reference (not against the chunked code re-run), with
``IGNORE_INDEX`` entries and chunk counts that do not divide the sequence --
the two boundaries where a chunked implementation loses or double-counts
positions.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from hpmesh.components.loss import IGNORE_INDEX, chunked_lm_head_cross_entropy

T = 13  # deliberately not divisible by 2, 3, or 4
H = 8
V = 11


def _inputs(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, nn.Linear]:
    torch.manual_seed(seed)
    hidden = torch.randn(T, H)
    lm_head = nn.Linear(H, V)
    # IGNORE_INDEX at a chunk boundary, inside chunks, and covering a whole
    # final chunk when num_chunks=13 (every tail position is its own chunk).
    labels = torch.tensor(
        [1, IGNORE_INDEX, 3, 4, 5, 0, 7, 8, IGNORE_INDEX, 2, 6, 9, 10]
    )
    return hidden, labels, lm_head


def _reference(
    hidden: torch.Tensor, labels: torch.Tensor, lm_head: nn.Linear, grad_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The full-logits path: one lm_head, one sum CE, one scaled backward."""
    hidden = hidden.detach().requires_grad_(True)
    head = nn.Linear(H, V)
    head.load_state_dict(lm_head.state_dict())
    loss_sum = F.cross_entropy(
        head(hidden).float(), labels, reduction="sum", ignore_index=IGNORE_INDEX
    )
    (loss_sum * grad_scale).backward()
    return (
        loss_sum.detach(),
        hidden.grad,
        head.weight.grad,
        head.bias.grad,
    )


def _chunked(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Linear,
    num_chunks: int,
    grad_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = hidden.detach().requires_grad_(True)
    head = nn.Linear(H, V)
    head.load_state_dict(lm_head.state_dict())
    loss_sum = chunked_lm_head_cross_entropy(
        head, hidden, labels, num_chunks=num_chunks, grad_scale=grad_scale
    )
    return loss_sum, hidden.grad, head.weight.grad, head.bias.grad


@pytest.mark.parametrize("num_chunks", [1, 2, 3, 4, 7, 13, 20])
def test_chunked_matches_full_loss_and_gradients(num_chunks: int) -> None:
    """Loss, hidden-state grad, and lm_head weight/bias grads all match.

    ``num_chunks=20`` exceeds T=13: ``torch.chunk`` yields fewer chunks than
    requested, which must still be exact. fp32 on CPU, so the only sanctioned
    divergence is summation order (~1e-6).
    """
    hidden, labels, lm_head = _inputs()
    grad_scale = 1.0 / float((labels != IGNORE_INDEX).sum())

    want = _reference(hidden, labels, lm_head, grad_scale)
    got = _chunked(hidden, labels, lm_head, num_chunks, grad_scale)

    torch.testing.assert_close(got[0], want[0], rtol=1e-5, atol=1e-7)
    for name, g, w in zip(
        ("hidden.grad", "weight.grad", "bias.grad"), got[1:], want[1:], strict=True
    ):
        assert g is not None, f"{name} missing on the chunked path"
        torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-7, msg=name)


def test_a_chunk_with_only_ignored_labels_contributes_nothing() -> None:
    """An all-ignored chunk must sum 0 and backprop zeros, not NaN.

    Mean reduction would NaN here; the sum reduction the trainer uses must
    not, and the chunk must leave the totals untouched.
    """
    hidden, labels, lm_head = _inputs()
    # Ignore positions 4..7 -- exactly the second chunk when num_chunks=3
    # splits T=12... but T=13 splits 5/4/4, so set labels explicitly per the
    # actual split: positions [5, 9) are the middle chunk for num_chunks=3.
    labels = labels.clone()
    labels[5:9] = IGNORE_INDEX

    grad_scale = 1.0 / float((labels != IGNORE_INDEX).sum())
    want = _reference(hidden, labels, lm_head, grad_scale)
    got = _chunked(hidden, labels, lm_head, 3, grad_scale)

    assert torch.isfinite(got[0])
    torch.testing.assert_close(got[0], want[0], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(got[1], want[1], rtol=1e-5, atol=1e-7)
    # The middle chunk's hidden rows carry exactly zero gradient.
    assert torch.all(got[1][5:9] == 0)


def test_grad_scale_is_applied_to_every_gradient() -> None:
    """``grad_scale`` scales lm_head and hidden gradients alike.

    The trainer passes ``1 / global_valid_tokens``; scaling only the hidden
    branch (or only the head) would silently denormalize the other.
    """
    hidden, labels, lm_head = _inputs()
    unscaled = _chunked(hidden, labels, lm_head, 4, 1.0)
    scaled = _chunked(hidden, labels, lm_head, 4, 0.25)

    # The reported loss is the raw sum either way; only gradients scale.
    torch.testing.assert_close(scaled[0], unscaled[0], rtol=1e-5, atol=1e-7)
    for s, u in zip(scaled[1:], unscaled[1:], strict=True):
        torch.testing.assert_close(s, u * 0.25, rtol=1e-5, atol=1e-7)


def test_returned_loss_is_detached_and_unnormalized() -> None:
    """The return is a report value, not a graph: backward already ran."""
    hidden, labels, lm_head = _inputs()
    loss_sum, *_ = _chunked(hidden, labels, lm_head, 2, 1.0)
    assert not loss_sum.requires_grad
    full = F.cross_entropy(
        lm_head(hidden.detach()).float(),
        labels,
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )
    torch.testing.assert_close(loss_sum, full, rtol=1e-5, atol=1e-7)


def test_invalid_inputs_raise() -> None:
    hidden, labels, lm_head = _inputs()
    with pytest.raises(ValueError, match="num_chunks"):
        chunked_lm_head_cross_entropy(
            lm_head, hidden.requires_grad_(True), labels, num_chunks=0, grad_scale=1.0
        )
    with pytest.raises(ValueError, match="does not match"):
        chunked_lm_head_cross_entropy(
            lm_head,
            hidden.requires_grad_(True),
            labels[:-1],
            num_chunks=2,
            grad_scale=1.0,
        )
    with pytest.raises(ValueError, match="require grad"):
        chunked_lm_head_cross_entropy(
            lm_head, hidden.detach(), labels, num_chunks=2, grad_scale=1.0
        )
    with pytest.raises(ValueError, match="T, H"):
        chunked_lm_head_cross_entropy(
            lm_head,
            hidden.requires_grad_(True).unsqueeze(0),
            labels,
            num_chunks=2,
            grad_scale=1.0,
        )
