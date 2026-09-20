"""The loss functions: pinned against an independent reference, not themselves.

`cross_entropy_loss` is exercised end-to-end in `test_trainer.py`, but always in
terms of itself -- a test that compares the sharded path to the plain path would
pass even if both were wrong the same way. These tests instead check each
function against the `torch.nn.functional` call it is supposed to be, which is
the only thing that pins the *value* rather than the *agreement*.

The vocab-parallel selection is deliberately shape-driven (`pred.shape[-1] !=
global_vocab_size`), so the tests that matter are the ones proving the plain
path is taken when it should be -- a mis-selected path that happens to also
return a sum would be invisible otherwise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hpmesh.components.loss import (
    IGNORE_INDEX,
    compute_logprobs,
    cross_entropy_loss,
    mse_loss,
)


def _logits_and_labels(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    logits = torch.randn(6, 9)
    # One position carries IGNORE_INDEX so every test sees the masking.
    labels = torch.tensor([1, 5, IGNORE_INDEX, 3, 8, 0])
    return logits, labels


# -- cross_entropy_loss -------------------------------------------------------


def test_cross_entropy_matches_the_sum_reduced_reference() -> None:
    logits, labels = _logits_and_labels()
    torch.testing.assert_close(
        cross_entropy_loss(logits, labels),
        F.cross_entropy(logits, labels, reduction="sum", ignore_index=IGNORE_INDEX),
        rtol=1e-5,
        atol=1e-8,
    )


def test_the_ignored_position_contributes_nothing_to_the_sum() -> None:
    """A sum reduction must still honour ``ignore_index``, not error on it.

    ``F.cross_entropy`` defaults to ``ignore_index=-100`` with a *mean*
    reduction; this call passes ``IGNORE_INDEX`` explicitly, so a change to
    either the constant or the reduction is caught here rather than showing up
    as a slightly-wrong loss curve.
    """
    logits, labels = _logits_and_labels()
    full = cross_entropy_loss(logits, labels)
    # Replace the ignored position with a real index: the sum must grow.
    unignored = labels.clone()
    unignored[2] = 4
    assert cross_entropy_loss(logits, unignored) > full


def test_a_full_vocab_takes_the_plain_path_even_with_a_tp_group() -> None:
    """Path selection is by shape: equal last dim means nothing is sharded.

    ``tp_group`` is only touched on the vocab-parallel branch, so passing a
    non-group object is what proves the branch was *not* taken.
    """
    logits, labels = _logits_and_labels()
    torch.testing.assert_close(
        cross_entropy_loss(
            logits, labels, tp_group=object(), global_vocab_size=logits.shape[-1]
        ),
        F.cross_entropy(logits, labels, reduction="sum", ignore_index=IGNORE_INDEX),
        rtol=1e-5,
        atol=1e-8,
    )


# -- compute_logprobs ---------------------------------------------------------


def test_logprobs_are_the_negated_unreduced_reference() -> None:
    logits, labels = _logits_and_labels()
    torch.testing.assert_close(
        compute_logprobs(logits, labels),
        -F.cross_entropy(logits, labels, reduction="none", ignore_index=IGNORE_INDEX),
        rtol=1e-5,
        atol=1e-8,
    )


def test_entropy_matches_the_shannon_formula() -> None:
    """``H(p) = logsumexp(logits) - sum(softmax(logits) * logits)``, per token."""
    logits, labels = _logits_and_labels()
    _, entropy = compute_logprobs(logits, labels, return_entropy=True)

    expected = torch.logsumexp(logits, dim=-1) - (
        torch.softmax(logits, dim=-1) * logits
    ).sum(dim=-1)
    torch.testing.assert_close(entropy, expected, rtol=1e-5, atol=1e-6)
    assert entropy.shape == (logits.shape[0],), "one entropy per token"


def test_entropy_does_not_extend_the_autograd_graph() -> None:
    """The documented contract: entropy is a metric, so it must not backprop.

    The implementation wraps it in ``no_grad``. If that ever regressed, the
    softmax over the logits would join the graph and every backward pass would
    carry an extra term -- silently, since the *values* would be unchanged.
    """
    logits, labels = _logits_and_labels()
    logits = logits.requires_grad_(True)

    logprobs, entropy = compute_logprobs(logits, labels, return_entropy=True)

    assert entropy.requires_grad is False
    assert entropy.grad_fn is None
    # The logprobs, by contrast, are the training signal and must be connected.
    assert logprobs.requires_grad is True


def test_the_ignored_position_gets_a_zero_logprob() -> None:
    """``ignore_index`` maps to cross-entropy's own 0, not to a real target."""
    logits, labels = _logits_and_labels()
    logprobs = compute_logprobs(logits, labels)
    assert logprobs[2].item() == 0.0


# -- mse_loss -----------------------------------------------------------------


def test_mse_matches_the_sum_reduced_reference() -> None:
    torch.manual_seed(1)
    pred = torch.randn(4, 3)
    target = torch.randn(4, 3)
    torch.testing.assert_close(
        mse_loss(pred, target),
        F.mse_loss(pred, target, reduction="sum"),
        rtol=1e-5,
        atol=1e-8,
    )


def test_mse_detaches_its_labels() -> None:
    """Continuous targets are data, not learnable parameters.

    Without the detach, a caller that built ``labels`` from a differentiable
    computation would get gradients flowed into it -- a silent graph extension
    that changes what the optimizer trains.
    """
    torch.manual_seed(2)
    pred = torch.randn(4, requires_grad=True)
    target = torch.randn(4, requires_grad=True)

    mse_loss(pred, target).backward()

    assert pred.grad is not None, "the prediction is what we train"
    assert target.grad is None, "the target must be cut from the graph"
