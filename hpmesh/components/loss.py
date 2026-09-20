# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loss functions: plain next-token CE, and its vocab-parallel form.

Vendored from torchtitan ``components/loss.py``. What was dropped is the
``Configurable`` hierarchy (``BaseLoss`` / ``CrossEntropyLoss`` / ``MSELoss``)
and the ``spmd.assert_type`` annotations -- hpmesh configures by argument and
checks shapes, not SPMD types. What was kept is the arithmetic, unchanged, plus
``IGNORE_INDEX`` and ``next_token_targets``.

Two entry points, and the difference between them is worth stating plainly:

* ``cross_entropy_loss`` -- the loss, with a sum reduction, so the caller can
  divide by a *global* token count once the per-rank counts have been reduced.
* ``compute_logprobs`` -- per-token log-probabilities, for inference-side use
  (GRPO, perplexity). Different job, mostly the same math.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F

__all__ = [
    "IGNORE_INDEX",
    "LossFunction",
    "compute_logprobs",
    "cross_entropy_loss",
    "mse_loss",
    "next_token_targets",
    "vocab_shard_bounds",
]

# PyTorch's default ignore index for cross-entropy loss.
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def vocab_shard_bounds(
    global_vocab_size: int, tp_world_size: int, tp_rank: int
) -> tuple[int, int]:
    """The ``[start, end)`` slice of the vocabulary that ``tp_rank`` owns.

    One definition, used by both the vocab-parallel embedding and the
    vocab-parallel loss. They *must* agree -- if the embedding gathers token ids
    from one shard and the loss attributes the target log-probability to another,
    every loss value is wrong while every shape still checks out.

    The shards are even-sized except possibly the last: with ``V=10, tp=3`` the
    chunks are ``[0,4)``, ``[4,8)``, ``[8,10)``. Every rank's start is clamped to
    ``V`` as well as its end, so a TP degree greater than the vocabulary yields
    empty (not negative) slices, which the callers reject explicitly.
    """
    if global_vocab_size < 1:
        raise ValueError(f"global_vocab_size must be >= 1, got {global_vocab_size}")
    if tp_world_size < 1:
        raise ValueError(f"tp_world_size must be >= 1, got {tp_world_size}")
    if not 0 <= tp_rank < tp_world_size:
        raise ValueError(f"tp_rank {tp_rank} is outside [0, {tp_world_size})")
    chunk_size = (global_vocab_size + tp_world_size - 1) // tp_world_size
    start = min(global_vocab_size, chunk_size * tp_rank)
    end = min(global_vocab_size, start + chunk_size)
    return start, max(start, end)


def next_token_targets(labels: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Shift ``labels`` into next-token targets, one document per row.

    ``labels`` is the flat ``(B * T,)`` token stream and ``seq_len`` the row
    length. The model emits ``logits[t]`` predicting ``labels[t + 1]``, and the
    shift is therefore within a *row*, not across the flattened sequence:
    position ``t`` of a row predicts position ``t + 1`` of that same row. The
    last position of every row would predict the first token of the *next*
    document, which is not a prediction the model was given context for, so it
    is marked ``IGNORE_INDEX`` rather than dropped -- keeping the tensor
    rectangular means the ignored entries are excluded from both the loss and
    its denominator (``(targets != IGNORE_INDEX).sum()``), so nothing is
    silently over- or under-counted.
    """
    if labels.numel() % seq_len != 0:
        raise ValueError(
            f"next_token_targets got {labels.numel()} labels, which is not a "
            f"whole number of rows of length {seq_len}"
        )
    targets = torch.full_like(labels, IGNORE_INDEX)
    # Row r of the (B, T) layout occupies [r*T, (r+1)*T); its predictions are the
    # positions [r*T + 1, (r+1)*T) and its targets are [r*T + 1, (r+1)*T).
    targets.view(-1, seq_len)[:, :-1] = labels.view(-1, seq_len)[:, 1:]
    return targets


def cross_entropy_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    *,
    tp_group: dist.ProcessGroup | None = None,
    global_vocab_size: int | None = None,
) -> torch.Tensor:
    """Cross-entropy over ``pred[T, V]`` and ``labels[T]`` with sum reduction.

    The vocab-parallel path is selected by *shape*, not by a flag: it runs
    exactly when the logits hold fewer than ``global_vocab_size`` classes, which
    is what a sharded lm_head produces and nothing else does. A flag could
    disagree with the tensors; ``pred.shape[-1]`` cannot.
    """
    if tp_group is not None and global_vocab_size is not None:
        if pred.shape[-1] != global_vocab_size:
            return _LossParallelCrossEntropy.apply(
                pred.float(), labels, tp_group, global_vocab_size, "sum"
            )

    return F.cross_entropy(
        pred.float(),
        labels,
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


class _LossParallelCrossEntropy(torch.autograd.Function):
    """Vocab-parallel cross-entropy on local ``[T, V_local]`` logits.

    For tensor parallelism that shards the lm_head weight on its vocab dim:
    each rank holds ``V/tp`` output classes and a target token is known to
    exactly one rank, so the softmax denominator has to be assembled across the
    group before any rank can produce a loss.

    Forward uses three TP all-reduces -- max (for a numerically stable shifted
    softmax), sum-of-exp (the denominator), and gather (picking the owner rank's
    log-probability for each target). Backward is fused (NLL + log-softmax
    derivative) with **zero** collectives: each rank already holds the full
    ``[T, V_local]`` log-probability slice it needs the gradient for.

    Supports uneven vocab sharding (the last TP rank may hold fewer classes) and
    ``IGNORE_INDEX`` labels. All inputs and outputs are plain local tensors, not
    DTensors.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
        global_vocab_size: int,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Exact CE from local vocab shards via TP all-reduces.

        ``reduction="sum"`` returns the scalar summed loss. ``reduction="none"``
        returns the per-token NLL ``[T]``, which GRPO negates to get per-token
        logprobs without ever all-gathering the vocabulary.
        """
        logits_dtype = logits.dtype
        logits = logits.float()

        # This rank's slice of the vocabulary, by the same bounds the
        # vocab-parallel embedding uses -- the two must agree or the embedding
        # and the loss disagree about which rank owns which token.
        tp_world_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        vocab_start, vocab_end = vocab_shard_bounds(
            global_vocab_size, tp_world_size, tp_rank
        )
        local_vocab_size = max(0, vocab_end - vocab_start)
        if logits.shape[-1] != local_vocab_size:
            raise ValueError(
                "_LossParallelCrossEntropy expected local vocab size "
                f"{local_vocab_size} for global vocab size {global_vocab_size}, "
                f"got {logits.shape[-1]}."
            )
        if local_vocab_size == 0:
            raise ValueError(
                "_LossParallelCrossEntropy does not support empty vocab shards. "
                f"Global vocab {global_vocab_size} is smaller than the TP degree "
                f"{tp_world_size}."
            )

        # A label outside [0, global_vocab_size) would be silently dropped by the
        # shard mask below rather than reported, so check it on device -- no host
        # sync, and no invalid target can reach the gather.
        torch._assert_async(
            torch.all(
                (labels == IGNORE_INDEX)
                | ((labels >= 0) & (labels < global_vocab_size))
            ),
            f"labels must be {IGNORE_INDEX} or in [0, {global_vocab_size})",
        )

        # All-reduce max for a numerically stable distributed log-softmax.
        local_max = torch.amax(logits, dim=-1, keepdim=True)
        local_max = funcol.all_reduce(
            local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group
        )

        # All-reduce the shifted sum-of-exp: the global softmax denominator.
        shifted = logits - local_max
        shifted_sumexp = torch.sum(torch.exp(shifted), dim=-1, keepdim=True)
        shifted_sumexp = funcol.all_reduce(
            shifted_sumexp, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )
        log_probs = shifted - torch.log(shifted_sumexp)

        # Mask labels outside this shard; the all-reduce below then selects the
        # owner rank's log-probability for each target.
        safe_labels = torch.where(labels != IGNORE_INDEX, labels, 0)
        out_of_range = (safe_labels < vocab_start) | (
            safe_labels >= vocab_start + local_vocab_size
        )
        local_labels = safe_labels - vocab_start
        local_labels[out_of_range] = 0

        local_result = torch.gather(log_probs, -1, local_labels.unsqueeze(-1))
        local_result[out_of_range.unsqueeze(-1)] = 0
        local_result = funcol.all_reduce(
            local_result, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )

        # Per-token NLL, with ignored labels zeroed (their log-prob is 0 above).
        result = -local_result.squeeze(-1)
        result = torch.where(labels != IGNORE_INDEX, result, 0)

        ctx.save_for_backward(log_probs, labels)
        ctx.logits_dtype = logits_dtype
        ctx.vocab_start = vocab_start
        ctx.local_vocab_size = local_vocab_size
        ctx.reduction = reduction
        if reduction == "none":
            return result
        return result.sum()

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        log_probs, labels = ctx.saved_tensors
        safe_labels = torch.where(labels != IGNORE_INDEX, labels, 0)
        out_of_range = (safe_labels < ctx.vocab_start) | (
            safe_labels >= ctx.vocab_start + ctx.local_vocab_size
        )
        local_labels = safe_labels - ctx.vocab_start
        local_labels[out_of_range] = 0

        # d/dz [ -log_softmax(z)_y ] = softmax(z) - onehot(y), assembled only for
        # this rank's slice -- targets outside the shard contribute
        # softmax(z) * 1, which is what the out_of_range branch adds back.
        grad_input = torch.zeros_like(log_probs)
        row_idx = torch.arange(local_labels.shape[0], device=local_labels.device)
        grad_update = out_of_range.to(grad_input.dtype) - 1.0
        grad_input[row_idx, local_labels] = grad_update

        # reduction="none" hands back a per-token [T] upstream grad; unsqueeze it
        # to broadcast over the local vocab. "sum" gives the scalar, which does.
        if ctx.reduction == "none":
            grad_output = grad_output.unsqueeze(-1)
        grad_output = torch.where(
            (labels != IGNORE_INDEX).unsqueeze(-1), grad_output, 0
        )
        grad_logits = (grad_input + torch.exp(log_probs)) * grad_output
        return grad_logits.to(ctx.logits_dtype), None, None, None, None


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MSE loss with sum reduction, for models trained on continuous targets."""
    return F.mse_loss(pred.float(), labels.float().detach(), reduction="sum")


def compute_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    tp_group: dist.ProcessGroup | None = None,
    global_vocab_size: int | None = None,
    return_entropy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probabilities from ``logits[T, V]`` and ``labels[T]``.

    With a sharded vocabulary each rank holds only its own classes, so the
    cross-entropy has to run through the vocab-parallel path before the gather.
    Reachability note: nothing in hpmesh shards the lm_head yet, so today this
    takes the plain path -- it is here because the local-vocab case is the whole
    reason hpmesh has an ``_LossParallelCrossEntropy`` at all.

    When ``return_entropy`` is set, also returns per-token Shannon entropy
    ``H(p) = logsumexp(logits) - sum(softmax(logits) * logits)``, shape ``[T]``.
    Entropy is a metric only, so it is computed under ``no_grad``: it contributes
    no gradient and must not extend the autograd graph over the logits softmax.

    Returns ``logprobs``, or ``(logprobs, entropy)`` when ``return_entropy``.
    """
    if tp_group is not None and global_vocab_size is not None:
        if logits.shape[-1] != global_vocab_size:
            # reduction="none" is the vocab-parallel path's per-token form: it
            # returns -NLL directly, so no vocab all-gather is needed here.
            return -_LossParallelCrossEntropy.apply(
                logits, labels, tp_group, global_vocab_size, "none"
            )

    # One bf16 -> fp32 upcast, shared by the logprobs and (if asked) the entropy.
    logits = logits.float()
    logprobs = -F.cross_entropy(
        logits,
        labels,
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    if not return_entropy:
        return logprobs
    with torch.no_grad():
        entropy = torch.logsumexp(logits, dim=-1) - (
            torch.softmax(logits, dim=-1) * logits
        ).sum(dim=-1)
    return logprobs, entropy
