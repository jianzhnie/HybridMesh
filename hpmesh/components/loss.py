"""Loss functions: plain next-token CE, and its vocab-parallel form.

Vendored from torchtitan ``components/loss.py``. The arithmetic is unchanged.
The functions are called directly rather than through the ``BaseLoss`` /
``CrossEntropyLoss`` / ``MSELoss`` hierarchy, and the ``spmd.assert_type``
annotations are gone -- hpmesh configures by argument and checks shapes, not
SPMD types. Concretely, the vocab-parallel path is selected by comparing
``pred.shape[-1]`` with ``global_vocab_size`` rather than by upstream's
``spmd_mesh_size("tp") > 1``, which keeps a caller that has no SPMD context
working and cannot disagree with the tensors in front of it.

Two names here are hpmesh's rather than upstream's: ``next_token_targets`` and
``vocab_shard_bounds``. Nothing in torchtitan defines either. ``next_token_targets``
is the row-wise label shift the trainer applies; ``vocab_shard_bounds`` is the
bound formula lifted out of upstream's ``_LossParallelCrossEntropy.forward`` so
the vocab-parallel embedding and the vocab-parallel loss cannot disagree about
which rank owns which token. ``IGNORE_INDEX`` is genuinely upstream.

Three entry points, and the difference between them is worth stating plainly:

* ``cross_entropy_loss`` -- the loss, with a sum reduction, so the caller can
  divide by a *global* token count once the per-rank counts have been reduced.
* ``chunked_lm_head_cross_entropy`` -- the same summed CE, run in sequence
  chunks so peak logits memory is bounded; it runs its own backward.
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

from ..accelerator import dist_utils
from ..utils.batch_invariant import is_in_batch_invariant_mode

__all__ = [
    "IGNORE_INDEX",
    "LossFunction",
    "chunked_lm_head_cross_entropy",
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


def _shard_local_labels(
    labels: torch.Tensor, vocab_start: int, local_vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global labels to shard-local indices, masking non-owned tokens.

    Returns ``(local_labels, out_of_range)``: ``local_labels`` shifted by
    ``vocab_start`` with out-of-shard and ``IGNORE_INDEX`` entries clamped to 0
    (safe to index with), and ``out_of_range`` marking exactly those entries.
    Shared by forward and backward so both attribute a target to the same shard.
    """
    safe_labels = torch.where(labels != IGNORE_INDEX, labels, 0)
    out_of_range = (safe_labels < vocab_start) | (
        safe_labels >= vocab_start + local_vocab_size
    )
    local_labels = safe_labels - vocab_start
    local_labels[out_of_range] = 0
    return local_labels, out_of_range


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
        tp_world_size = dist_utils.get_world_size(tp_group)
        tp_rank = dist_utils.get_rank(tp_group)
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
        local_labels, out_of_range = _shard_local_labels(
            labels, vocab_start, local_vocab_size
        )

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
        local_labels, out_of_range = _shard_local_labels(
            labels, ctx.vocab_start, ctx.local_vocab_size
        )

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


def chunked_lm_head_cross_entropy(
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_chunks: int,
    grad_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Summed next-token CE, computed in sequence chunks to bound peak memory.

    Materializing ``lm_head(hidden_states)`` costs ``T * V`` floats at once,
    which with a 150k vocabulary dwarfs the rest of the step. This splits the
    token axis into ``num_chunks`` pieces and runs ``lm_head`` + cross-entropy
    on one piece at a time, so the peak logits memory is ``T * V / num_chunks``
    (the accumulated gradients are ``T * H`` and ``V * H`` -- independent of
    the chunking).

    The function runs the backward ITSELF and returns the detached summed loss
    (un-normalized, the same sum reduction ``cross_entropy_loss`` uses). The
    backward is per chunk -- that is what keeps the peak low, since each
    chunk's logits are freed before the next chunk's are computed -- and each
    chunk's backward is scaled by ``grad_scale`` (the trainer passes
    ``1 / global_valid_tokens``), so the accumulated gradients equal those of
    ``(full_sum * grad_scale).backward()``. Chunking a sum and backwarding the
    pieces is exact: one chunk's logits gradient does not depend on any other
    chunk.

    Two rounds of backward happen: the per-chunk ones (which accumulate the
    lm_head weight gradient and each chunk's hidden-state gradient), then a
    single ``torch.autograd.backward(hidden_states, assembled_grads)`` that
    propagates the assembled ``T * H`` gradient through the decoder in one
    pass -- the decoder graph is traversed once, not once per chunk.

    Chunks need not divide the sequence: ``torch.chunk`` leaves a short final
    piece, and the sum reduction makes the split invisible to the value. Under
    CP/TP the hidden states are already a shard of the sequence along the
    token axis; chunking that local shard composes with the sum reduction the
    same way.

    ``hidden_states`` must require grad -- this is a training path, and a
    silent no-backward would look like a working step.
    """
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must be (T, H), got shape {tuple(hidden_states.shape)}"
        )
    if labels.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} does not match the token axis of "
            f"hidden_states ({hidden_states.shape[0]})"
        )
    if not hidden_states.requires_grad:
        raise ValueError(
            "chunked_lm_head_cross_entropy is a training path: hidden_states "
            "must require grad."
        )

    hidden_chunks = hidden_states.chunk(num_chunks, dim=0)
    label_chunks = labels.chunk(num_chunks, dim=0)
    total = hidden_states.new_zeros((), dtype=torch.float32)
    hidden_grads: list[torch.Tensor] = []
    for hidden_chunk, label_chunk in zip(hidden_chunks, label_chunks, strict=True):
        # detach + requires_grad_ makes the chunk a leaf, so its backward
        # stops at the lm_head boundary and the chunk's hidden gradient lands
        # in ``.grad`` for assembly below.
        detached = hidden_chunk.detach().requires_grad_(True)
        logits = lm_head(detached)
        chunk_loss = F.cross_entropy(
            logits.float(),
            label_chunk,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
        total = total + chunk_loss.detach()
        (chunk_loss * grad_scale).backward()
        assert detached.grad is not None
        hidden_grads.append(detached.grad)

    torch.autograd.backward(hidden_states, grad_tensors=torch.cat(hidden_grads))
    return total


def _vocab_parallel_entropy(
    logits: torch.Tensor, tp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Exact per-token Shannon entropy from vocab-sharded logits, no gather.

    ``H(p) = logsumexp(z) - sum(softmax(z) * z)`` needs two cross-shard sums:
    the softmax denominator and the first moment ``sum(p * z)``. Both are
    assembled with all-reduces against a shared max, so the vocabulary never
    has to be gathered to compute the metric.
    """
    logits = logits.float()

    local_max = torch.amax(logits, dim=-1, keepdim=True)
    global_max = funcol.all_reduce(
        local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group
    )

    shifted = logits - global_max
    shifted_exp = torch.exp(shifted)
    local_sumexp = shifted_exp.sum(dim=-1)
    # Avoid 0 * -inf for finite distributions with masked logits while
    # preserving NaNs for invalid distributions such as all -inf logits.
    shifted_weighted = torch.where(
        torch.isneginf(shifted),
        torch.zeros_like(shifted),
        shifted_exp * shifted,
    )
    local_weighted_sum = shifted_weighted.sum(dim=-1)
    global_stats = funcol.all_reduce(
        torch.stack((local_sumexp, local_weighted_sum)),
        reduceOp=dist.ReduceOp.SUM.name,
        group=tp_group,
    )
    sumexp, weighted_sum = global_stats.unbind()
    return torch.log(sumexp) - weighted_sum / sumexp


class _GatherVocabShards(torch.autograd.Function):
    """All-gather vocab-sharded logits, with a slice backward.

    The gathered ``[T, V]`` logits are identical on every rank, so the gradient
    of the downstream loss w.r.t. this rank's shard is the slice of the shared
    full-vocab gradient the shard contributed -- a slice, not an all-reduce,
    which would over-count by the TP degree.

    ``vocab_shard_bounds`` gives the last rank a short shard when ``V`` is not
    divisible by ``tp``, so the collective sees equal sizes by padding to the
    chunk width first and trimming afterwards. The collective runs inside the
    Function's ``forward`` (grad mode off): functional collectives register
    their own backward otherwise, which is both the wrong gradient here and a
    version-sensitive path.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        tp_group: dist.ProcessGroup,
        global_vocab_size: int,
    ) -> torch.Tensor:
        tp_world_size = dist_utils.get_world_size(tp_group)
        tp_rank = dist_utils.get_rank(tp_group)
        ctx.vocab_start, ctx.vocab_end = vocab_shard_bounds(
            global_vocab_size, tp_world_size, tp_rank
        )
        chunk_size = -(-global_vocab_size // tp_world_size)
        padded = F.pad(logits, (0, chunk_size - logits.shape[-1]))
        gathered = funcol.all_gather_tensor(padded, gather_dim=-1, group=tp_group)
        return gathered[..., :global_vocab_size]

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        return (
            grad_output[..., ctx.vocab_start : ctx.vocab_end].contiguous(),
            None,
            None,
        )


def _gather_vocab_shards(
    logits: torch.Tensor, tp_group: dist.ProcessGroup, global_vocab_size: int
) -> torch.Tensor:
    """All-gather vocab-sharded ``[T, V_local]`` logits into full ``[T, V]``."""
    return _GatherVocabShards.apply(logits, tp_group, global_vocab_size)


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
    log-probabilities come from the vocab-parallel path and the entropy from
    ``_vocab_parallel_entropy`` -- neither gathers the vocabulary. In
    batch-invariant mode the shards are gathered first instead, so the trainer
    performs the same operation sequence over full-vocab logits as an
    inference generator does. Reachability note: nothing in hpmesh shards the
    lm_head yet, so today this takes the plain path -- it is here because the
    local-vocab case is the whole reason hpmesh has an
    ``_LossParallelCrossEntropy`` at all.

    When ``return_entropy`` is set, also returns per-token Shannon entropy
    ``H(p) = logsumexp(logits) - sum(softmax(logits) * logits)``, shape ``[T]``.
    Entropy is a metric only, so it is computed under ``no_grad``: it contributes
    no gradient and must not extend the autograd graph over the logits softmax.

    Returns ``logprobs``, or ``(logprobs, entropy)`` when ``return_entropy``.
    """
    if tp_group is not None and global_vocab_size is not None:
        if logits.shape[-1] != global_vocab_size:
            if is_in_batch_invariant_mode():
                logits = _gather_vocab_shards(logits, tp_group, global_vocab_size)
            else:
                # reduction="none" is the vocab-parallel path's per-token form:
                # it returns -NLL directly, so no vocab all-gather is needed.
                logprobs = -_LossParallelCrossEntropy.apply(
                    logits, labels, tp_group, global_vocab_size, "none"
                )
                if not return_entropy:
                    return logprobs
                with torch.no_grad():
                    entropy = _vocab_parallel_entropy(logits, tp_group)
                return logprobs, entropy

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
