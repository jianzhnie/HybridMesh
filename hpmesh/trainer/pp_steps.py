"""Pipeline-parallel step machinery, extracted from ``trainer.py``.

Microbatch row-splitting (``_pp_microbatches``) and the schedule-driving
forward/backward body (``_pp_forward_backward_body``) live here; the trainer
keeps same-named thin delegates so call sites and tests are unchanged. The
first parameter stays named ``self`` -- the bodies moved verbatim and keep
reading the trainer's ``pp_schedule`` / ``pp_has_*`` / ``_pp_loss_sentinel``
state, which is built in ``Trainer.__init__`` and stays there.
"""

from __future__ import annotations

from typing import Any

import torch

from ..accelerator.spmd_context import spmd_context
from ..datasets.loader import TrainerBatch
from ..datasets.random_data import Batch


def _pp_microbatches(self, batch: Batch | TrainerBatch) -> list[dict[str, Any]]:
    """Split the rank's batch into the schedule's micro-batches.

        Rows are split, never tokens: each micro-batch is collapsed with the
        same semantics as the non-PP body, so every micro-batch holds whole
        documents and its loss is the same summed CE. Divisibility is enforced
        at setup (``apply_pp``), so ``chunk`` never leaves a short final piece.

        The split happens here rather than inside ``preprocess_inputs``, which
        is a deliberate divergence from the reference: torchtitan's protocol
        returns a *list* of micro-batches, but hpmesh's PP path row-chunks one
        batch after the model has already collapsed it, and splitting inside the
        model would make every other caller of that method carry a batch dim it
        does not want. Keeping the loop holding rows also means the model's
        seam has exactly one shape contract.
        """
    raw = batch.labels if isinstance(batch, Batch) else batch["labels"]
    num_microbatches = self.cfg.parallel.num_pp_microbatches
    if isinstance(batch, dict):
        total_rows = raw.shape[0]
        rows_per_mb = total_rows // num_microbatches
        mbs = []
        for index in range(num_microbatches):
            chunk = {
                key: (
                    value[index * rows_per_mb : (index + 1) * rows_per_mb]
                    if isinstance(value, torch.Tensor) and value.ndim > 0
                    else value
                )
                for key, value in batch.items()
            }
            mbs.append(chunk)
        return mbs
    input_chunks = batch.input_ids.chunk(num_microbatches, dim=0)
    label_chunks = batch.labels.chunk(num_microbatches, dim=0)
    return [
        Batch(input_ids=ids, labels=labels)
        for ids, labels in zip(input_chunks, label_chunks, strict=True)
    ]

def _pp_forward_backward_body(
    self,
    batch: Batch | TrainerBatch,
    *,
    global_valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """The pipeline-parallel body: drive the schedule instead of the model.

        Only the first stage is handed the inputs (``arg_mbs``) and only the
        last the labels (``target_mbs``); intermediate stages receive the
        previous stage's activations over the schedule's p2p channel. Every
        stage preprocesses its own micro-batches, because a non-first stage's
        chunk holds hidden states rather than token ids and only the model knows
        which of the two it is looking at.

        The schedule's loss is the same summed next-token CE the non-PP body
        computes (``pipeline_parallel/apply.py:_scalar_loss_fn``), so the return
        keeps the caller's normalization unchanged: the sum over the last
        stage's micro-batches. That sum is over the last stage's *own* shard of
        the sequence, which is why the caller's denominator -- counted before
        the sequence was cut up -- is the right one.

        ``global_valid_tokens`` is threaded to the schedule through
        ``loss_kwargs``, where the loss function divides by it before the
        schedule's backward. The losses the schedule reports are therefore
        sum/G, and they are multiplied back by G here so the caller keeps
        receiving the raw sum it normalizes and reports.

        The token count is not taken here: the caller needs it before the
        micro-batches are cut, and a stage's count would be over its own slice.
        """
    arg_mbs: list[tuple[torch.Tensor, ...]] = []
    kwarg_mbs: list[dict[str, Any]] = []
    target_mbs: list[torch.Tensor] | None = [] if self.pp_has_last_stage else None
    for mb in self._pp_microbatches(batch):
        inputs, labels, extra_kwargs = self._preprocess({"batch": mb})
        if self.pp_has_first_stage:
            arg_mbs.append((inputs,))
        kwarg_mbs.append(extra_kwargs)
        if target_mbs is not None:
            target_mbs.append(labels)

    losses: list[torch.Tensor] | None = [] if self.pp_has_last_stage else None
    with self._param_context(), spmd_context(self.parallel_dims):
        # ``_step_microbatches`` is the Torch 2.10-compatible equivalent
        # of the older public ``step(arg_mbs=..., kwarg_mbs=...)`` seam.
        # The public API would split the already-split lists as kwargs and
        # attempts to shard scalar loss kwargs along dimension 0.
        if hasattr(self.pp_schedule, "_step_microbatches"):
            self.pp_schedule._hpmesh_global_valid_tokens = global_valid_tokens
            self.pp_schedule._step_microbatches(
                arg_mbs if self.pp_has_first_stage else None,
                kwarg_mbs,
                target_mbs,
                losses,
                return_outputs=False,
            )
        else:
            self.pp_schedule.step(
                arg_mbs=arg_mbs if self.pp_has_first_stage else None,
                kwarg_mbs=kwarg_mbs,
                target_mbs=target_mbs,
                losses=losses,
                loss_kwargs={"global_valid_tokens": global_valid_tokens},
                return_outputs=False,
            )

    if self.pp_has_last_stage:
        assert losses is not None
        assert global_valid_tokens is not None
        # Backward has consumed these losses. Report detached views, then
        # release the originals and their autograd graphs.
        detached_losses = [loss.detach() for loss in losses]
        losses.clear()
        return torch.sum(torch.stack(detached_losses)) * global_valid_tokens
    # Not the last stage: there is no loss here, and the caller's own loss
    # sum must stay a real sum on every rank so the finiteness reduction --
    # which every rank joins -- sees the same shape everywhere. Finite by
    # construction, and never logged, because the metrics rank is a
    # last-stage rank.
    return self._pp_loss_sentinel
