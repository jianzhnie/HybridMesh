"""The validation (eval) pass, extracted from ``trainer.py``.

These are module-level functions whose first parameter is deliberately named
``self``: the bodies moved here verbatim from ``Trainer``, and ``Trainer``
keeps same-named thin delegates (``should_validate`` / ``validate`` /
``_validate_body`` / ``_check_validation_feasibility``) so its public surface,
error types and message texts are unchanged. Everything the bodies touch --
``_dp_rank_world_size``, ``_loss_sum``, ``metrics``, ... -- stays on the
trainer; this module owns only the pass itself.
"""

from __future__ import annotations

import torch

from hpmesh.config import ValidationConfig

from ..accelerator.dist import all_reduce
from ..accelerator.spmd_context import spmd_context
from ..datasets import build_dataloader
from ..datasets.random_data import Batch, DataLoaderExhausted


def _check_validation_feasibility(
    validation: ValidationConfig,
    *,
    pp_enabled: bool,
    dp_world_size: int,
    training_dataset: str,
) -> None:
    """Reject the validation configurations that cannot terminate cleanly.

        Runs at trainer build time, where the real parallel degrees are known
        (config-level ``__post_init__`` cannot see them: ``dp_shard`` defaults
        to the derive-me marker ``-1``). Each rejected combination would
        otherwise fail later and worse:

        * ``steps=-1`` consumes the finite dataset once, so every rank stops
          when its own shard is exhausted. With DP > 1 the ranks can exhaust at
          different iterations and hang on the pass's collectives (the token
          and loss reductions every rank must enter together).
        * ``steps=-1`` against the synthetic corpus has no exhaustion at all:
          the random source is infinite, so "one finite pass" never ends.
        * Pipeline parallelism drives the schedule through a train-shaped seam
          (the loss is computed and backwarded *inside* the schedule step);
          there is no eval-only pipeline path to run a validation pass
          through, so the combination loud-raises rather than silently
          skipping validation or training on the pass.
        """
    if pp_enabled:
        raise NotImplementedError(
            "validation with pipeline parallelism is not supported: "
            "hpmesh drives the pipeline schedule through its training "
            "seam, where the last stage's loss is computed and backwarded "
            "inside the schedule step. There is no eval-only pipeline "
            "path; run validation with pipeline_parallel_size=1."
        )
    if validation.steps != -1:
        return
    if dp_world_size > 1:
        raise ValueError(
            "validation.steps=-1 runs one finite pass over the dataset "
            "(the loader is built with repeat=False). With data-parallel "
            f"degree > 1 ({dp_world_size}), ranks can exhaust at different "
            "iterations and hang on the validation collectives. Set "
            "validation.steps to a positive count so every rank runs the "
            "same number of batches, or run with data-parallel degree 1."
        )
    dataset = (
        training_dataset if validation.dataset is None else validation.dataset
    )
    if dataset == "random":
        raise ValueError(
            "validation.steps=-1 consumes the dataset once, but the "
            "'random' corpus is an infinite synthetic source that never "
            "exhausts. Set validation.steps to a positive count, or name a "
            "finite validation dataset."
        )

def should_validate(self, step: int) -> bool:
    """Whether a validation pass runs at the end of ``step``.

        Step 1 always validates (a run sees its first eval number immediately,
        which is the cheap sanity check that the eval path works at all);
        after that, every ``validation.freq`` steps.
        """
    validation = self.cfg.validation
    return validation is not None and (
        step == 1 or step % validation.freq == 0
    )

@torch.no_grad()
def validate(self, step: int) -> None:
    """Run one eval-mode, gradient-free pass and log its loss.

        The reported number is the pass's summed next-token cross-entropy
        divided by the *global* valid-token count -- the same normalization as
        the training loss, over the same two meshes (tokens reduced across DP,
        the loss sum across the dp*cp*tp ``loss`` view), so the eval and train
        numbers are directly comparable and identical on every rank.

        The pass is a pure observer: the model runs in eval mode (restored to
        train mode afterwards, even on error), no gradients are computed, no
        optimizer or scheduler state moves, and ``ntokens_seen`` -- the
        checkpointed training counter -- is not touched.

        The dataloader is built fresh per pass and closed when the pass ends:
        it is a temporary read over the corpus, not training state, so it is
        neither checkpointed nor shared with the training loader. ``steps=-1``
        reads it to exhaustion (built with repeat=False); a positive ``steps``
        bounds the pass (built repeating, so the bound is always reachable).

        Two outcomes are loud errors rather than a silently skipped report: a
        pass that read zero batches (the dataset supplied less than one batch
        of tokens on this rank, which concat-then-split packing turns into no
        rows at all), and a pass over zero valid tokens (every label masked),
        which has no average to report.
        """
    validation = self.cfg.validation
    assert validation is not None, "validate() is gated by should_validate"

    for part in self.model_parts:
        part.eval()
    try:
        self._validate_body(validation, step)
    finally:
        for part in self.model_parts:
            part.train()

def _validate_body(self, validation: ValidationConfig, step: int) -> None:
    parallel_dims = self.parallel_dims
    # The same mesh split as ``train_step``: the token count is taken from
    # the unsharded batch, so it is summed over the dp axis alone; the loss
    # is summed over each rank's own slice of the batch, so it is reduced
    # over the dp*cp*tp ``loss`` view when the sequence is sharded at all.
    dp_mesh = (
        None if parallel_dims is None else parallel_dims.get_optional_mesh("dp")
    )
    loss_sharded = parallel_dims is not None and (
        parallel_dims.dp_cp_enabled or parallel_dims.tp_enabled
    )
    loss_mesh = (
        None
        if parallel_dims is None
        else (
            parallel_dims.get_optional_mesh("loss") if loss_sharded else dp_mesh
        )
    )

    dp_rank, dp_world_size = self._dp_rank_world_size()
    batch_size_per_rank = self._batch_size_per_rank(dp_world_size)
    validation_dataloader = build_dataloader(
        self.cfg,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        num_tokens_per_batch=batch_size_per_rank * self.cfg.max_seq_len,
        repeat=validation.steps != -1,
        dataset=validation.dataset,
    )

    accumulated_loss: torch.Tensor | None = None
    total_global_valid_tokens = torch.zeros(
        (), dtype=torch.int64, device=self.device
    )
    num_steps = 0
    try:
        data_iterator = iter(validation_dataloader)
        while validation.steps == -1 or num_steps < validation.steps:
            try:
                batch = next(data_iterator)
            except (DataLoaderExhausted, StopIteration):
                break
            labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
            # Throughput accounting only, mirroring ``batch_generator``:
            # every label the loader produced counts, whether or not it is
            # predictable. ``ntokens_seen`` is deliberately not touched --
            # it is the checkpointed *training* counter.
            self.metrics.add_tokens(labels.numel())
            # Counted from the unsharded batch, exactly as in training, so
            # the dp-axis reduction below counts the whole batch once even
            # when CP later slices the sequence.
            local_valid_tokens = self._count_valid_tokens(batch)
            global_valid_tokens = torch.tensor(
                local_valid_tokens, dtype=torch.int64, device=self.device
            )
            if dp_mesh is not None:
                all_reduce(global_valid_tokens, group=dp_mesh.get_group())
            if isinstance(batch, dict):
                # ``num_valid_tokens`` is the trainer's bookkeeping; a
                # plain int among tensors would be splatted into the model
                # forward as a kwarg.
                batch.pop("num_valid_tokens", None)
            inputs, labels, extra_kwargs = self._example_model.preprocess_inputs(
                self._to_device(batch),
                parallel_dims=self.parallel_dims,
                parallelism=self.cfg.parallel,
                max_context_length=self.cfg.max_seq_len,
            )
            with self._param_context(), spmd_context(self.parallel_dims):
                logits = self._example_model(inputs, **extra_kwargs)
                loss_sum = self._loss_sum(logits, labels)
            if accumulated_loss is None:
                accumulated_loss = loss_sum.clone()
            else:
                accumulated_loss.add_(loss_sum)
            total_global_valid_tokens.add_(global_valid_tokens)
            num_steps += 1
    finally:
        # Releases the Grain prefetch thread; a no-op for loaders without
        # one. The loader is temporary, so nothing else holds it open.
        validation_dataloader.close()

    if accumulated_loss is None:
        raise ValueError(
            "Validation ran zero batches on this rank. This happens when "
            "the validation dataset supplies fewer than one batch of "
            "tokens on this rank, because concat-then-split packing drops "
            "partially filled batches. Decrease the per-rank batch size or "
            "use a larger validation dataset."
        )
    num_global_valid_tokens = int(total_global_valid_tokens.item())
    if num_global_valid_tokens == 0:
        raise ValueError(
            "Validation ran on zero valid tokens; cannot compute an "
            "average validation loss. Ensure the validation batches "
            "contain unmasked labels."
        )
    if loss_mesh is not None:
        global_loss_sum = accumulated_loss.clone()
        all_reduce(global_loss_sum, group=loss_mesh.get_group())
    else:
        global_loss_sum = accumulated_loss
    global_avg_loss = float(global_loss_sum) / num_global_valid_tokens
    self.metrics.log_validation(loss=global_avg_loss, step=step)
