"""The validation (eval) pass, extracted from ``trainer.py``.

These are module-level functions whose first parameter is deliberately named
``self``: the bodies moved here verbatim from ``Trainer``, and ``Trainer``
keeps same-named thin delegates (``should_validate`` / ``validate`` /
``validate_body`` / ``check_validation_feasibility``) so its public surface,
error types and message texts are unchanged. Everything the bodies touch --
``dp_rank_world_size``, ``loss_sum``, ``metrics``, ... -- stays on the
trainer; this module owns only the pass itself.
"""

from __future__ import annotations

import torch

from llmtuner.accelerator.capabilities import require
from llmtuner.config import ValidationConfig
from llmtuner.parallel import matrix

from ..accelerator.dist import all_reduce
from ..accelerator.spmd_context import spmd_context
from ..datasets import build_dataloader
from ..datasets.random_data import DataLoaderExhausted
from ..datasets.types import Batch


def check_validation_feasibility(
    validation: ValidationConfig,
    *,
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
        """
    if validation.steps != -1:
        return
    if dp_world_size > 1:
        matrix.validation_once_requires_dp1(dp_world_size)
    dataset = (
        training_dataset if validation.dataset is None else validation.dataset
    )
    if dataset == "random":
        matrix.validation_once_requires_finite_corpus()

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
        self.validate_body(validation, step)
    finally:
        for part in self.model_parts:
            part.train()

def validate_body(self, validation: ValidationConfig, step: int) -> None:
    parallel_dims = self.parallel_dims
    if parallel_dims is not None and parallel_dims.pp_enabled:
        validate_body_pp(self, validation, step)
        return
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

    dp_rank, dp_world_size = self.dp_rank_world_size()
    batch_size_per_rank = self.batch_size_per_rank(dp_world_size)
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
            local_valid_tokens = self.count_valid_tokens(batch)
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
            inputs, labels, extra_kwargs = self.example_model.preprocess_inputs(
                self.to_device(batch),
                parallel_dims=self.parallel_dims,
                parallelism=self.cfg.parallel,
                max_context_length=self.cfg.max_seq_len,
            )
            with self.param_context(), spmd_context(self.parallel_dims):
                logits = self.example_model(inputs, **extra_kwargs)
                loss_sum = self.loss_sum(logits, labels, **self.loss_vocab_kwargs())
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


def validate_body_pp(self, validation: ValidationConfig, step: int) -> None:
    """The PP validation pass: drive the schedule's ``eval`` per step.

    Upstream torchtitan's ``Validator`` seam: the pipeline schedule carries an
    eval-only driver, so a validation pass runs the same stages forward-only.
    The microbatch plumbing mirrors the training body (``pp_steps``): first
    stage gets the inputs, last stage the labels, and every stage gets the
    per-microbatch kwargs (positions, masks).

    The schedule's loss function divides by the run's denominator attribute;
    for the pass it is pinned to 1 so the reported per-microbatch losses are
    raw summed cross-entropies, and the division happens once at the end, by
    the pass's global valid-token count -- the same normalization as the
    non-PP path and the training loss. The training body re-publishes the
    real denominator before every train step, so no restore is needed.

    Only the last stage's ranks hold losses; every other stage's ranks run
    the same loop (same loader reads, same token counting, same eval calls)
    and simply do not report.
    """
    require("pipelining_schedule_eval", feature="validation with pipeline parallelism")

    parallel_dims = self.parallel_dims
    dp_mesh = parallel_dims.get_optional_mesh("dp")
    loss_sharded = parallel_dims.dp_cp_enabled or parallel_dims.tp_enabled
    loss_mesh = parallel_dims.get_optional_mesh("loss") if loss_sharded else dp_mesh

    dp_rank, dp_world_size = self.dp_rank_world_size()
    batch_size_per_rank = self.batch_size_per_rank(dp_world_size)
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
            self.metrics.add_tokens(labels.numel())
            local_valid_tokens = self.count_valid_tokens(batch)
            global_valid_tokens = torch.tensor(
                local_valid_tokens, dtype=torch.int64, device=self.device
            )
            if dp_mesh is not None:
                all_reduce(global_valid_tokens, group=dp_mesh.get_group())

            arg_mbs: list[tuple[torch.Tensor, ...]] = []
            kwarg_mbs: list[dict] = []
            target_mbs: list[torch.Tensor] | None = (
                [] if self.pp_has_last_stage else None
            )
            for mb in self.pp_microbatches(batch):
                inputs, mb_labels, extra_kwargs = self.preprocess({"batch": mb})
                if self.pp_has_first_stage:
                    arg_mbs.append((inputs,))
                kwarg_mbs.append(extra_kwargs)
                if target_mbs is not None:
                    target_mbs.append(mb_labels)

            losses: list[torch.Tensor] | None = [] if self.pp_has_last_stage else None
            with self.param_context(), spmd_context(self.parallel_dims):
                self.pp_schedule._llmtuner_global_valid_tokens = torch.ones(
                    (), dtype=torch.float32
                )
                self.pp_schedule.eval(
                    arg_mbs=arg_mbs if self.pp_has_first_stage else None,
                    kwarg_mbs=kwarg_mbs,
                    target_mbs=target_mbs,
                    losses=losses,
                )
            if self.pp_has_last_stage:
                assert losses is not None
                step_loss = torch.sum(torch.stack([loss.detach() for loss in losses]))
                if accumulated_loss is None:
                    accumulated_loss = step_loss
                else:
                    accumulated_loss = accumulated_loss + step_loss
            total_global_valid_tokens.add_(global_valid_tokens)
            num_steps += 1
    finally:
        validation_dataloader.close()

    if num_steps == 0:
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
    if not self.pp_has_last_stage:
        return
    assert accumulated_loss is not None
    if loss_mesh is not None:
        global_loss_sum = accumulated_loss.clone()
        all_reduce(global_loss_sum, group=loss_mesh.get_group())
    else:
        global_loss_sum = accumulated_loss
    global_avg_loss = float(global_loss_sum) / num_global_valid_tokens
    self.metrics.log_validation(loss=global_avg_loss, step=step)
