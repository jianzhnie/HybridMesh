"""Trainer -- the single training loop, shared by every learning step.

Shape vendored from torchtitan ``trainer.py``: ``train`` -> ``train_step`` ->
``forward_backward_step`` -> ``_forward_backward_body``, one function per level
of the step, so each can be read and tested on its own. The distributed
complexity still lives in ``parallel/``; the loop is meant to read end to end.

What the migration added, and why each earned its place:

* **Token-normalized loss.** The loss is a SUM over predicted tokens divided by
  the token count reduced across DP. That makes the reported number independent
  of how the batch was split across ranks, and it is the precondition for
  gradient accumulation (in progress) to sum correctly.
* **Gradient clipping + ``grad_norm`` reporting.** ``clip_grad_norm_`` reduces
  the norm across PP stages before clipping, which ``torch.nn.utils`` cannot do
  because each stage holds disjoint parameters.
* **A finiteness check.** A NaN loss or gradient looked exactly like a healthy
  step: training continued and every later number was garbage. This stops at the
  first bad step instead, and does it with an on-device check so it neither
  synchronizes (unlike ``.item()``) nor becomes a CUDA-graph break.
* **Checkpoints**, so a run can be resumed rather than restarted. The loop only
  drives the manager (``components/checkpointer``) -- it decides *when* to save
  and load; the manager owns *how*, including the interval and retention
  policies. ``Trainer.state_dict``/``load_state_dict`` are what make the step
  and token counters part of the checkpoint.

What was NOT ported: the component system (``Configurable``, ``model_spec``,
metrics processor, ``sdc_replayer``, profiler, validator, CUDA graphs). Those are
infrastructure the loop calls into, not loop logic, and hpmesh has no
counterparts to call.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .. import parallel
from ..components.checkpointer import TRAIN_STATE, CheckpointManager
from ..datasets.random_data import (
    Batch,
    DataLoaderExhausted,
    RandomTokenSource,
    batch_iterator,
)
from ..mesh import build_mesh, build_parallel_dims, init_distributed
from ..models.hf_wrapper import HFTransformerModel, build_model_config_for
from ..parallel.collectives import clip_grad_norm_, dist_max, dist_sum, dist_sum_tensor
from ..parallel.spmd_types import spmd_context
from ..utils.logger_utils import get_logger
from .config import HybridMeshConfig

# Rank-aware: the helper installs a handler on rank 0 only, so a torchrun run
# logs one line per step instead of one per rank.
logger = get_logger(__name__)

__all__ = ["Trainer"]


class Trainer:
    def __init__(self, cfg: HybridMeshConfig):
        self.cfg = cfg
        self.rank, self.local_rank, self.world_size = init_distributed()

        # Deterministic seeding BEFORE model build so all ranks build identical
        # initial weights -- the precondition for bit-exact DP comparisons.
        self._seed_everything(cfg.seed, deterministic=cfg.deterministic)

        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # 1. mesh (the process topology every dimension is built on). ``parallel_dims``
        #    is the same resolved degrees the mesh was built from, kept so the
        #    trainer can ask "how many DP ranks?" without re-indexing the mesh.
        self.parallel_dims = build_parallel_dims(cfg, self.world_size)
        self.mesh = build_mesh(self.parallel_dims)

        # 2. the model -- HF's own initialization, wrapped for this loop
        model = HFTransformerModel(build_model_config_for(cfg)).to(self.device)

        # 3. parallelism, in Titan's order: tp/pp/cp/ep declared first, fsdp last
        #    (outer wraps inner). Each is a no-op when its degree is 1.
        self.model = parallel.parallelize_hf_transformers(
            model,
            cfg=cfg,
            mesh=self.mesh,
            parallel_dims=self.parallel_dims,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # 4. checkpointing, last because it needs the model and optimizer it is
        #    going to serialize, and because a checkpoint is meaningless until
        #    there is something shaped like a training state to save.
        #
        #    ``self`` rides along as TRAIN_STATE: the manager saves ``states``
        #    wholesale, and the step/token counters are not reachable from either
        #    the model or the optimizer, so a resumed run would otherwise restart
        #    its schedule from zero with weights that are already trained.
        self.checkpointer = CheckpointManager(
            cfg.checkpoint,
            model_parts=[self.model],
            optimizer=self.optimizer,
            states={TRAIN_STATE: self},
            folder=cfg.dump_folder,
        )

        # Counters the checkpoint carries. Kept as plain ints so a resumed run
        # can log "step 61 (resumed at 60)" without re-deriving them.
        self.step = 0
        self.ntokens_seen = 0

    # -- setup helpers ---------------------------------------------------------

    @staticmethod
    def _seed_everything(seed: int, *, deterministic: bool) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=False)

    def _data_iterator(self) -> Iterator[Batch]:
        """The micro-batch source.

        A method rather than an attribute so a future real corpus is swapped in
        by overriding one thing, and so tests can drive the loop with a fixed
        batch without touching the loop itself.
        """
        return batch_iterator(
            RandomTokenSource(
                seed=self.cfg.seed,
                vocab_size=self.cfg.vocab_size,
                batch_size=self.cfg.global_batch_size,
                seq_len=self.cfg.max_seq_len,
            )
        )

    def _dp_slice(self, batch: Batch) -> Batch:
        """Give each DP rank its shard of the global batch (data parallel semantics)."""
        if self.parallel_dims is None:
            dp, dp_rank = 1, 0
        else:
            # The dense DP group spans replicate * shard; unsplit on torchrun
            # it is a plain 1-D mesh, so ``mesh["dp"]`` sizes the batch.
            dp_mesh = self.parallel_dims.get_optional_mesh(
                "dp", include_singleton_axes=True
            )
            dp = dp_mesh.size()
            dp_rank = dp_mesh.get_local_rank()
        per = self.cfg.global_batch_size // dp
        sl = slice(dp_rank * per, (dp_rank + 1) * per)
        return Batch(
            input_ids=batch.input_ids[sl].to(self.device),
            labels=batch.labels[sl].to(self.device),
        )

    # -- the step, one function per level --------------------------------------

    @staticmethod
    def _flatten(batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten ``(B, T)`` into the ``(B*T,)`` shape the wrapper takes.

        The wrapper is a single-sequence entry point -- it adds and removes its
        own batch dim around the decoder call. This micro-batch is one document
        per row of length ``max_seq_len``, so the concatenation is exactly the
        single causal document the fallback attention path expects; RoPE is
        driven per row because positions restart at each row boundary.
        """
        return batch.input_ids.reshape(-1), batch.labels.reshape(-1)

    @staticmethod
    def _loss_sum(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Summed next-token cross-entropy, plus the number of predictions made.

        Not normalized here: the denominator is a *global* token count, and it
        is not knowable until the per-rank counts have been reduced. Returning
        the pair keeps that reduction in the caller, where it belongs.

        TODO: ``logits[:-1]`` pairs the last token of each row with the first
        token of the next, so the loss crosses document boundaries even though
        attention and RoPE do not. Harmless on the synthetic data (rows are
        independent random tokens) and wrong on real packed sequences. Fixing it
        changes the loss denominator and so is a computation change, not a
        refactor -- it needs its own numerical check.
        """
        logits, targets = logits[:-1].float(), labels[1:]
        return F.cross_entropy(logits, targets, reduction="sum"), targets.numel()

    def forward_backward_step(self, batch: Batch) -> tuple[torch.Tensor, int]:
        """Run one micro-batch forward and backward.

        Two bodies, matching torchtitan's split: with pipeline parallelism the
        step drives a *schedule* over several micro-batches rather than calling
        the model once, so the two share nothing but the return shape.

        Returns ``(summed_loss, num_valid_tokens)``: the loss reduced over every
        predicted token rather than averaged, and the denominator that pairs
        with it. Returning them together is what lets the caller normalize by a
        *global* count once the per-rank counts have been reduced.
        """
        if self.parallel_dims is not None and self.parallel_dims.pp_enabled:
            return self._pp_forward_backward_body(batch)
        return self._forward_backward_body(batch)

    def _forward_backward_body(self, batch: Batch) -> tuple[torch.Tensor, int]:
        input_ids, labels = self._flatten(batch)
        # ``spmd_context`` is what makes a process group answerable *by name*
        # (``spmd_mesh_group("tp")`` and friends) for the duration of the body.
        # It is entered here, around the forward/backward only, because that is
        # the region whose components read the ambient mesh -- the optimizer and
        # the checkpointers take their groups as arguments. On a single process
        # it is a no-op, so the same code runs from one device to a full mesh.
        with self._param_context(), spmd_context(self.parallel_dims):
            logits = self.model(input_ids)
            loss_sum, num_valid_tokens = self._loss_sum(logits, labels)
            del logits
            loss_sum.backward()
        return loss_sum.detach(), num_valid_tokens

    def _pp_forward_backward_body(self, batch: Batch) -> tuple[torch.Tensor, int]:
        """The pipeline-parallel body: drive the schedule instead of the model.

        Not implemented, and it fails loudly rather than falling through to the
        single-rank body -- which would silently train every stage on the whole
        model, and look like a working run.

        NOTE: unreachable today. ``parallelize_hf_transformers`` rejects ``pp > 1``
        during ``__init__``, so no Trainer with ``pp > 1`` is ever constructed.
        It stays because it is the second half of the contract, and the two
        halves get wired separately: stage 4 can make ``apply_pp`` return real
        stages before the loop knows how to drive them. At that moment this is
        what catches the gap.

        What stage 4 has to fill in, in order:
          1. ``pipeline_parallel/pipeline.py`` splits the layers into this
             rank's stages; a new ``pp.py`` builds the schedule over them.
          2. Each stage needs its own ``DeviceMesh`` axis, and the model must be
             cut into ``model_parts`` rather than kept whole.
          3. Only the first stage receives ``input_ids`` and only the last
             produces labels -- the middle stages take activations. That is the
             "send ``input_ids``/``labels`` only to the stages that want them"
             note in ``parallelize_hf.py``.
          4. The returned loss is the sum over the last stage's micro-batches,
             paired with a token count, so the caller's normalization is
             unchanged from the non-PP path.
        """
        raise NotImplementedError(
            "Pipeline parallelism is not wired: pipeline_parallel/pipeline.py "
            "splits the model into stages but no schedule drives them. "
            "See docs/hybridmesh_design.md, stage 4."
        )

    def _param_context(self):
        """The context a forward/backward runs inside.

        Currently a placeholder: it is where activation checkpointing and the
        no-typecheck region go, both of which torchtitan wraps around the body.
        Returning ``nullcontext`` rather than inlining nothing keeps the seam
        visible, so it is added by naming it -- not by threading a parameter
        through a function that has since grown around its absence.
        """
        return nullcontext()

    def train_step(self, data_iterator: Iterator[Batch]) -> dict[str, float] | None:
        """One optimizer step. Returns the metrics to log, or ``None`` if not logging.

        The ordering mirrors torchtitan's: take the data, compute the global token
        count, run fwd/bwd, clip, check finiteness, step the optimizer, then (only
        if logging) reduce the loss across DP.
        """
        self.optimizer.zero_grad(set_to_none=True)

        # The reduced meshes are resolved once here rather than inline at each
        # collective: under PP the loss and token count must go to the loss mesh
        # (which spans PP, where the total only exists on the last stage), while
        # everything else stays on the dense DP mesh.
        dp_mesh = (
            None
            if self.parallel_dims is None
            else self.parallel_dims.get_optional_mesh("dp")
        )
        pp_mesh = (
            None
            if self.parallel_dims is None
            else self.parallel_dims.get_optional_mesh("pp")
        )
        loss_mesh = dp_mesh if pp_mesh is None else self.parallel_dims.get_mesh("loss")

        batch = self._dp_slice(next(data_iterator))
        loss_sum, local_valid_tokens = self.forward_backward_step(batch)
        self.ntokens_seen += local_valid_tokens

        # Keep the count on device so normalizing the loss adds no device sync
        # to the training path.
        local_valid_tokens_tensor = torch.tensor(
            local_valid_tokens, dtype=torch.int64, device=self.device
        )
        global_valid_tokens = dist_sum_tensor(local_valid_tokens_tensor, dp_mesh)

        grad_norm = clip_grad_norm_(
            [p for p in self.model.parameters()],
            max_norm=self.cfg.max_norm,
            foreach=True,
            pp_mesh=pp_mesh,
        )

        self._check_finite(loss_sum, grad_norm)

        self.optimizer.step()

        # Summed over tokens, divided by the global count: the loss is then
        # independent of how the batch was split across DP ranks. Division by a
        # tensor keeps the whole computation on device.
        loss = loss_sum / global_valid_tokens

        if not self.should_log():
            return None

        if loss_mesh is not None:
            local_avg = loss_sum / local_valid_tokens_tensor
            global_avg_loss = dist_sum(loss, loss_mesh)
            global_max_loss = dist_max(local_avg, loss_mesh)
        else:
            # Single rank: the two are the same number by construction.
            global_avg_loss = global_max_loss = float(loss)
        return {
            "loss": global_avg_loss,
            "max_loss": global_max_loss,
            "grad_norm": float(grad_norm),
        }

    def _check_finite(self, loss_sum: torch.Tensor, grad_norm: torch.Tensor) -> None:
        """Stop before the optimizer update if anything went non-finite.

        The check is entered by *every* rank on *every* step -- it is a
        collective in torchtitan's version, and a rank that skipped it would
        hang the others. Only the assertion's outcome is rank-dependent.

        ``torch._assert_async`` is private, but it is the right tool: it queues
        the check on the device instead of synchronizing the host, so it costs
        nothing per step and does not break CUDA-graph capture. A failed CUDA
        assertion invalidates the process, which is why the reference
        implementation accepts it.
        """
        is_finite = torch.isfinite(loss_sum).all() & torch.isfinite(grad_norm).all()
        torch._assert_async(
            is_finite,
            f"Loss or gradient norm is not finite at step {self.step}. "
            "Stopping before the optimizer update, since every later number "
            "would be garbage.",
        )

    # -- the loop ---------------------------------------------------------------

    def should_log(self) -> bool:
        return self.step % self.cfg.log_freq == 0

    def should_continue_training(self) -> bool:
        return self.step < self.cfg.steps

    # -- checkpoint state -------------------------------------------------------
    # The manager serializes ``states[TRAIN_STATE]`` (this object) alongside the
    # model and optimizer. These two counters are the whole of that state, and
    # they are here rather than in the checkpoint dict because the running
    # trainer is what has to be mutated back into a resumed step.

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]

    # -- the loop ---------------------------------------------------------------

    def train(self) -> None:
        try:
            if self.checkpointer.load(self.cfg.checkpoint.load_step):
                logger.info(f"Resuming from step {self.step}")

            data_iterator = self._data_iterator()
            while self.should_continue_training():
                self.step += 1

                try:
                    metrics = self.train_step(data_iterator)
                except DataLoaderExhausted:
                    logger.warning("Ran out of data; the last step was canceled.")
                    break

                if metrics is not None:
                    logger.info(
                        f"step {self.step:4d} | loss {metrics['loss']:.6f} "
                        f"| max {metrics['max_loss']:.6f} "
                        f"| grad_norm {metrics['grad_norm']:.4f} "
                        f"| tokens {self.ntokens_seen}"
                    )

                # The manager owns the interval policy: ``save`` decides for
                # itself whether this step is a checkpointing step. The final
                # step is forced so a run that ends off-interval still leaves a
                # resumable artifact rather than only a mid-run one.
                last_step = self.step == self.cfg.steps
                if self.checkpointer.save(self.step, last_step=last_step):
                    logger.info(f"Saved checkpoint for step {self.step}")
        finally:
            # Drain any async save still in flight and stop the purge thread.
            # In a ``finally`` so a run that dies mid-loop still finishes the
            # checkpoint it had already started writing.
            self.checkpointer.close()

        if dist.is_initialized():
            dist.destroy_process_group()
