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
* **Metrics**, reported through ``components/metrics`` rather than a bare
  ``logger.info``: the same loss and grad_norm, plus throughput, MFU and device
  memory, to stdout and optionally TensorBoard or WandB. The processor also owns
  the reporting frequency and the token/data-loading accounting, so the loop
  only has to call ``add_tokens`` and ``log``.
* **Profiling**, through ``components/profiler``: ``Profiler`` is entered once
  around the loop and stepped once per iteration, so Kineto traces land on a
  schedule and allocator memory snapshots are written periodically -- plus one
  more if the run dies of an OOM, which is the one that is usually wanted.

What was NOT ported: the component system (``Configurable``, ``model_spec``,
``sdc_replayer``, validator, CUDA graphs). Those are infrastructure the loop
calls into, not loop logic, and hpmesh has no counterparts to call.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .. import parallel
from ..components.checkpointer import DATALOADER, TRAIN_STATE, CheckpointManager
from ..components.loss import IGNORE_INDEX, next_token_targets
from ..components.metrics import MetricsProcessor
from ..components.profiler import Profiler
from ..datasets.loader import BaseDataLoader, DataloaderExhaustedError, TrainerBatch
from ..datasets.random_data import Batch, DataLoaderExhausted, RandomTokenDataLoader
from ..mesh import build_mesh, build_parallel_dims, init_distributed
from ..models.common.aux_loss import (
    AuxLoss,
    collect_aux_loss_metrics,
    register_aux_loss_zero_hook,
)
from ..models.hf_wrapper import (
    HFTransformerModel,
    build_model_config_for,
    num_flops_per_token,
)
from ..parallel.collectives import clip_grad_norm_, dist_max, dist_sum, dist_sum_tensor
from ..parallel.context_parallel import shard_batch_for_cp
from ..parallel.spmd_types import spmd_context
from ..utils.logger_utils import get_logger
from .config import HybridMeshConfig

# Rank-aware: the helper installs a handler on rank 0 only, so a torchrun run
# logs one line per step instead of one per rank.
logger = get_logger(__name__)

__all__ = ["Trainer"]


class Trainer:
    # Class-level default so a Trainer built with ``__new__`` -- which is how
    # the tests exercise the pure helpers without a process group -- sees the
    # same "source not built yet" state as an attribute would give, rather than
    # an AttributeError. ``None`` means "fall back to the synthetic source".
    dataloader: BaseDataLoader | None = None

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

        # Aux losses (the MoE load-balance loss a swapped-in MoE carries)
        # accumulate per forward; this pre-hook rolls the per-instance sums
        # into the step registers at each optimizer step. Harmless when no
        # aux loss exists.
        register_aux_loss_zero_hook(self.optimizer, [self.model], self.parallel_dims)

        # 4. the micro-batch source. Built before the checkpointer, which
        #    serializes its read position alongside the model.
        self.dataloader = self._build_dataloader()

        # 5. checkpointing, last because it needs the model and optimizer it is
        #    going to serialize, and because a checkpoint is meaningless until
        #    there is something shaped like a training state to save.
        #
        #    ``self`` rides along as TRAIN_STATE: the manager saves ``states``
        #    wholesale, and the step/token counters are not reachable from either
        #    the model or the optimizer, so a resumed run would otherwise restart
        #    its schedule from zero with weights that are already trained.
        #
        #    A loadable dataloader rides along too: resuming without its read
        #    position would resume the weights and restart the data, silently
        #    training a second pass over the beginning of the corpus.
        states: dict[str, Any] = {TRAIN_STATE: self}
        if self.dataloader is not None:
            states[DATALOADER] = self.dataloader
        self.checkpointer = CheckpointManager(
            cfg.checkpoint,
            model_parts=[self.model],
            optimizer=self.optimizer,
            states=states,
            folder=cfg.dump_folder,
        )

        # Counters the checkpoint carries. Kept as plain ints so a resumed run
        # can log "step 61 (resumed at 60)" without re-deriving them.
        self.step = 0
        self.ntokens_seen = 0

        # 6. metrics, last because it needs the mesh (for the throughput
        #    divisor and the metrics rank) and the model config (for FLOPs per
        #    token). It replaces the plain per-step ``logger.info`` the loop used
        #    to emit: the same loss and grad_norm, plus throughput, MFU and
        #    memory, and the frequency is now one knob instead of two.
        #
        #    ``num_flops_per_token`` is measured from the parameters, not from
        #    the config's sizes, so the number describes the model that actually
        #    exists -- including one whose sizes came from the Hub.
        self.metrics = MetricsProcessor(
            cfg.metrics,
            parallel_dims=self.parallel_dims,
            dump_folder=cfg.dump_folder,
            pp_schedule=cfg.pipeline_parallel_schedule,
            num_flops_per_token=num_flops_per_token(cfg),
            tag=cfg.metrics.tag,
        )

    # -- setup helpers ---------------------------------------------------------

    @staticmethod
    def _seed_everything(seed: int, *, deterministic: bool) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=False)

    def _dp_rank_world_size(self) -> tuple[int, int]:
        """This rank's position and extent along the dense DP axis."""
        # ``getattr``, not attribute access: a Trainer built with ``__new__``
        # (the tests' way of exercising the pure helpers) has no mesh, and the
        # only correct answer there is "one rank, no sharding".
        if getattr(self, "parallel_dims", None) is None:
            return 0, 1
        # The dense DP group spans replicate * shard; unsplit on torchrun it is
        # a plain 1-D mesh.
        dp_mesh = self.parallel_dims.get_optional_mesh("dp", include_singleton_axes=True)
        return dp_mesh.get_local_rank(), dp_mesh.size()

    def _build_dataloader(self) -> BaseDataLoader | None:
        """Build the micro-batch source the config names.

        Returns ``None`` when the source cannot be checkpointed, which today
        means the synthetic one: its position is derivable from the step
        counter, so carrying it separately would only add a second copy of
        state that could disagree with the first.
        """
        dp_rank, dp_world_size = self._dp_rank_world_size()
        loader = self.cfg.dataloader.build(
            seed=self.cfg.seed,
            vocab_size=self.cfg.vocab_size,
            batch_size=self.cfg.global_batch_size,
            seq_len=self.cfg.max_seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            max_context_length=self.cfg.max_seq_len,
            # Per rank, not global: the Grain loader splits every dataset's
            # rows across ``dp_world_size`` ranks itself, so this many tokens
            # per rank is this many tokens per rank of the global batch.
            num_tokens_per_batch=self.cfg.global_batch_size
            // dp_world_size
            * self.cfg.max_seq_len,
        )
        if isinstance(loader, RandomTokenDataLoader):
            # The random positions ARE the step counter, already in the
            # checkpoint -- a separate cursor would be a second copy of it.
            return None
        return loader

    def _data_iterator(self) -> Iterator[Batch | TrainerBatch]:
        """The micro-batch source.

        A method rather than an attribute so tests can drive the loop with a
        fixed batch without touching the loop itself. The loader is a
        :class:`~hpmesh.datasets.loader.BaseDataLoader` whenever there is one,
        so both the synthetic and the Grain path arrive here the same way.
        """
        if self.dataloader is not None:
            return iter(self.dataloader)
        dp_rank, dp_world_size = self._dp_rank_world_size()
        return iter(
            RandomTokenDataLoader(
                seed=self.cfg.seed,
                vocab_size=self.cfg.vocab_size,
                batch_size=self.cfg.global_batch_size,
                seq_len=self.cfg.max_seq_len,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
            )
        )

    def _as_batch(
        self, batch: Batch | TrainerBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int | None]:
        """Normalize either loader's batch into the tensors the step consumes.

        The two loaders disagree about what a batch is -- the synthetic one
        yields ``(B, T)`` rows of one document each, the Grain one a flat
        packed token stream -- so they are reconciled here, once, rather than
        at every call site. Returns ``(input_ids, labels, positions,
        num_valid_tokens)``, where the last two are ``None`` when the loader
        did not supply them and the caller should fall back to its defaults.

        Both paths leave here holding a flat token stream whose target at
        position ``t`` is the token at ``t + 1`` *within the same document*,
        which is the contract ``_loss_sum``'s plain shift assumes. The synthetic
        path needs real work to reach it (its labels are unshifted, and its
        rows are separate documents that must not be predicted across); the
        Grain path arrives already shifted and already ``IGNORE_INDEX``-masked
        at every document boundary, so for it this is a read.
        """
        if isinstance(batch, Batch):
            # Rows are independent documents of length T, so the shift is
            # within a row: the row-final position would predict the next
            # document's first token, which the model had no context for, and
            # comes back IGNORE_INDEX from ``next_token_targets``.
            seq_len = batch.labels.shape[-1]
            targets = next_token_targets(batch.labels.reshape(-1), seq_len=seq_len)
            return batch.input_ids, targets, None, None

        # ``num_valid_tokens`` is popped, not read: everything left in the dict
        # becomes a model kwarg, and this one is a plain int the forward has no
        # use for. The collator counted it while it already had the labels in
        # hand, so it is exact -- the trainer does not rescore the batch.
        num_valid_tokens = batch.pop("num_valid_tokens", None)
        # ``padding_mask`` marks the collator's filler. It is implied by the
        # IGNORE_INDEX labels and consumed by nothing on this path, so it is
        # dropped rather than forwarded to a forward that would reject it.
        batch.pop("padding_mask", None)
        positions = batch.pop("positions", None)
        return batch["input"], batch["labels"], positions, num_valid_tokens

    # -- the step, one function per level --------------------------------------

    @staticmethod
    def _flatten(
        input_ids: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten ``(B, T)`` into the ``(B*T,)`` shape the wrapper takes.

        The wrapper is a single-sequence entry point -- it adds and removes its
        own batch dim around the decoder call. The synthetic source yields one
        document per row of length ``max_seq_len``, so the concatenation is
        exactly the single causal document the fallback attention path expects;
        RoPE is driven per row because positions restart at each row boundary.

        The Grain source already hands over a flat token stream, so for it this
        is the identity -- which is what makes the two sources one code path
        from here on.
        """
        return input_ids.reshape(-1), labels.reshape(-1)

    @staticmethod
    def _loss_sum(
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        num_valid_tokens: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Summed next-token cross-entropy, plus the number of predictions made.

        ``labels`` arrives already aligned with ``logits`` -- ``logits[t]``
        predicts ``labels[t]``, both sources having done their shift upstream
        (see ``_as_batch``). No shift happens here, which is what lets the two
        sources share one loss: the synthetic path slots its rows together and
        the Grain path arrives already packed, and both mark the positions that
        must not be predicted with ``IGNORE_INDEX`` rather than dropping them.
        Those positions are the row ends of the synthetic path and the document
        boundaries and packing padding of the Grain one.

        Not normalized: the denominator is a *global* token count, and it is
        not knowable until the per-rank counts have been reduced. Returning the
        pair keeps that reduction in the caller, where it belongs.

        ``num_valid_tokens`` defaults to the count of predictable labels, which
        is the same thing the collator computes. It only needs passing when the
        caller has it already (the Grain path does, from the collator) so the
        trainer does not rescan on the critical path.
        """
        if num_valid_tokens is None:
            num_valid_tokens = int((labels != IGNORE_INDEX).sum())
        loss = F.cross_entropy(
            logits.float(), labels, reduction="sum", ignore_index=IGNORE_INDEX
        )
        return loss, num_valid_tokens

    def forward_backward_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        num_valid_tokens: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Run one micro-batch forward and backward.

        Two bodies, matching torchtitan's split: with pipeline parallelism the
        step drives a *schedule* over several micro-batches rather than calling
        the model once, so the two share nothing but the return shape.

        ``num_valid_tokens`` is the count of labels that actually contribute to
        the loss, when the source knows it (the collator counts it there rather
        than the trainer rescanning on the critical path). ``None`` means
        "every label counts", which is true of the synthetic source.

        Returns ``(summed_loss, num_valid_tokens)``: the loss reduced over every
        predicted token rather than averaged, and the denominator that pairs
        with it. Returning them together is what lets the caller normalize by a
        *global* count once the per-rank counts have been reduced.
        """
        if self.parallel_dims is not None and self.parallel_dims.pp_enabled:
            return self._pp_forward_backward_body(input_ids, labels)
        return self._forward_backward_body(
            input_ids, labels, positions=positions, num_valid_tokens=num_valid_tokens
        )

    def _forward_backward_body(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        num_valid_tokens: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        input_ids, labels = self._flatten(input_ids, labels)
        # To the device first: both the CP shard below and the forward expect
        # it, and a shard of a CPU tensor placed on a device mesh would mix
        # placements.
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        if positions is not None:
            positions = positions.reshape(-1).to(self.device)
        cp_mesh = (
            None
            if self.parallel_dims is None
            else self.parallel_dims.get_optional_mesh("cp")
        )
        if cp_mesh is not None:
            # CP shards the sequence: positions are generated explicitly (the
            # wrapper's arange default would restart at 0 on every rank) and
            # sharded alongside the tokens, so RoPE follows each token to its
            # rank. The loss sums over tokens, so the headtail rearrangement
            # needs no undoing here.
            if positions is None:
                positions = torch.arange(input_ids.numel(), device=self.device)
            input_ids, labels, positions = shard_batch_for_cp(
                input_ids,
                labels,
                positions,
                cp_mesh,
                load_balancer=self.cfg.parallel.context_parallel_load_balancer,
            )
        # Aux losses normalize by the step's global valid-token count -- the
        # same denominator the main loss is normalized by in ``train_step``.
        # The forward consumes it (``AuxLoss.inject``), so it must be reduced
        # here, before the forward, and it spans the same token mesh the
        # main-loss count uses. The count itself is the collator's when the
        # source supplies one (masked prompt tokens and packing padding do not
        # count), and otherwise every label but the first.
        if AuxLoss._group_counts:
            if self.parallel_dims is None:
                token_mesh = None
            else:
                token_mesh = (
                    self.parallel_dims.get_optional_mesh("dp")
                    if cp_mesh is None
                    else self.parallel_dims.get_mesh("loss")
                )
            local_count = (
                labels.numel() - 1
                if num_valid_tokens is None
                else num_valid_tokens
            )
            AuxLoss.set_step_denominator(
                dist_sum_tensor(
                    torch.tensor(
                        local_count, dtype=torch.float32, device=self.device
                    ),
                    token_mesh,
                )
            )
        # ``spmd_context`` is what makes a process group answerable *by name*
        # (``spmd_mesh_group("tp")`` and friends) for the duration of the body.
        # It is entered here, around the forward/backward only, because that is
        # the region whose components read the ambient mesh -- the optimizer and
        # the checkpointers take their groups as arguments. On a single process
        # it is a no-op, so the same code runs from one device to a full mesh.
        with self._param_context(), spmd_context(self.parallel_dims):
            logits = self.model(input_ids, positions=positions)
            loss_sum, num_valid_tokens = self._loss_sum(
                logits, labels, num_valid_tokens=num_valid_tokens
            )
            del logits
            loss_sum.backward()
        return loss_sum.detach(), num_valid_tokens

    def _pp_forward_backward_body(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
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

    def train_step(
        self, data_iterator: Iterator[Batch | TrainerBatch]
    ) -> dict[str, float] | None:
        """One optimizer step. Returns the metrics to log, or ``None`` if not logging.

        The ordering mirrors torchtitan's: take the data, compute the global token
        count, run fwd/bwd, clip, check finiteness, step the optimizer, then (only
        if logging) reduce the loss across DP.
        """
        self.optimizer.zero_grad(set_to_none=True)

        # The reduced meshes are resolved once here rather than inline at each
        # collective: under PP the loss and token count must go to the loss mesh
        # (which spans PP, where the total only exists on the last stage), and
        # under CP they must span the CP axis too -- every CP rank holds a
        # sequence shard, so a dp-only reduction would undercount both the
        # token total and the loss sum by a factor of cp. The loss mesh is
        # exactly dp * cp, so it is the right group for both cases; everything
        # else stays on the dense DP mesh.
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
        cp_mesh = (
            None
            if self.parallel_dims is None
            else self.parallel_dims.get_optional_mesh("cp")
        )
        loss_mesh = (
            dp_mesh
            if pp_mesh is None and cp_mesh is None
            else self.parallel_dims.get_mesh("loss")
        )

        data_load_start = perf_counter()
        input_ids, labels, positions, num_valid_tokens = self._as_batch(
            next(data_iterator)
        )
        self.metrics.add_data_loading_time(perf_counter() - data_load_start)
        self.metrics.add_tokens(labels.numel())

        loss_sum, local_valid_tokens = self.forward_backward_step(
            input_ids,
            labels,
            positions=positions,
            num_valid_tokens=num_valid_tokens,
        )
        self.ntokens_seen += local_valid_tokens

        # Keep the count on device so normalizing the loss adds no device sync
        # to the training path.
        local_valid_tokens_tensor = torch.tensor(
            local_valid_tokens, dtype=torch.int64, device=self.device
        )
        token_mesh = dp_mesh if cp_mesh is None else loss_mesh
        global_valid_tokens = dist_sum_tensor(local_valid_tokens_tensor, token_mesh)

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

        # Only a logging step derives the two reported losses: they are the only
        # place this function touches the host, and the metrics dict is typed
        # for floats. The loss the optimizer uses is the tensor above.
        if not self.should_log():
            return None

        if loss_mesh is not None:
            local_avg = loss_sum / local_valid_tokens_tensor
            global_avg_loss = float(dist_sum(loss, loss_mesh))
            global_max_loss = float(dist_max(local_avg, loss_mesh))
        else:
            # Single rank: the two are the same number by construction.
            global_avg_loss = global_max_loss = float(loss)
        metrics = {
            "loss": global_avg_loss,
            "max_loss": global_max_loss,
            "grad_norm": float(grad_norm),
        }
        # Aux-loss step registers are rolled up by the optimizer step pre-hook
        # above; this reduces them for logging. Single-process runs skip the
        # collection (there is no mesh to reduce over and no second rank's
        # contribution); the injection itself is unaffected.
        if AuxLoss._group_counts and self.parallel_dims is not None:
            metrics.update(collect_aux_loss_metrics(self.parallel_dims))
        return metrics

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
        # Delegated rather than reimplemented: the metrics processor also
        # guarantees the first step logs, and two copies of that rule would
        # drift the moment one of them changed.
        return self.metrics.should_log(self.step)

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
            # Entered around the loop rather than around a single step: the
            # torch profiler's schedule counts iterations across the whole run
            # and only dumps a trace at the end of a cycle, so a per-step
            # context would never reach one. Left open when profiling is off --
            # the Profiler holds no handles in that case.
            with Profiler(
                self.cfg.profiler,
                global_step=self.step,
                base_folder=self.cfg.dump_folder,
            ) as profiler:
                while self.should_continue_training():
                    self.step += 1

                    try:
                        step_metrics = self.train_step(data_iterator)
                    except (DataLoaderExhausted, DataloaderExhaustedError):
                        # Two spellings of one event: the synthetic source and
                        # the Grain loader each raise their own. Treated
                        # identically -- abandon the step rather than train on
                        # a partial batch.
                        logger.warning("Ran out of data; the last step was canceled.")
                        break

                    if step_metrics is not None:
                        self.metrics.log(
                            self.step,
                            global_avg_loss=step_metrics["loss"],
                            global_max_loss=step_metrics["max_loss"],
                            grad_norm=step_metrics["grad_norm"],
                            extra_metrics={
                                k: v
                                for k, v in step_metrics.items()
                                if k not in ("loss", "max_loss", "grad_norm")
                            },
                        )

                    # The manager owns the interval policy: ``save`` decides for
                    # itself whether this step is a checkpointing step. The final
                    # step is forced so a run that ends off-interval still leaves
                    # a resumable artifact rather than only a mid-run one.
                    last_step = self.step == self.cfg.steps
                    if self.checkpointer.save(self.step, last_step=last_step):
                        logger.info(f"Saved checkpoint for step {self.step}")

                    # Advances the schedule. After the save, so the profiler's
                    # active iteration covers an ordinary step rather than one
                    # that also wrote a checkpoint.
                    profiler.step()
        finally:
            # Drain any async save still in flight and stop the purge thread.
            # In a ``finally`` so a run that dies mid-loop still finishes the
            # checkpoint it had already started writing.
            self.checkpointer.close()
            self.metrics.close()
            # Releases the Grain prefetch thread. Without this the loader is
            # only collected at interpreter shutdown, where its ``__del__``
            # raises against an already-torn-down state.
            if self.dataloader is not None:
                self.dataloader.close()

        if dist.is_initialized():
            dist.destroy_process_group()
