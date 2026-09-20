"""Trainer -- the single training loop, shared by every learning step.

Shape vendored from torchtitan ``trainer.py``: ``train`` -> ``train_step`` ->
``forward_backward_step`` -> ``_forward_backward_body``, one function per level
of the step, so each can be read and tested on its own. The distributed
complexity still lives in ``parallel/``; the loop is meant to read end to end.

What the migration added, and why each earned its place:

* **Token-normalized loss.** The loss is a SUM over predicted tokens divided by
  the token count reduced across DP. That makes the reported number independent
  of how the batch was split across ranks, and it is the precondition for
  gradient accumulation to sum correctly.
* **The denominator is a whole-batch property.** It counts the tokens of the
  *unsharded* batch -- before context parallelism slices it and before pipeline
  parallelism cuts it into micro-batches -- so it is the same number on every
  rank of the workload and for every micro-batch of a step. Sharding the
  sequence must not change the reported loss.
* **Gradient accumulation.** ``gradient_accumulation_steps`` runs the
  forward/backward once per group and advances the optimizer once. The division
  happens *inside* each group's backward -- the group's summed loss over every
  group's token counts -- but the effective step length is the token count of
  the *whole* accumulation window. At ``gradient_accumulation_steps == 1`` the
  two coincide and every reading is the familiar per-batch one. Above 1 the
  gradients are consistent with a batch G times longer, so the reported loss
  converges to the true per-token loss while the window is still filling up.
  See ``train_step`` for the ordering.
* **Gradient clipping + ``grad_norm`` reporting.** ``clip_grad_norm_`` reduces
  the norm across PP stages before clipping, which ``torch.nn.utils`` cannot do
  because each stage holds disjoint parameters. The reported norm is therefore
  the norm of the *normalized* gradient, matching the reference: a config
  reporting 281 un-normalized reports ~0.56 once the loss is divided by the
  token count before backward.
* **A finiteness check**, reduced to one global flag across the loss and PP
  meshes. A NaN loss or gradient looked exactly like a healthy step: training
  continued and every later number was garbage. This stops at the first bad step
  instead, and does it with an on-device check so it neither synchronizes (unlike
  ``.item()``) nor becomes a CUDA-graph break.
* **Garbage collection on the training loop's schedule.** The cyclic collector
  is disabled process-wide and run at a step boundary instead, so it cannot fire
  mid-forward -- see ``utils/gc``.
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

``train_step``'s execution order follows torchtitan's and is load-bearing:
zero the gradients, snapshot the learning rate, read *every* batch the step
consumes, reduce the token count, run the forward/backward groups, clip, check
finiteness, wait for any in-flight checkpoint staging, step the optimizer and
then the scheduler, and only then normalize the loss for reporting. Steps whose
value must be identical across all the batches of a step -- the denominator and
the reported lr -- are taken before any of them is consumed.

What was NOT ported: torchtitan's component system (``model_spec``,
``sdc_replayer``, validator, CUDA graphs). Those are infrastructure the loop
calls into, not loop logic, and hpmesh has no counterparts to call.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from itertools import chain
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from .. import parallel
from ..components.checkpointer import DATALOADER, TRAIN_STATE, CheckpointManager
from ..components.loss import IGNORE_INDEX, next_token_targets
from ..components.lr_scheduler import LRScheduler, build_lr_scheduler
from ..components.metrics import MetricsProcessor
from ..components.profiler import Profiler
from ..datasets import build_dataloader
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
from ..parallel.parallel_dims import ParallelDims
from ..parallel.pipeline_parallel import PipelineParallelSetup
from ..parallel.spmd_types import spmd_context
from ..utils.gc import GarbageCollection
from ..utils.logger_utils import get_logger
from .config import HybridMeshConfig

# Rank-aware: the helper installs a handler on rank 0 only, so a torchrun run
# logs one line per step instead of one per rank.
logger = get_logger(__name__)

__all__ = ["Trainer"]


class Trainer:
    # The state a built trainer carries, declared here so the shape of the
    # object is readable without reading ``__init__`` -- the reference does the
    # same. Anything built later is annotated with ``| None`` so a ``Trainer``
    # made with ``__new__`` (how the tests exercise the pure helpers without a
    # process group) sees the same "not built yet" state an attribute would
    # give, rather than an AttributeError.
    cfg: HybridMeshConfig
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    parallel_dims: ParallelDims | None
    mesh: DeviceMesh | None
    model: torch.nn.Module | None
    model_parts: list[torch.nn.Module]
    pp_schedule: Any | None
    pp_has_first_stage: bool
    pp_has_last_stage: bool
    _pp_loss_sentinel: torch.Tensor | None
    optimizer: torch.optim.Optimizer
    # Defaulted, not just annotated: ``_data_iterator`` reads it, and that is
    # the one helper the tests drive off a ``Trainer`` built with ``__new__``.
    # ``None`` is also a real state -- it means "fall back to the synthetic
    # source" -- so the default is the correct value, not just a placeholder.
    dataloader: BaseDataLoader | None = None
    lr_scheduler: LRScheduler | None
    checkpointer: CheckpointManager | None
    metrics: MetricsProcessor | None
    gc_handler: GarbageCollection | None

    # Additional training state, saved in the checkpoint.
    step: int
    ntokens_seen: int

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
        if self.parallel_dims is not None and self.parallel_dims.pp_enabled:
            # The dense (dp, cp, tp) mesh does not cover the world under PP,
            # so ``build_mesh``'s coverage backstop would reject it. The same
            # view over this rank's non-PP coordinates exists per stage and is
            # what the per-part apply_* functions index (apply_pp resolves it
            # off parallel_dims itself); keep the attribute consistent.
            self.mesh = self.parallel_dims.spmd_dense_mesh()
        else:
            self.mesh = build_mesh(self.parallel_dims)

        # 2. the model -- HF's own initialization, wrapped for this loop
        model = HFTransformerModel(build_model_config_for(cfg)).to(self.device)

        # 3. parallelism, in Titan's order: tp/pp/cp/ep declared first, fsdp last
        #    (outer wraps inner). Each is a no-op when its degree is 1.
        orchestration = parallel.parallelize_hf_transformers(
            model,
            cfg=cfg,
            mesh=self.mesh,
            parallel_dims=self.parallel_dims,
            device=self.device,
        )
        if isinstance(orchestration, PipelineParallelSetup):
            # pp > 1: no single model survives the split -- this rank holds its
            # stages' chunks only, and the schedule drives them in
            # ``_pp_forward_backward_body``.
            self.model = None
            self.model_parts = orchestration.model_parts
            self.pp_schedule = orchestration.schedule
            self.pp_has_first_stage = orchestration.has_first_stage
            self.pp_has_last_stage = orchestration.has_last_stage
            # The loss exists only on the last stage; every other stage reports
            # this sentinel, which is finite (the finiteness check runs on every
            # rank) and never logged (the metrics rank is a last-stage rank).
            self._pp_loss_sentinel = torch.full((1,), -1.0, device=self.device)
        else:
            self.model = orchestration
            self.model_parts = [orchestration]

        self.optimizer = torch.optim.AdamW(
            chain.from_iterable(part.parameters() for part in self.model_parts),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # The lr schedule. Built regardless of whether the knobs were touched:
        # the default is warmup_steps=0 with no decay, so the factor is a
        # constant 1.0 and step 1 runs at exactly ``cfg.lr``. That costs one
        # multiply per step and removes the branch that would otherwise decide
        # whether the lr is scheduled -- a branch whose two sides would have to
        # be kept numerically identical forever.
        self.lr_scheduler = build_lr_scheduler(
            cfg.lr_scheduler_config,
            optimizer=self.optimizer,
            training_steps=cfg.steps,
        )

        # Aux losses (the MoE load-balance loss a swapped-in MoE carries)
        # accumulate per forward; this pre-hook rolls the per-instance sums
        # into the step registers at each optimizer step. Harmless when no
        # aux loss exists.
        register_aux_loss_zero_hook(
            self.optimizer, self.model_parts, self.parallel_dims
        )

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
        #
        #    The schedule is NOT registered. It holds one integer -- last_epoch --
        #    and every step's value of it is the step number, which the trainer
        #    above already serializes. A second copy could only ever disagree with
        #    the first, and a resumed run re-derives the lr from ``last_epoch``
        #    and the optimizer's own ``base_lrs``, which come back with the
        #    optimizer's state.
        states: dict[str, Any] = {TRAIN_STATE: self}
        if self.dataloader is not None:
            states[DATALOADER] = self.dataloader
        self.checkpointer = CheckpointManager(
            cfg.checkpoint,
            model_parts=self.model_parts,
            optimizer=self.optimizer,
            states=states,
            folder=cfg.dump_folder,
            # Under PP the optimizer's positional state indices collide across
            # stages (every stage's first parameter is index 0), so the
            # checkpoint keys optimizer state by parameter FQN instead.
            optimizer_fqn_keying=self.parallel_dims is not None
            and self.parallel_dims.pp_enabled,
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
        """This rank's position and extent along the dataloading (DP) axis."""
        # ``getattr``, not attribute access: a Trainer built with ``__new__``
        # (the tests' way of exercising the pure helpers) has no mesh, and the
        # only correct answer there is "one rank, no sharding".
        if getattr(self, "parallel_dims", None) is None:
            return 0, 1
        if self.parallel_dims.pp_enabled:
            # Under PP the dataloading view is the "batch" axis: it spans
            # dp_replicate * dp_shard and excludes pp, so every stage of one
            # pipeline reads the same shard of the global batch.
            batch_mesh = self.parallel_dims.get_optional_mesh(
                "batch", include_singleton_axes=True
            )
            return batch_mesh.get_local_rank(), batch_mesh.size()
        # The dense DP group spans replicate * shard; unsplit on torchrun it is
        # a plain 1-D mesh.
        dp_mesh = self.parallel_dims.get_optional_mesh(
            "dp", include_singleton_axes=True
        )
        return dp_mesh.get_local_rank(), dp_mesh.size()

    def _build_dataloader(self) -> BaseDataLoader | None:
        """Build the micro-batch source the config names.

        Returns ``None`` when the source cannot be checkpointed, which today
        means the synthetic one: its position is derivable from the step
        counter, so carrying it separately would only add a second copy of
        state that could disagree with the first.
        """
        dp_rank, dp_world_size = self._dp_rank_world_size()
        loader = build_dataloader(
            self.cfg.dataloader,
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
        """The raw micro-batch source.

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

    def batch_generator(
        self, data_iterable: Iterable[Batch | TrainerBatch]
    ) -> Iterator[Batch | TrainerBatch]:
        """Wrap the source with the per-fetch accounting the step needs.

        Mirrors the reference's ``batch_generator``: the timing and token
        counters are updated as each batch is *taken*, not when it is used, so
        a step that raises partway through still accounts for the reads it
        performed. Yields the batch unchanged -- the loader's shape is the
        step's business, not this wrapper's.

        The token counter takes every label, not just the predictable ones.
        It reports throughput -- tokens the loader produced, which is what
        ``MFU`` wants -- and is not the loss denominator. That one is
        ``local_valid_tokens`` in ``train_step``, which is separately reduced
        across DP.

        Running out of data raises ``DataloaderExhaustedError`` rather than
        letting ``StopIteration`` escape: the loop must abandon the whole step
        rather than train on a partial batch, and ``StopIteration`` inside a
        generator would be read as "this generator is empty" and silently end
        training instead of failing the step.
        """
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = perf_counter()
            try:
                batch = next(data_iterator)
            except (DataLoaderExhausted, StopIteration) as ex:
                # Two spellings of one event -- the synthetic source raises its
                # own type, a real loader just stops -- mapped to the one the
                # loop catches.
                raise DataloaderExhaustedError() from ex
            labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
            self.metrics.add_tokens(labels.numel())
            self.metrics.add_data_loading_time(perf_counter() - data_load_start)
            yield batch

    @staticmethod
    def _as_batch(
        batch: Batch | TrainerBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        """Normalize either loader's batch into the tensors the step consumes.

        The two loaders disagree about what a batch is -- the synthetic one
        yields ``(B, T)`` rows of one document each, the Grain one a flat
        packed token stream -- so they are reconciled here, once, rather than
        at every call site. Returns ``(input_ids, labels, positions,
        num_valid_tokens)``. ``positions`` is the only optional member: the
        synthetic source has no use for it and the wrapper falls back to its
        own ``arange``, whereas a packed stream must supply it to restart the
        position counter at each document boundary.

        The count is never optional. It is the denominator of the loss, which
        is divided out before the first backward, so a ``None`` here would
        leave the step with no way to normalize at all -- the synthetic path
        derives it from the freshly shifted targets, and the Grain path takes
        the collator's.

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
            # Counted here, not inside the forward. The loss divides by the
            # step's global token count before it backwards (see
            # ``_forward_backward_body``), so the number has to exist before
            # the first forward of the step -- and this is the only place the
            # un-sharded synthetic labels are still available to count.
            return (
                batch.input_ids,
                targets,
                None,
                int((targets != IGNORE_INDEX).sum()),
            )

        # Only the tensors the forward and the loss consume are carried out.
        # The rest of the collator's dict -- notably ``padding_mask``, implied
        # by the IGNORE_INDEX labels and consumed by nothing on this path --
        # was already logged by ``batch_generator``, so it is dropped here
        # rather than held across the accumulation window.
        #
        # Tensors stay on the CPU, as the reference's ``batch_generator``
        # documents: the move happens per micro-batch, in
        # ``_forward_backward_body``, so holding the rest of the step's batches
        # costs host memory rather than device memory.
        input_ids = batch["input"]
        labels = batch["labels"]
        # Counted by the collator, which had the labels in hand; ``_loss_sum``
        # does not rescore the batch. Recounted here when absent rather than
        # permissively defaulted, so a dict that silently lacks the key still
        # produces a correct denominator.
        num_valid_tokens = batch.get("num_valid_tokens")
        if num_valid_tokens is None:
            num_valid_tokens = int((labels != IGNORE_INDEX).sum())
        # ``positions`` is the one optional extra kwarg the wrapper understands
        # (RoPE's argument); it is what lets a packed stream restart its
        # position counter at each document boundary.
        return input_ids, labels, batch.get("positions"), num_valid_tokens

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
        num_valid_tokens: int,
    ) -> torch.Tensor:
        """Summed next-token cross-entropy over the predictable labels.

        ``labels`` arrives already aligned with ``logits`` -- ``logits[t]``
        predicts ``labels[t]``, both sources having done their shift upstream
        (see ``_as_batch``). No shift happens here, which is what lets the two
        sources share one loss: the synthetic path slots its rows together and
        the Grain path arrives already packed, and both mark the positions that
        must not be predicted with ``IGNORE_INDEX`` rather than dropping them.
        Those positions are the row ends of the synthetic path and the document
        boundaries and packing padding of the Grain one.

        ``num_valid_tokens`` is required even though it is not used here. It is
        the count of the labels that actually contribute, and passing it in --
        rather than recomputing it from ``labels`` -- is the contract: it comes
        from the *unsharded* batch, while ``labels`` may since have been sliced
        by context parallelism, so recounting here would undercount the
        denominator by a factor of ``cp``. Requiring the argument makes the
        wrong version unrepresentable. The parameter is also what the caller
        threads to the backward, which is where the division actually happens
        (see ``_forward_backward_body``).

        Not normalized: the denominator is a *global* token count, and it is
        not knowable until the per-rank counts have been reduced. The caller
        owns that reduction.
        """
        del num_valid_tokens
        return F.cross_entropy(
            logits.float(), labels, reduction="sum", ignore_index=IGNORE_INDEX
        )

    def forward_backward_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        positions: torch.Tensor | None,
        num_valid_tokens: int,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Run one micro-batch forward and backward. Returns this rank's loss sum.

        Two bodies, matching torchtitan's split: with pipeline parallelism the
        step drives a *schedule* over several micro-batches rather than calling
        the model once, so the two share nothing but the return shape.

        ``num_valid_tokens`` is the count of labels that actually contribute to
        the loss, taken from the unsharded batch (see ``_loss_sum``). It is
        only reported onward; the normalization uses ``global_valid_tokens``.

        ``global_valid_tokens`` is the step's denominator, reduced across the
        DP axis. It is passed in rather than computed here because it must be
        the *same* number for every micro-batch of the step -- under gradient
        accumulation the count only exists once all of them have been read, so
        the caller reduces it first and hands it down.

        Returns the summed loss the forward computed. The backward has already
        divided gradients by the global count, so the returned tensor is for
        reporting only, and the caller sums it over the accumulation groups.
        """
        if self.parallel_dims is not None and self.parallel_dims.pp_enabled:
            return self._pp_forward_backward_body(
                input_ids, labels, global_valid_tokens=global_valid_tokens
            )
        return self._forward_backward_body(
            input_ids,
            labels,
            positions=positions,
            num_valid_tokens=num_valid_tokens,
            global_valid_tokens=global_valid_tokens,
        )

    def _forward_backward_body(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        positions: torch.Tensor | None,
        num_valid_tokens: int,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
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
        # ``spmd_context`` is what makes a process group answerable *by name*
        # (``spmd_mesh_group("tp")`` and friends) for the duration of the body.
        # It is entered here, around the forward/backward only, because that is
        # the region whose components read the ambient mesh -- the optimizer and
        # the checkpointers take their groups as arguments. On a single process
        # it is a no-op, so the same code runs from one device to a full mesh.
        with self._param_context(), spmd_context(self.parallel_dims):
            logits = self.model(input_ids, positions=positions)
            loss_sum = self._loss_sum(logits, labels, num_valid_tokens=num_valid_tokens)
            del logits
            # Normalize BEFORE backward, while the sum is still differentiable.
            # Dividing after backwarding the raw sum would work for a single
            # group but not under accumulation: gradients add, so N groups each
            # having backpropped an un-normalized sum would have to be rescaled
            # afterwards, and the intermediate buffers (`clip_grad_norm_`'s
            # norm) would already have been computed from the wrong values.
            # Dividing here, inside the graph, is also what keeps the backward
            # numerically identical to `loss.backward()` on a pre-divided loss
            # -- it is the division that is linear, not an extra op.
            (loss_sum / global_valid_tokens).backward()
        return loss_sum.detach()

    def _pp_microbatches(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Split the rank's batch into the schedule's micro-batches.

        Rows are split, never tokens: each micro-batch is flattened with the
        same ``_flatten`` semantics as the non-PP body, so every micro-batch
        holds whole documents and the per-micro-batch loss is the same summed
        CE. Divisibility is enforced at setup (``apply_pp``), so ``chunk``
        never produces a short final piece.
        """
        num_microbatches = self.cfg.parallel.num_pp_microbatches
        # ``labels`` arrives flat from ``_as_batch``; ``input_ids`` keeps the
        # row shape, so re-row the labels against it before chunking -- a flat
        # chunk would split rows whenever seq_len did not divide evenly.
        num_rows, seq_len = input_ids.shape
        labels = labels.reshape(num_rows, seq_len)
        input_mbs = [mb.reshape(-1) for mb in input_ids.chunk(num_microbatches, dim=0)]
        label_mbs = [mb.reshape(-1) for mb in labels.chunk(num_microbatches, dim=0)]
        return input_mbs, label_mbs

    def _pp_forward_backward_body(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """The pipeline-parallel body: drive the schedule instead of the model.

        Only the first stage is handed ``input_ids`` (``arg_mbs``) and only the
        last the labels (``target_mbs``); intermediate stages receive the
        previous stage's activations over the schedule's p2p channel. The
        schedule's loss is the same summed next-token CE the non-PP body
        computes (``pipeline_parallel/pp.py:_scalar_loss_fn``), so the return
        keeps the caller's normalization unchanged: the sum over the last
        stage's micro-batches. That sum is over the last stage's *own* shard of
        the sequence, which is why the caller's denominator -- counted before
        the sequence was cut up -- is the right one.

        ``global_valid_tokens`` is threaded to the schedule through
        ``loss_kwargs``, where the loss function divides by it before the
        schedule's backward. The losses the schedule reports are therefore
        sum/G, and they are multiplied back by G here so the caller keeps
        receiving the raw sum it normalizes and reports.

        Every stage receives the batch -- only the first and last *use* it --
        but the token count is not taken here: the caller needs it before the
        micro-batches are cut, and a stage's count would be over its own slice.
        """
        input_mbs, label_mbs = self._pp_microbatches(
            input_ids.to(self.device), labels.to(self.device)
        )

        losses: list[torch.Tensor] | None = [] if self.pp_has_last_stage else None
        with self._param_context(), spmd_context(self.parallel_dims):
            self.pp_schedule.step(
                arg_mbs=(
                    [(mb,) for mb in input_mbs] if self.pp_has_first_stage else None
                ),
                target_mbs=label_mbs if self.pp_has_last_stage else None,
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

        The ordering mirrors torchtitan's and it is the whole content of this
        function: zero the gradients, snapshot the learning rate, read *every*
        micro-batch the step will consume, reduce the token count those batches
        imply, then run the forward/backward groups, clip, check finiteness,
        step the optimizer, and finally normalize and reduce the loss for
        reporting. Reads that must agree across every micro-batch of a step --
        the denominator and the lr snapshot -- are taken up front, before any
        of them is consumed.

        With ``gradient_accumulation_steps > 1`` the loop below runs the whole
        forward/backward once per group, and the optimizer advances once at the
        end. Each group's loss is still divided by the *step's* global token
        count, not its own, so the accumulated gradient is the step's summed
        loss over the step's token total -- the same quantity a single group
        would produce if the batch had not been split.
        """
        # ``set_to_none=True`` is what the reference uses whenever CUDA graphs
        # are off, and hpmesh runs no graph path: freeing the gradient buffers
        # outright rather than zeroing them in place is both cheaper and what
        # makes the accumulated-gradient bookkeeping below trivially correct.
        self.optimizer.zero_grad(set_to_none=True)

        # Snapshot the lr *before* the schedule advances below. The value
        # reported for a step must be the one the optimizer applied during it;
        # reading after ``lr_scheduler.step()`` would report the next step's
        # value and, on the last step, one past the end of the schedule.
        lr_metrics = self.lr_scheduler.get_metrics()
        should_log = self.should_log()

        # The meshes are resolved once here rather than inline at each
        # collective, and each reduction gets the group *its* quantity spans.
        #
        # The token count is taken from the unsharded batch, so every CP rank
        # of a DP group holds the same number: summing over dp (replicate *
        # shard) is the whole batch's count, once.
        #
        # The loss is summed over each rank's own *slice* of the sequence, so
        # it needs the dp * cp group -- that sum reaches every token exactly
        # once, whereas a dp-only sum would miss the shards held by the other
        # CP ranks and report an average cp times too large. The two coincide
        # when CP is off, which is why one mesh serves both. Divides by a
        # multiple of the true total either way, so any correct-enough group
        # gives the same *average*; correctness is what rules out the
        # alternatives, not the arithmetic.
        #
        # ``get_optional_mesh``, not ``get_mesh``: a pure-PP run has a size-1
        # loss axis, which ``get_mesh`` rejects, and a size-1 axis has nothing
        # to reduce over anyway. Under PP each stage's subgroup reduces
        # independently and only the last stage's (the metrics rank's) is ever
        # logged.
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
            else self.parallel_dims.get_optional_mesh("loss")
        )

        # Read the whole step's data up front. The denominator must be known
        # before the first forward (the loss divides by it there), so every
        # batch that feeds this step has to have been read by then -- and,
        # within a batch, the count has to be taken before the sequence is cut
        # up for CP or PP, both of which turn one whole-batch count into
        # per-rank slices. Only the tensors the step will consume are kept;
        # the rest of the batch is dropped here rather than held across the
        # accumulation window.
        microbatches: list[dict[str, Any]] = []
        loss_sums: list[torch.Tensor] = []
        local_valid_tokens = 0
        for _ in range(self.cfg.gradient_accumulation_steps):
            batch = next(data_iterator)
            # Accounting happens before the shape is normalized so it is
            # per-architecture and shared by both loaders: every token the
            # batch carries counts, whether or not it predicts anything.
            labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
            self.ntokens_seen += labels.numel()
            self.metrics.add_tokens(labels.numel())
            microbatches.append(
                dict(
                    zip(
                        ("input_ids", "labels", "positions", "num_valid_tokens"),
                        self._as_batch(batch),
                        strict=False,
                    )
                )
            )
            # Both sources count here, before the sequence is sharded for CP.
            # That is the number the normalization wants: the loss sums over
            # post-shard tokens, so dividing by a pre-shard global total gives
            # the mean cross-entropy over the whole batch rather than over each
            # rank's slice of it, and the result does not move when cp changes.
            num_valid_tokens = microbatches[-1]["num_valid_tokens"]
            assert num_valid_tokens is not None
            local_valid_tokens += num_valid_tokens

        # Keep the count on device so normalizing the loss adds no device sync
        # to the training path.
        local_valid_tokens_tensor = torch.tensor(
            local_valid_tokens, dtype=torch.int64, device=self.device
        )
        global_valid_tokens = dist_sum_tensor(local_valid_tokens_tensor, dp_mesh)

        # Auxiliary losses normalize by the same per-step token count as the
        # main loss, so their scale is independent of parallelism degrees.
        # Set once, before any forward: every micro-batch of the step divides
        # by the same number.
        if AuxLoss._group_counts:
            AuxLoss.set_step_denominator(global_valid_tokens)

        # Process each group, then free it. ``loss_sums`` accumulates the raw
        # per-group sums, not the normalized returns, so the reported loss is
        # the step total over the step's global token count.
        for microbatch in microbatches:
            loss_sums.append(
                self.forward_backward_step(
                    microbatch["input_ids"],
                    microbatch["labels"],
                    positions=microbatch["positions"],
                    num_valid_tokens=microbatch["num_valid_tokens"],
                    global_valid_tokens=global_valid_tokens,
                )
            )

        loss_sum = torch.sum(torch.stack(loss_sums))

        grad_norm = clip_grad_norm_(
            [p for part in self.model_parts for p in part.parameters()],
            max_norm=self.cfg.max_norm,
            foreach=True,
            pp_mesh=pp_mesh,
        )

        # Finiteness is reduced to ONE flag before it is asserted, and every
        # rank enters the reduction. Asserting on the local loss instead would
        # let a rank whose own shard happened to come out finite sail past a
        # step that another rank already knows is garbage -- and the parameter
        # update that follows is collective, so the disagreement is not
        # recoverable. int32, not bool: NCCL has no bool reduction.
        step_is_finite = torch.ones((), dtype=torch.int32, device=self.device)
        step_is_finite.logical_and_(torch.isfinite(loss_sum).all())
        # Only the last PP stage holds a real loss; the others carry the
        # sentinel, which is finite by construction and says nothing. Skipping
        # the loss-mesh reduction there matches torchtitan and costs nothing --
        # the flag still crosses stages through the pp reduction below.
        if pp_mesh is None or self.pp_has_last_stage:
            if loss_mesh is not None:
                dist.all_reduce(
                    step_is_finite,
                    op=dist.ReduceOp.MIN,
                    group=loss_mesh.get_group(),
                )
        if pp_mesh is not None:
            dist.all_reduce(
                step_is_finite, op=dist.ReduceOp.MIN, group=pp_mesh.get_group()
            )
        # grad_norm arrives already world-reduced (clip_grad_norm_ materializes
        # the DTensor norm and reduces across PP), so this term is the same on
        # every rank; it is folded in for the reader, not for the reduction.
        step_is_finite.logical_and_(torch.isfinite(grad_norm).all())

        self._check_finite(step_is_finite)

        # Before the optimizer update: a background checkpoint save may still be
        # staging, and letting it overlap the update would have two writers
        # touching the model's state dict at once. A no-op when the backend does
        # not stage.
        self.checkpointer.maybe_wait_for_staging()

        self.optimizer.step()
        # After the update, so the lr the optimizer just applied is the one this
        # schedule produced for the previous step -- which is what makes step 1
        # run at ``lambda(0)`` rather than ``lambda(1)``. The snapshot taken at
        # the top of this function is the value handed to the optimizer.
        self.lr_scheduler.step()

        # Summed over tokens, divided by the global count: the loss is then
        # independent of how the batch was split across DP ranks or across
        # accumulation groups. Division by a tensor keeps it on device. Above
        # ``accumulation_steps == 1`` this lands near the per-batch value but
        # not on it -- the denominator is the whole window's token count while
        # only part of the window has contributed -- so early steps of a long
        # accumulation read slightly low. That is the value consistent with the
        # gradients the optimizer just applied.
        loss = loss_sum / global_valid_tokens

        # Only a logging step derives the two reported losses: they are the only
        # place this function touches the host, and the metrics dict is typed
        # for floats. The loss the optimizer used was the tensor above.
        if not should_log:
            return None

        if loss_mesh is not None and local_valid_tokens > 0:
            # The maximum is over each rank's *average* loss, so the local sum
            # is divided by the local count -- a rank holding a short slice of
            # the sequence is not penalized for it.
            local_avg = loss_sum / local_valid_tokens_tensor
            global_avg_loss = float(dist_sum(loss, loss_mesh))
            global_max_loss = float(dist_max(local_avg, loss_mesh))
        else:
            # Single rank, or a rank holding no valid tokens: the two are the
            # same number by construction in the first case, and the second has
            # no local average to report.
            global_avg_loss = global_max_loss = float(loss)
        metrics = {
            "loss": global_avg_loss,
            "max_loss": global_max_loss,
            "grad_norm": float(grad_norm),
        }
        # The snapshot from the top of the step: reported, not checkpointed.
        # The schedule is deterministic in the step number (see
        # load_state_dict), so a resumed run's lr is a pure function of counters
        # that already round-trip; a saved copy could only go stale against
        # them.
        metrics.update(lr_metrics)
        # Aux-loss step registers are rolled up by the optimizer step pre-hook
        # above; this reduces them for logging. Single-process runs skip the
        # collection (there is no mesh to reduce over and no second rank's
        # contribution); the injection itself is unaffected.
        if AuxLoss._group_counts and self.parallel_dims is not None:
            metrics.update(collect_aux_loss_metrics(self.parallel_dims))
        return metrics

    def _check_finite(self, step_is_finite: torch.Tensor) -> None:
        """Stop before the optimizer update if anything went non-finite.

        ``step_is_finite`` is already the *global* verdict: ``train_step`` folds
        this rank's loss and grad_norm into it, then reduces it across the loss
        and PP meshes. The reduction is what every rank participates in, which
        is why it happens in the caller -- a rank that skipped it would hang the
        others, and the check would no longer be rank-uniform.

        ``torch._assert_async`` is private, but it is the right tool: it queues
        the check on the device instead of synchronizing the host, so it costs
        nothing per step and does not break CUDA-graph capture. A failed CUDA
        assertion invalidates the process, which is why the reference
        implementation accepts it.
        """
        torch._assert_async(
            step_is_finite,
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

            # ``batch_generator`` wraps the bare source so every fetch carries
            # its token and loading-time accounting; the loop below sees only
            # batches. One exception type crosses that boundary.
            data_iterator = self.batch_generator(self._data_iterator())
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
                # Takes the cyclic collector over from CPython for the duration
                # of training: the default schedule fires at unpredictable
                # times, often inside a forward, and walking a multi-gigabyte
                # object graph there is pure stall. Built here rather than in
                # ``__init__`` because that is when the model and optimizer
                # exist, so the first collection already covers the real graph.
                self.gc_handler = GarbageCollection(gc_freq=self.cfg.gc_freq)
                while self.should_continue_training():
                    self.step += 1
                    self.gc_handler.run(self.step)

                    try:
                        step_metrics = self.train_step(data_iterator)
                    except DataloaderExhaustedError:
                        # The step is abandoned rather than trained on a
                        # partial batch: a batch's worth of tokens either all
                        # contribute to a gradient or none of them do.
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
            # Teardown lives in ``close`` rather than inline so a caller that
            # drives the trainer programmatically -- rather than through
            # ``train`` -- gets the same cleanup from the same place. It is in
            # a ``finally`` so a run that dies mid-loop still finishes the
            # checkpoint it had already started writing.
            self.close()

    def close(self) -> None:
        """Release everything ``train`` acquired, in reverse order of use.

        Safe to call more than once: each release is guarded, so a trainer that
        was never fully built (or is being closed twice) does not raise on the
        cleanup path, where an exception would mask the real failure.
        """
        if self.checkpointer is not None:
            # Drains any async save still in flight and stops the purge thread.
            self.checkpointer.close()
        if self.metrics is not None:
            self.metrics.close()
        if self.dataloader is not None:
            # Releases the Grain prefetch thread. Without this the loader is
            # only collected at interpreter shutdown, where its ``__del__``
            # raises against an already-torn-down state.
            self.dataloader.close()

        if dist.is_initialized():
            dist.destroy_process_group()
