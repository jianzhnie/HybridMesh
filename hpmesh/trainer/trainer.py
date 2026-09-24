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
  rank of the workload and for every micro-batch of a step, and it is reduced
  over the DP axis alone. Sharding the sequence must not change the reported
  loss, which is why the *loss* reduce-group is chosen differently: it follows
  the sequence, so it spans dp * cp whenever either is enabled.
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
  only has to call ``add_tokens`` and ``log``. ``n_tokens_seen`` -- the
  checkpointed cumulative count -- is logged alongside them, summed over the
  same group the loss average spans so it counts each token once.
* **A lowered process-group timeout once training is under way.** The groups
  are created with the long startup timeout, because that is what model build
  and the first collective genuinely need; ``train`` drops every one of them
  (plus the world group) to ``parallel.train_timeout_seconds`` after this
  process's first completed step, so a later hang is reported in seconds rather
  than mistaken for a slow launch -- see ``accelerator.collectives.set_pg_timeouts``.
* **Profiling**, through ``components/profiler``: ``Profiler`` is entered once
  around the loop and stepped once per iteration, so Kineto traces land on a
  schedule and allocator memory snapshots are written periodically -- plus one
  more if the run dies of an OOM, which is the one that is usually wanted.
* **Periodic validation**, ported from torchtitan's validator: an eval-mode,
  gradient-free pass over a fresh dataloader every ``validation.freq`` steps,
  reporting the summed loss over the *global* valid-token count reduced across
  DP -- the same normalization the training loss uses, so the two numbers are
  comparable. Opt-in via ``training.validation_config``; when it is ``None``
  the loop below is bit-identical to not having the feature. The pass updates
  no parameters and touches no checkpoint state. Zero batches and zero valid
  tokens are loud errors, not a silently skipped report, and the two
  configurations that cannot terminate cleanly (``steps=-1`` with DP > 1, and
  pipeline parallelism, whose schedule hpmesh drives through a train-shaped
  seam) are rejected at build time rather than hanging mid-pass.

``train_step``'s execution order follows torchtitan's and is load-bearing:
zero the gradients, snapshot the learning rate, read *every* batch the step
consumes, reduce the token count, run the forward/backward groups, clip, check
finiteness, wait for any in-flight checkpoint staging, step the optimizer and
then the scheduler, and only then normalize the loss for reporting. Steps whose
value must be identical across all the batches of a step -- the denominator and
the reported lr -- are taken before any of them is consumed.

What was NOT ported: torchtitan's component system (``model_spec``,
``sdc_replayer``, CUDA graphs). Those are infrastructure the loop
calls into, not loop logic, and hpmesh has no counterparts to call.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from datetime import timedelta
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from .. import parallel
from ..accelerator.collectives import clip_grad_norm_, set_pg_timeouts
from ..accelerator.device import (
    device_module,
    device_type,
    get_distributed_backend,
    get_env_dist_info,
)
from ..accelerator.dist import all_reduce
from ..accelerator.dist_utils import _init_dist_pytorch, is_distributed
from ..accelerator.mesh import build_mesh, build_parallel_dims
from ..accelerator.spmd_context import spmd_context
from ..components.checkpointer import DATALOADER, TRAIN_STATE, CheckpointManager
from ..components.loss import (
    IGNORE_INDEX,
    chunked_lm_head_cross_entropy,
    next_token_targets,
)
from ..components.metrics import MetricsProcessor
from ..components.optimizer import (
    EMA,
    LRSchedulersContainer,
    OptimizersContainer,
    build_lr_scheduler,
)
from ..components.profiler import Profiler
from ..datasets import build_dataloader
from ..datasets.loader import BaseDataLoader, DataloaderExhaustedError, TrainerBatch
from ..datasets.random_data import Batch, DataLoaderExhausted, RandomTokenDataLoader
from ..models.common.aux_loss import (
    AuxLoss,
    collect_aux_loss_metrics,
    register_aux_loss_zero_hook,
)
from ..models.common.moe import (
    MoE,
    register_moe_load_balancing_hook,
    register_moe_quantile_balancing_hook,
)
from ..models.hf_state_dict_adapter import HFTransformerStateDictAdapter
from ..models.hf_wrapper import (
    HFTransformerModel,
    build_model_config_for,
    materialize_meta_model,
    num_flops_per_token,
)
from ..parallel.parallel_dims import ParallelDims
from ..parallel.pipeline_parallel import PipelineParallelSetup
from ..parallel.tensor_parallel.tp import (
    ColwiseLinear,
    ColwiseLinearNoGather,
    RowwiseLinear,
)
from ..utils.gc import GarbageCollection
from ..utils.logger_utils import get_logger
from ..utils.seed import derive_distinct_seed
from .config import HybridMeshConfig, ValidationConfig

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
    _chunked_loss_num_chunks: int
    optimizer: torch.optim.Optimizer
    # Defaulted, not just annotated: ``_data_iterator`` reads it, and that is
    # the one helper the tests drive off a ``Trainer`` built with ``__new__``.
    # A real Trainer always holds a loader (it is registered in the checkpoint
    # states), so ``None`` is only the not-built state of such a test double.
    dataloader: BaseDataLoader | None = None
    lr_scheduler: LRSchedulersContainer | None
    ema: EMA | None
    checkpointer: CheckpointManager | None
    metrics: MetricsProcessor | None
    gc_handler: GarbageCollection | None

    # Additional training state, saved in the checkpoint.
    step: int
    ntokens_seen: int

    def __init__(self, cfg: HybridMeshConfig):
        self.cfg = cfg
        if (
            not is_distributed()
            and "RANK" in os.environ
            and "WORLD_SIZE" in os.environ
        ):
            _init_dist_pytorch(get_distributed_backend())
        self.rank, self.world_size, self.local_rank = get_env_dist_info()

        # Resolve the degrees first: the PP seed offset below needs this rank's
        # stage coordinate, and degree resolution draws no random numbers, so
        # seeding after it leaves every non-PP run bit-identical.
        self.parallel_dims = build_parallel_dims(cfg, self.world_size)

        # Validation's infeasible combinations are rejected here, before the
        # model and dataloader exist: a ``steps=-1`` pass that cannot terminate
        # cleanly would otherwise hang on its collectives mid-run, and a
        # pipeline-parallel pass has no eval seam to run through at all.
        if cfg.validation is not None:
            self._check_validation_feasibility(
                cfg.validation,
                pp_enabled=(
                    self.parallel_dims is not None and self.parallel_dims.pp_enabled
                ),
                dp_world_size=(
                    1
                    if self.parallel_dims is None
                    else self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard
                ),
                training_dataset=cfg.dataloader.dataset,
            )

        # Deterministic seeding BEFORE model build so ranks sharing an SPMD
        # group build identical initial weights -- the precondition for
        # bit-exact DP comparisons. Pipeline stages hold different layers, so
        # seeding every stage identically would correlate their initialization;
        # under PP each stage offsets the base seed by its stage rank (the
        # upstream distinct_seed_mesh_dims=["pp"] semantics), while ranks at
        # the same stage keep the base seed.
        seed = cfg.seed
        if self.parallel_dims is not None and self.parallel_dims.pp_enabled:
            pp_mesh = self.parallel_dims.get_optional_mesh("pp")
            seed = derive_distinct_seed(
                seed, [(pp_mesh.get_local_rank(), pp_mesh.size())]
            )
        self._seed_everything(seed, deterministic=cfg.deterministic)

        self.device = torch.device(
            f"{device_type}:{self.local_rank}" if device_type != "cpu" else "cpu"
        )

        # 1. mesh (the process topology every dimension is built on). ``parallel_dims``
        #    is the same resolved degrees the mesh was built from, kept so the
        #    trainer can ask "how many DP ranks?" without re-indexing the mesh.
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
        #
        # EP expert tensors are rank-heterogeneous plain tensors. Until they
        # have an EP-aware checkpoint representation, any save or load would
        # silently collapse all ranks onto one expert slice. Reject the whole
        # checkpoint surface before model construction rather than merely warn.
        if (
            self.parallel_dims is not None
            and self.parallel_dims.ep_enabled
            and cfg.checkpoint.enable
        ):
            raise NotImplementedError(
                f"expert_parallel_size={self.parallel_dims.ep} with checkpointing "
                "is not supported: expert weights are rank-heterogeneous plain "
                "tensors and the current checkpoint backends treat them as "
                "replicated. Disable checkpointing until EP-aware expert state "
                "serialization is implemented."
            )
        # Chunked loss + PP is rejected up front: under PP the last stage's
        # loss is computed inside the schedule
        # (``pipeline_parallel/apply.py:_scalar_loss_fn``), which receives logits
        # from the stage forward. Rewiring that seam for hidden states plus a
        # per-chunk backward is a PP-side change, so the combination loud-raises
        # here rather than training on a silently un-chunked (or wrong) loss.
        self._chunked_loss_num_chunks = cfg.training.chunked_loss_num_chunks
        if (
            self._chunked_loss_num_chunks > 1
            and self.parallel_dims is not None
            and self.parallel_dims.pp_enabled
        ):
            raise NotImplementedError(
                f"chunked_loss_num_chunks={self._chunked_loss_num_chunks} with "
                f"pipeline_parallel_size={self.parallel_dims.pp} is not "
                "supported: the pipeline last stage's loss runs inside the "
                "schedule on materialized logits. Run chunked loss without "
                "pipeline parallelism."
            )
        hf_model_config = build_model_config_for(cfg)
        load_hf_weights = bool(
            cfg.checkpoint.enable
            and cfg.checkpoint.initial_load_in_hf
            and cfg.checkpoint.initial_load_path
        )
        if load_hf_weights:
            with torch.device("meta"):
                model = HFTransformerModel(hf_model_config)
        else:
            model = HFTransformerModel(hf_model_config).to(self.device)

        # 3. parallelism, in Titan's order: tp/pp/cp/ep declared first, fsdp last
        #    (outer wraps inner). Each is a no-op when its degree is 1. The
        #    parallel layer's contract is ParallelConfig plus explicit scalars,
        #    so the training-side values it needs are unpacked here.
        orchestration = parallel.parallelize_hf_transformers(
            model,
            cfg=cfg.parallel,
            mesh=self.mesh,
            parallel_dims=self.parallel_dims,
            device=self.device,
            compile=cfg.training.compile,
            compile_config=cfg.training.compile_config,
            activation_checkpoint=cfg.training.activation_checkpoint_mode,
            selective_ac=cfg.training.selective_ac,
            memory_budget_ac=cfg.training.memory_budget_ac,
            global_batch_size=cfg.training.global_batch_size,
            dataset=cfg.training.dataloader.dataset,
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

        if load_hf_weights:
            for model_part in self.model_parts:
                materialize_meta_model(model_part, self.device)

        self.optimizer = OptimizersContainer(
            cfg.optimizer, model_parts=self.model_parts
        )

        # The lr schedule. Built regardless of whether the knobs were touched:
        # the default is warmup_steps=0 with no decay, so the factor is a
        # constant 1.0 and step 1 runs at exactly ``cfg.lr``. That costs one
        # multiply per step and removes the branch that would otherwise decide
        # whether the lr is scheduled -- a branch whose two sides would have to
        # be kept numerically identical forever.
        #
        # Handed the *inner* optimizers, not the container: a LambdaLR reads
        # ``lr`` off its optimizer's param groups, and the container's own
        # groups carry none (they are the merged parameter view). This is why
        # the scheduler is a container too.
        self.lr_scheduler = build_lr_scheduler(
            cfg.lr_scheduler_config,
            optimizers=list(self.optimizer),
            training_steps=cfg.steps,
        )

        # The weight EMA, a sibling of the optimizer rather than part of it:
        # stepped explicitly in ``train_step`` after the real update, and
        # registered with the checkpointer under its own ``ema`` key. Built
        # only when configured -- None costs nothing.
        ema_config = cfg.training.ema
        self.ema = (
            EMA(
                model_parts=self.model_parts,
                decay=ema_config.decay,
                half_life_fraction=ema_config.half_life_fraction,
                start_step=ema_config.start_step,
                step_bias=ema_config.step_bias,
                update_every_n_steps=ema_config.update_every_n_steps,
                buffer_patterns=ema_config.buffer_patterns,
            )
            if ema_config is not None
            else None
        )

        # Aux losses (the MoE load-balance loss a swapped-in MoE carries)
        # accumulate per forward; this pre-hook rolls the per-instance sums
        # into the step registers at each optimizer step. Harmless when no
        # aux loss exists.
        #
        # Registered on the container, so it fires once per step() call --
        # not once per inner optimizer, which is what a loop over the inner
        # optimizers would give under pipeline parallelism.
        register_aux_loss_zero_hook(
            self.optimizer, self.model_parts, self.parallel_dims
        )
        # A second pre-hook on the same container, same granularity. No-op for
        # a model without MoE layers, which is every model except a swapped-in
        # one (the swap is what installs ``load_balance_coeff``).
        register_moe_load_balancing_hook(
            self.optimizer, self.model_parts, self.parallel_dims
        )
        # The quantile counterpart, registered alongside: the two schemes are
        # mutually exclusive per model, so exactly one of the two hooks ever
        # fires -- this one no-ops unless the swap installed quantile routers
        # (``moe_quantile_balancing``).
        register_moe_quantile_balancing_hook(
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
        #    The schedule rides along for one integer, ``last_epoch``, that
        #    nothing else in the checkpoint carries. The optimizer restores its
        #    ``base_lrs`` -- so the *current* lr comes back right -- but
        #    ``last_epoch`` is the scheduler's own counter, and a resumed run's
        #    fresh scheduler starts it at 0. Without it the curve restarts from
        #    the beginning on the step after a resume: silent whenever warmup
        #    and decay are both off (the lr is then constant and the mistake
        #    invisible), and wrong for the rest of the run once either is set.
        states: dict[str, Any] = {TRAIN_STATE: self}
        if self.dataloader is not None:
            states[DATALOADER] = self.dataloader
        self.checkpointer = CheckpointManager(
            cfg.checkpoint,
            model_parts=self.model_parts,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            ema=self.ema,
            states=states,
            folder=cfg.dump_folder,
            sd_adapter=HFTransformerStateDictAdapter(
                hf_model_config, cfg.checkpoint.initial_load_path or cfg.hf_model
            ),
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
        # Under PP the loss exists on one rank and ``LOG_RANK`` decides which
        # ranks print, so a mismatched pair trains correctly and reports
        # nothing -- which reads exactly like a hang. Warn now, rather than
        # leave the user to work it out at step 1.
        self.metrics.ensure_pp_loss_visible()

    # -- setup helpers ---------------------------------------------------------

    @staticmethod
    def _seed_everything(seed: int, *, deterministic: bool) -> None:
        torch.manual_seed(seed)
        if device_type != "cpu":
            device_module.manual_seed_all(seed)
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

    def _batch_size_per_rank(self, dp_world_size: int) -> int:
        """This rank's share of the global batch, checked rather than floored.

        Both loader paths divide the global batch by ``dp_world_size`` -- the
        random path slices rows, the Grain path is handed a token count -- and
        both are wrong in the same silent way when it does not divide: the run
        reads a smaller global batch than the config names, and every number
        derived from it (the lr, the token count, the value logged as
        ``batch_size``) describes a batch that is not the one being read.

        ``RandomTokenDataLoader`` rejects an indivisible ``batch_size`` of its
        own, but only once it is constructed and only on the random path. The
        token count is computed here, before either loader exists, so this is
        the one place both paths pass through -- which is what makes the
        failure the same for both, and the same up front.
        """
        global_batch_size = self.cfg.global_batch_size
        if global_batch_size % dp_world_size != 0:
            raise ValueError(
                f"global_batch_size ({global_batch_size}) must be divisible by "
                f"the number of data-parallel ranks ({dp_world_size}); each rank "
                f"reads global_batch_size // dp_world_size samples, and the "
                f"remainder would be dropped silently."
            )
        return global_batch_size // dp_world_size

    def _build_dataloader(self) -> BaseDataLoader | None:
        """Build the micro-batch source the config names.

        Whatever the source, it rides along in the checkpoint's ``states``:
        resuming without its read position would resume the weights and restart
        the data, silently training a second pass over the beginning of the
        corpus. The synthetic loader's ``load_state_dict`` reaches the saved
        position by replaying generated batches, which is exact (batch k is a
        pure function of ``(seed, k)``) if not free.
        """
        dp_rank, dp_world_size = self._dp_rank_world_size()
        batch_size_per_rank = self._batch_size_per_rank(dp_world_size)
        loader = build_dataloader(
            self.cfg,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            # Per rank, not global: the Grain loader splits every dataset's
            # rows across ``dp_world_size`` ranks itself, so this many tokens
            # per rank is this many tokens per rank of the global batch.
            num_tokens_per_batch=batch_size_per_rank * self.cfg.max_seq_len,
        )
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
    def _count_valid_tokens(batch: Batch | TrainerBatch) -> int:
        """The number of labels that contribute to the loss, pre-shard.

        The trainer's half of the token accounting, and it stays in the trainer
        for a reason: the count divides the loss *before* the first backward and
        is reduced across DP before that, so it cannot be produced per
        micro-batch by a model-side counter.

        It is taken from the batch as the loader handed it over -- before the
        model normalizes shapes, shifts for row ends, or shards for CP -- which
        is what makes it rank-independent: every CP rank of a DP group holds
        the same unsharded batch, so the dp-only reduction counts the whole batch
        once rather than once per sequence shard.

        A collator counts its own tokens while it already has the labels in
        hand; the synthetic source has no collator, so its count is derived here
        from the row shift. Both agree on what counts: a document's final
        position predicts nothing and is excluded.
        """
        if isinstance(batch, Batch):
            seq_len = batch.labels.shape[-1]
            targets = next_token_targets(batch.labels.reshape(-1), seq_len=seq_len)
            return int((targets != IGNORE_INDEX).sum())
        num_valid_tokens = batch.get("num_valid_tokens")
        if num_valid_tokens is None:
            # Recounted rather than permissively defaulted, so a dict that
            # silently lacks the key still produces a correct denominator.
            labels = batch["labels"]
            num_valid_tokens = int((labels != IGNORE_INDEX).sum())
        return num_valid_tokens

    # -- the step, one function per level --------------------------------------

    @property
    def _example_model(self):
        """The model a batch is normalized against, present on every PP stage.

        Unlike ``self.model`` (``None`` under PP, where this rank holds several
        chunks and the schedule drives them), every rank keeps a module that can
        run ``preprocess_inputs``: the first stage owns the embedding chunk. The
        PP path already assumes as much -- that is how it builds the schedule.
        """
        return self.model if self.model is not None else self.model_parts[0]

    def _microbatch(self, batch: Batch | TrainerBatch) -> dict[str, Any]:
        """Everything one accumulation group's forward/backward needs.

        The split of responsibility here mirrors torchtitan's ``train_step``,
        and each half is load-bearing:

        * **The count is popped by the trainer.** It is the loss denominator,
          which must be reduced across DP before the first backward, so it
          cannot come out of a per-micro-batch model call.
        * **The accounting is taken by the trainer**, from the loader's own
          labels, before any reshaping: throughput is a report about the loader
          ("tokens it produced"), not about the loss. The two numbers differ --
          a document's final position is loaded but never predicted -- and that
          is why they are not one field. The cumulative count is divided by
          ``cp * tp`` because every rank of a CP/TP group reads the same batch
          and the report sums it over the loss mesh (see ``train_step``).
        * **Everything else stays on the host until its group is consumed.**
          ``_preprocess`` moves one group's tensors to the device just ahead of
          that group's forward (see ``_to_device``), so holding the rest of the
          accumulation window costs host memory, not device memory -- the CPU
          invariant ``torchtitan`` documents for ``batch_generator``.
        """
        labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
        # Count this rank's share of the sequence, not the whole batch: every
        # rank of a CP/TP group reads the *same* batch from the loader, and the
        # sequence is cut across the group later (``shard_batch_for_cp`` /
        # ``shard_batch_for_tp``). Counting the full batch here and then summing
        # over the dp*cp*tp loss mesh in ``train_step`` would report a corpus
        # cp * tp times its true size; counting the share makes that sum
        # reconstruct the tokens actually read.
        parallel_dims = getattr(self, "parallel_dims", None)
        sequence_shards = (
            1 if parallel_dims is None else parallel_dims.cp * parallel_dims.tp
        )
        self.ntokens_seen += labels.numel() // sequence_shards
        num_valid_tokens = self._count_valid_tokens(batch)

        if isinstance(batch, dict):
            # ``num_valid_tokens`` is the model's to ignore, and a plain int
            # among tensors would be splatted into the forward as a kwarg.
            batch.pop("num_valid_tokens", None)
        return {"batch": batch, "num_valid_tokens": num_valid_tokens}

    def _to_device(self, batch: Batch | TrainerBatch) -> Batch | TrainerBatch:
        """Move one consumption group's tensors to the training device.

        Called by ``_preprocess``, once per group just ahead of that group's
        forward -- not at read time. Reading the whole accumulation window onto
        the device up front would keep every micro-batch resident in device
        memory for the whole window, which is exactly what deferring avoids.
        """
        if isinstance(batch, dict):
            return {
                key: value.to(self.device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
        return Batch(
            input_ids=batch.input_ids.to(self.device, non_blocking=True),
            labels=batch.labels.to(self.device, non_blocking=True),
        )

    def _preprocess(
        self, microbatch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Ask the model to turn its batch into forward inputs.

        A thin wrapper so the two bodies call the seam the same way and neither
        has to know which model object is canonical on its stage. The group's
        tensors are moved to the device here -- at consumption, not at read --
        so an accumulation window's unread groups stay on the host.
        """
        return self._example_model.preprocess_inputs(
            self._to_device(microbatch["batch"]),
            parallel_dims=self.parallel_dims,
            parallelism=self.cfg.parallel,
            max_context_length=self.cfg.max_seq_len,
        )

    def forward_backward_step(
        self,
        microbatch: dict[str, Any],
        *,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Run one accumulation group's forward and backward; return its loss sum.

        A *group*, not a micro-batch: with pipeline parallelism one group is one
        schedule step, which internally drives several micro-batches.

        Two bodies, matching torchtitan's split. The PP one takes the raw batch
        and calls ``preprocess_inputs`` itself, once per schedule micro-batch;
        the non-PP one preprocesses here, because it has exactly one.

        ``global_valid_tokens`` is the step's denominator, reduced across the DP
        axis. It is passed in rather than computed here because it must be the
        *same* number for every group of the step -- under gradient
        accumulation the count only exists once all of them have been read, so
        the caller reduces it first and hands it down.
        """
        if self.parallel_dims is not None and self.parallel_dims.pp_enabled:
            return self._pp_forward_backward_body(
                microbatch["batch"], global_valid_tokens=global_valid_tokens
            )
        inputs, labels, extra_kwargs = self._preprocess(microbatch)
        return self._forward_backward_body(
            inputs,
            labels,
            extra_kwargs=extra_kwargs,
            global_valid_tokens=global_valid_tokens,
        )

    def _forward_backward_body(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        *,
        extra_kwargs: dict[str, Any],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # ``spmd_context`` is what makes a process group answerable *by name*
        # (``spmd_mesh_group("tp")`` and friends) for the duration of the body.
        # It is entered here, around the forward/backward only, because that is
        # the region whose components read the ambient mesh -- the optimizer and
        # the checkpointers take their groups as arguments. On a single process
        # it is a no-op, so the same code runs from one device to a full mesh.
        with self._param_context(), spmd_context(self.parallel_dims):
            if self._chunked_loss_num_chunks > 1:
                # Chunked loss: the forward skips lm_head and returns hidden
                # states; lm_head + cross-entropy then run per sequence chunk,
                # so the peak logits memory is 1/num_chunks of the full T*V
                # tensor. The call runs the backward itself (per chunk, scaled
                # by 1/global_valid_tokens -- the same normalization the
                # non-chunked path applies inside the graph) and returns the
                # detached sum.
                hidden_states = self.model(inputs, **extra_kwargs, skip_lm_head=True)
                return chunked_lm_head_cross_entropy(
                    self.model.lm_head,
                    hidden_states,
                    labels,
                    num_chunks=self._chunked_loss_num_chunks,
                    grad_scale=1.0 / global_valid_tokens,
                )
            logits = self.model(inputs, **extra_kwargs)
            loss_sum = self._loss_sum(logits, labels)
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

    @staticmethod
    def _loss_sum(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Summed next-token cross-entropy over the predictable labels.

        ``labels`` arrives already aligned with ``logits`` -- ``logits[t]``
        predicts ``labels[t]``, both sources having done their shift upstream
        (see ``HFTransformerModel.preprocess_inputs``). No shift happens here,
        which is what lets the two sources share one loss: the synthetic path
        slots its rows together and the packed path arrives already shifted, and
        both mark the positions that must not be predicted with ``IGNORE_INDEX``
        rather than dropping them. Those positions are the row ends of the
        synthetic path and the document boundaries and packing padding of the
        packed one.

        Not normalized, and deliberately not told the token count. The
        denominator is a *global* count reduced across DP, which the caller
        owns; the per-rank count that pairs with it is taken upstream from the
        unsharded batch (``_count_valid_tokens``) precisely so a loss that has
        since been sliced by context parallelism cannot be recounted. Passing
        the count in here would suggest this function has a use for it, and a
        recount would silently undercount by a factor of ``cp``.
        """
        return F.cross_entropy(
            logits.float(), labels, reduction="sum", ignore_index=IGNORE_INDEX
        )

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

    def _allreduce_replicated_tp_grads(self) -> None:
        """Sum the gradients of TP-*replicated* parameters across the TP group.

        Under tensor parallelism every rank enters the forward holding only its
        own ``T / tp`` sequence shard (the sequence-parallelism premise), so a
        parameter that is not itself TP-sharded -- the token embedding, the
        RMSNorms, the LM head -- accumulates a gradient over just this rank's
        tokens. No collective inside the TP modules covers them (the fused
        GEMMs reduce only their own sharded weights' gradients), so without
        this all-reduce the copies train on ``1/tp`` of the tokens and drift
        apart. The sharded weights are identified by module type: ``apply_tp``
        realizes every sharded projection as one of the three classes below,
        and everything else in the model is replicated.

        Sum, not average: each rank's partial gradient covers a disjoint set of
        tokens, and the true gradient is the total. No-op when tp == 1.
        """
        tp_mesh = (
            None
            if self.parallel_dims is None
            else self.parallel_dims.get_optional_mesh("tp")
        )
        if tp_mesh is None:
            return
        sharded_ids = {
            id(module.weight)
            for part in self.model_parts
            for module in part.modules()
            if isinstance(module, ColwiseLinear | RowwiseLinear | ColwiseLinearNoGather)
        }
        group = tp_mesh.get_group()
        for part in self.model_parts:
            for param in part.parameters():
                if param.grad is None or id(param) in sharded_ids:
                    continue
                grad = param.grad
                # FSDP2 parameters carry DTensor gradients; reducing the local
                # shard in place is the reduction, since every rank of a TP
                # group holds the same shard of the same parameter.
                if isinstance(grad, DTensor):
                    all_reduce(grad.to_local(), group=group)
                else:
                    all_reduce(grad, group=group)

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
        # The token count is taken from the unsharded batch, so every rank of a
        # TP or CP group holds the same number: summing over dp (replicate *
        # shard) is the whole batch's count, once. The groups that would be
        # wrong are the pure-cp axis (multiplying the count by cp), the tp axis
        # (TP ranks read the same batch), and the ``loss`` axis (by dp * cp *
        # tp) -- all over-count a batch no rank ever held in full.
        #
        # The loss is summed over each rank's own *slice* of the batch -- rows
        # under dp, sequence shards under cp and tp -- so it needs the
        # dp * cp * tp group (the ``loss`` view): that sum reaches every token
        # exactly once, whereas a dp-only sum would miss the sequence shards
        # held by the other CP/TP ranks and report an average cp * tp times too
        # small. The two coincide when CP and TP are off, which is why one mesh
        # serves both averages. Under PP each stage's subgroup reduces
        # independently and only the last stage's (the metrics rank's) is ever
        # logged, so a size-1 loss axis is nothing to reduce over rather than
        # an error -- hence ``get_optional_mesh`` rather than ``get_mesh``.
        #
        # When the loss mesh *is* used is not "is any one parallelism on" but
        # "is the loss split across ranks at all": with cp or tp on and dp = 1
        # the sequence is sharded and dp alone is a size-1 group, so skipping
        # the reduction would report one rank's shard as the whole batch's
        # loss. Gate on the disjunction of all three, the property torchtitan
        # gates on (``dp_cp_enabled``) extended by tp for the sequence-parallel
        # loss shard upstream does not have.
        parallel_dims = self.parallel_dims
        dp_mesh = (
            None if parallel_dims is None else parallel_dims.get_optional_mesh("dp")
        )
        pp_mesh = (
            None if parallel_dims is None else parallel_dims.get_optional_mesh("pp")
        )
        loss_sharded = parallel_dims is not None and (
            parallel_dims.dp_cp_enabled or parallel_dims.tp_enabled
        )
        loss_mesh = (
            dp_mesh
            if pp_mesh is None and not loss_sharded
            else parallel_dims.get_optional_mesh("loss")
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
        local_valid_tokens = 0
        for _ in range(self.cfg.gradient_accumulation_steps):
            # ``_microbatch`` owns the split of responsibility: it takes the
            # count (the denominator) and the accounting off the loader's own
            # batch, before any reshaping, and leaves everything else to the
            # model. Both of those have to happen here rather than per
            # micro-batch: the count has to be reduced across DP before the
            # first backward, and the account is a report about the loader.
            microbatch = self._microbatch(next(data_iterator))
            microbatches.append(microbatch)
            local_valid_tokens += microbatch["num_valid_tokens"]

        # Keep the count on device so normalizing the loss adds no device sync
        # to the training path.
        local_valid_tokens_tensor = torch.tensor(
            local_valid_tokens, dtype=torch.int64, device=self.device
        )
        global_valid_tokens = local_valid_tokens_tensor
        if dp_mesh is not None:
            # Clone before the in-place collective: the local count is read
            # again below for this rank's own per-rank average.
            global_valid_tokens = global_valid_tokens.clone()
            all_reduce(global_valid_tokens, group=dp_mesh.get_group())

        # Auxiliary losses normalize by the same per-step token count as the
        # main loss, so their scale is independent of parallelism degrees.
        # Set once, before any forward: every micro-batch of the step divides
        # by the same number.
        if AuxLoss._group_counts:
            AuxLoss.set_step_denominator(global_valid_tokens)

        # Process each group, then free it. Loss values are retained only on a
        # logging step: backward has already consumed them, and non-logging
        # steps need only the on-device finiteness verdict. On logging steps,
        # take ownership of the first detached value and accumulate later
        # groups in place, avoiding a list plus a final stack proportional to
        # ``gradient_accumulation_steps``.
        accumulated_loss: torch.Tensor | None = None
        # int32 is supported by NCCL reductions, unlike bool.
        loss_is_finite = torch.ones((), dtype=torch.int32, device=self.device)
        for microbatch in microbatches:
            detached_loss = self.forward_backward_step(
                microbatch, global_valid_tokens=global_valid_tokens
            )
            local_loss = (
                detached_loss.to_local()
                if isinstance(detached_loss, DTensor)
                else detached_loss
            )
            loss_is_finite.logical_and_(torch.isfinite(local_loss).all())
            if should_log:
                if accumulated_loss is None:
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)

        # After the last backward, before clipping: replicated parameters under
        # TP hold token-partial gradients that nothing else reduces.
        self._allreduce_replicated_tp_grads()

        parameters = [p for part in self.model_parts for p in part.parameters()]
        expert_parameters = [
            p
            for part in self.model_parts
            for module in part.modules()
            if isinstance(module, MoE)
            for p in module.routed_experts.inner_experts.parameters()
        ]
        ep_mesh = (
            self.parallel_dims.get_optional_mesh("ep")
            if self.parallel_dims is not None and self.parallel_dims.ep_enabled
            else None
        )
        grad_norm = clip_grad_norm_(
            parameters,
            max_norm=self.cfg.max_norm,
            foreach=True,
            pp_mesh=pp_mesh,
            ep_mesh=ep_mesh,
            expert_parameters=expert_parameters if ep_mesh is not None else None,
        )

        # Finiteness is reduced to ONE flag before it is asserted, and every
        # rank enters the reduction. Asserting on the local loss instead would
        # let a rank whose own shard happened to come out finite sail past a
        # step that another rank already knows is garbage -- and the parameter
        # update that follows is collective, so the disagreement is not
        # recoverable. int32, not bool: NCCL has no bool reduction.
        step_is_finite = loss_is_finite
        # Only the last PP stage holds a real loss; the others carry the
        # sentinel, which is finite by construction and says nothing. Skipping
        # the loss-mesh reduction there matches torchtitan and costs nothing --
        # the flag still crosses stages through the pp reduction below.
        if pp_mesh is None or self.pp_has_last_stage:
            if loss_mesh is not None:
                all_reduce(
                    step_is_finite,
                    op="min",
                    group=loss_mesh.get_group(),
                )
        if pp_mesh is not None:
            all_reduce(
                step_is_finite, op="min", group=pp_mesh.get_group()
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
        if self.ema is not None:
            # ``self.step`` is the step just optimized, which is what the EMA
            # schedule's start_step/update_every_n_steps are defined against.
            self.ema.step(self.step)

        if not should_log:
            return None

        assert accumulated_loss is not None

        # Summed over tokens, divided by the global count: the loss is then
        # independent of how the batch was split across DP ranks or across
        # accumulation groups. Division by a tensor keeps it on device. Above
        # ``accumulation_steps == 1`` this lands near the per-batch value but
        # not on it -- the denominator is the whole window's token count while
        # only part of the window has contributed -- so early steps of a long
        # accumulation read slightly low. That is the value consistent with the
        # gradients the optimizer just applied.
        loss = accumulated_loss / global_valid_tokens

        if loss_mesh is not None:
            # The collectives are entered UNCONDITIONALLY: gating them on a
            # local predicate (this rank saw no valid tokens this window) would
            # let that rank skip a reduction the others enter, hanging the
            # step. Only the per-rank division needs the guard -- a rank with
            # no valid tokens contributes 0 to the max.
            local_avg = (
                accumulated_loss / local_valid_tokens_tensor
                if local_valid_tokens > 0
                else torch.zeros_like(accumulated_loss)
            )
            loss_sum = loss.clone()
            local_max = local_avg.clone()
            all_reduce(loss_sum, group=loss_mesh.get_group())
            all_reduce(local_max, op="max", group=loss_mesh.get_group())
            global_avg_loss = float(loss_sum)
            global_max_loss = float(local_max)
            # Cumulative tokens seen, summed over the ranks holding *distinct*
            # tokens: ``ntokens_seen`` is a count of labels this rank actually
            # fed a step, and CP and TP each take their own slice of that
            # sequence, so one rank's slice is a strict subset. ``loss_mesh`` is
            # the group those slices partition -- the same one the loss average
            # above spans. ``dp_mesh`` is its subgroup: summing over dp alone
            # would under-count by ``cp * tp``, exactly as it would for the loss.
            # (The two coincide when CP and TP are off, which is why one mesh
            # serves both.)
            #
            # Unlike ``global_valid_tokens``, which counts only the *predictable*
            # labels (the loss denominator), this counts every label -- the data
            # consumed. Both are per-step sums over the same meshes, so they
            # differ by exactly the final position of each document.
            #
            # One host sync per logging step, not per step: the tensor is int64
            # and nothing downstream needs it on the device.
            ntokens_seen_tensor = torch.tensor(
                self.ntokens_seen, dtype=torch.int64, device=self.device
            )
            all_reduce(ntokens_seen_tensor, group=loss_mesh.get_group())
            global_ntokens_seen = float(ntokens_seen_tensor)
        else:
            # Single rank: the two reported losses are the same number by
            # construction.
            global_avg_loss = global_max_loss = float(loss)
            global_ntokens_seen = float(self.ntokens_seen)
        metrics = {
            "loss": global_avg_loss,
            "max_loss": global_max_loss,
            "grad_norm": float(grad_norm),
            "n_tokens_seen": global_ntokens_seen,
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

    # -- validation ---------------------------------------------------------------

    @staticmethod
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

            # The step this process's *first* train step lands on. Startup work --
            # process groups, the model build, the first collective, compile --
            # runs under the long process-group timeout those groups were created
            # with, because that is what genuinely takes minutes. Once one step
            # has completed, that work is behind this process, so the timeout can
            # be lowered to the one a real stall should be measured against (see
            # ``set_pg_timeouts``). Relative to the loaded step rather than
            # absolute ``1`` so a resumed run lowers it too.
            first_step_of_this_process = self.step + 1

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

                    # Validation, after the checkpoint save and before the
                    # profiler advances -- the reference's order. The pass is
                    # eval-only and leaves no state behind, so its position in
                    # the step affects only which step's weights it scores.
                    if self.should_validate(self.step):
                        self.validate(self.step)

                    # Advances the schedule. After the save, so the profiler's
                    # active iteration covers an ordinary step rather than one
                    # that also wrote a checkpoint.
                    profiler.step()

                    if self.step == first_step_of_this_process:
                        # Startup is finished on this process; from here a long
                        # wait is a stall, not a slow launch. Skipped entirely on
                        # a single process: it has no group to time out, and its
                        # barrier would be the only collective in the program.
                        if self.parallel_dims is not None:
                            set_pg_timeouts(
                                timedelta(
                                    seconds=self.cfg.parallel.train_timeout_seconds
                                ),
                                self.parallel_dims,
                                device=self.device,
                            )
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
