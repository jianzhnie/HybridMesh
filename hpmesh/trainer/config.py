"""hpmesh configuration: every dataclass a run is described by.

This is the one place configuration lives, and every class here is named the
same way -- ``*Config``. What varies is only whether a config is a top-level
parser group or a nested one:

* Top-level groups, handed to ``HfArgumentParser`` in ``train.py`` --
  ``ModelConfig``, ``ParallelConfig``, ``OptimizerConfig``, ``TrainingConfig``.
* Nested configs, each also parsed as its own group and then grafted onto the
  group that holds it -- ``CheckpointConfig``, ``DataloaderConfig``,
  ``MetricsConfig``, ``ProfilerConfig``, ``LRSchedulerConfig``.

These are descriptions, not builders. Nothing here constructs the runtime object
it describes -- the loader in ``datasets/build.py``, the scheduler in
``components/lr_scheduler.py`` -- for two reasons. A config that carries its own
builder suggests the built object is one of its fields, and it is not: building
takes arguments the config does not have (which rank am I, how many tokens per
batch). And a builder has to name the type it produces, which for a config
nested under a component would mean importing the component into this module
while the component imports this one back.

So the seam is a factory function that takes the config as its first argument.
``trainer/trainer.py`` calls those; this module is only ever read.

A component -- the checkpointer, the metrics processor, the profiler, the
learning-rate schedule -- does not define its own config class next to itself
either; it takes an instance from here and reads fields off it. It names the
type only under ``TYPE_CHECKING``, for the same cycle reason.

Design notes (see docs/hybridmesh_design.md): grouped by concern, then COMPOSED
-- not mixed in via multiple inheritance -- so each group's ``__post_init__``
validation runs automatically via ``default_factory``, with no fragile manual
chaining. The single entry point is ``HybridMeshConfig``.

Kept deliberately small for the learning path: one optimizer (adamw), one LR
schedule, deterministic seeding. Add knobs only when a learning step needs them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
from transformers import AutoConfig

from hpmesh.components.checkpointer import LR_SCHEDULER, MODEL, OPTIMIZER
from hpmesh.datasets.hf.text import DATASETS
from hpmesh.utils import filesystem
from hpmesh.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "CheckpointConfig",
    "DataloaderConfig",
    "HybridMeshConfig",
    "LRSchedulerConfig",
    "MetricsConfig",
    "ModelConfig",
    "OptimizerConfig",
    "ParallelConfig",
    "ProfilerConfig",
    "TrainingConfig",
]


@dataclass
class ModelConfig:
    """Model architecture. None fields are auto-filled from the HF config.

    Offline path: when ``model_name_or_path`` has no '/', build a tiny local model
    with AutoConfig.for_model using the explicit sizes below. Online path: a hub id
    ("org/name") pulls real architecture via AutoConfig.from_pretrained.
    """

    model_name_or_path: str = field(
        default="llama",
        metadata={
            "help": "HF architecture name (offline) or hub id 'org/name' (online)"
        },
    )
    vocab_size: int = field(default=128, metadata={"help": "Vocabulary size"})
    hidden_size: int = field(default=64, metadata={"help": "Hidden dimension"})
    intermediate_size: int = field(
        default=128, metadata={"help": "FFN inner dimension"}
    )
    num_hidden_layers: int = field(
        default=2, metadata={"help": "Number of decoder layers"}
    )
    num_attention_heads: int = field(
        default=4, metadata={"help": "Number of attention heads"}
    )
    num_key_value_heads: int = field(
        default=4, metadata={"help": "Number of KV heads (GQA)"}
    )


@dataclass(kw_only=True, slots=True)
class ParallelConfig:
    """The parallelism degrees, plus the process group backend.

    The field names and semantics are torchtitan's, deliberately: the mesh
    builder reads the ``*_degree`` fields directly, so keeping the spelling
    keeps ``ParallelDims`` portable. Short aliases (``tp`` / ``pp`` / ``cp`` /
    ``ep`` / ``dp``) are exposed as properties at the end of the class, so
    callers and tests can use the same names ``HybridMeshConfig`` does without
    going through the long spelling.
    """

    data_parallel_replicate_degree: int = 1
    """
    The `data_parallel_replicate_degree` argument specifies the degree of
    data parallelism for weight replication. When this value is greater
    than 1, weights will be replicated across `data_parallel_replicate_degree`
    ranks. If `data_parallel_shard_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is DDP (Distributed Data Parallelism).
    1 means disabled.
    """

    data_parallel_shard_degree: int = -1
    """
    The `data_parallel_shard_degree` argument specifies the degree of data
    parallelism for weight sharding. When this value is greater than 1, weights
    will be sharded across `data_parallel_shard_degree` ranks. If
    `data_parallel_replicate_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is FSDP (Fully Sharded Data Parallelism).
    -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
    only `data_parallel_shard_degree` can be negative. 1 means disabled.
    """

    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    """
    `reshard_after_forward` specifies the policy for applying
    `reshard_after_forward` within an FSDP setup. `reshard_after_forward`
    controls parameter behavior after forward, trading off memory and
    communication. See torch's `fully_shard` API for more documentation on
    `reshard_after_forward`.

    The supported policies include "default", "always" and "never":

    - "default" applies default resharding behavior, implementing "smart
      defaults" for known optimal scenarios.
    - "always" will enable `reshard_after_forward` for all forward passes.
    - "never" will disable `reshard_after_forward` for all forward passes.
    """

    enable_fsdp_symm_mem: bool = False
    """
    Whether to enable FSDP2 symmetric-memory communication optimizations for
    all FSDP modules after `fully_shard` has been applied.
    """

    tensor_parallel_degree: int = 1
    """Tensor Parallelism degree. 1 means disabled."""

    enable_sequence_parallel: bool = True
    """Whether to use SequenceParallel as part of tensor parallelism. Enabled
    by default."""

    pipeline_parallel_degree: int = 1
    """
    Pipeline Parallelism degree, or number of ranks. 1 means disabled.
    If using looped schedules, this still specifies the number of physical
    ranks, not the number of stages. Stages per rank are inferred from split
    points degree, and schedule.
    """

    module_fqns_per_model_part: list[list[str]] | None = None
    """
    Specify a list of lists containing the FQNs (Fully Qualified Names) of
    modules for each model chunk.
    Each inner list represents one model chunk and contains the module names
    that belong to that chunk.
    e.g. [['tok_embeddings', 'layers.0'], ['layers.1', 'layers.2'],
    ['layers.3', 'layers.4']]
    will create 3 chunks: the first containing tok_embeddings and layers.0,
    the second containing layers.1 and layers.2, and the third containing
    layers.3 and layers.4.
    This provides more explicit control over which modules belong to each chunk
    compared to split points.
    """

    pipeline_parallel_first_stage_less_layers: int = 1
    """
    The number of layers to reduce in the first stage of pipeline parallelism.
    This is because the first stage has the extra overhead of the embedding
    layer, which is not present in the other stages.
    """

    pipeline_parallel_last_stage_less_layers: int = 1
    """
    The number of layers to reduce in the last stage of pipeline parallelism.
    This is because the last stage has the extra overhead of the output layer,
    which is not present in the other stages.
    """

    pipeline_parallel_layers_per_stage: int | None = None
    """
    The number of layers per (virtual) pipeline stage. If specified, the
    module_fqns_per_model_part will be calculated from the number of layers and
    pipeline_parallel_degree. If not specified, the layers per stage will be
    inferred from the model, schedule, and pipeline_parallel_degree.
    """

    pipeline_parallel_schedule: str = "1F1B"
    # Supported schedules (see schedules.py#L2161 for the list):
    # https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py  # noqa: E501
    """
    Specify the Pipeline Parallel schedule to use. The schedule must be
    compatible with the split points and stages_per_rank.
    Looped schedules (e.g. Interleaved1F1B) require specifying
    pipeline_parallel_degree = number of ranks,
    and split_points = number of stages - 1
    """

    pipeline_parallel_schedule_csv: str | None = ""
    """
    Specify the path to the pipeline parallel schedule csv file to use.
    The pipeline_parallel_schedule argument must be either
    PipelineScheduleSingle, PipelineScheduleMulti, or _PipelineScheduleRuntime.
    """

    num_pp_microbatches: int = 1
    """
    Number of pipeline microbatches per data-parallel rank and gradient
    accumulation iteration. This setting is ignored when pipeline parallelism
    is disabled (`pipeline_parallel_degree = 1`, the default).
    """

    context_parallel_degree: int = 1
    """Context parallelism degree. 1 means disabled."""

    context_parallel_strategy: str = "kv_allgather"
    """
    CP attention redistribution strategy. Options:
    - "kv_allgather": all-gather K/V; Q stays token-sharded
    - "ulysses": all-to-all between the token and head shards, so attention
      runs on the full sequence with num_heads / cp heads per rank. Requires
      num_attention_heads and num_key_value_heads divisible by cp, and
      context_parallel_load_balancer=None (the all-to-all reassembles the
      sequence by concatenating rank shards in rank order).
    """

    context_parallel_load_balancer: str | None = "headtail"
    """
    Load balancer type for context parallelism. Options:
    - "headtail": Use HeadTailLoadBalancer for SDPA
    - "ptrr": Use PTRRLoadBalancer for FlexInnerAttention
    - None: Disable load balancing
    """

    context_parallel_ptrr_mask_key: str | None = None
    """
    When the load balancer is "ptrr" and the attention masks are a
    dict[str, BlockMask], this selects which mask in the dict the
    PTRRLoadBalancer is built from. The chosen balancer is then used to shard
    every mask in the dict as well as the inputs. Only relevant for the "ptrr"
    load balancer with dict-valued attention masks; ignored otherwise.
    """

    expert_parallel_degree: int = 1
    """
    Expert parallelism degree. 1 means disabled. No effect for non-MoE models.

    Mesh constraint: the dense region (dp_shard * cp * tp) and sparse region
    (efsdp * ep) cover the same ranks, so dp_shard * cp * tp == efsdp * ep.
    EP borrows ranks from FSDP and TP: efsdp = dp_shard * cp * tp / ep.
    pp and dp_replicate are outer dimensions unaffected by this constraint.
    """

    def non_dp_degrees(self) -> int:
        """Product of the fixed (non-derivable) degrees: dp_replicate*tp*pp*cp*ep."""
        return (
            self.data_parallel_replicate_degree
            * self.tensor_parallel_degree
            * self.pipeline_parallel_degree
            * self.context_parallel_degree
            * self.expert_parallel_degree
        )

    def derive_dp(self, world_size: int) -> int:
        """Resolve ``data_parallel_shard_degree`` against world_size.

        ``-1`` means "derive from world_size": the leftover ranks after
        dp_replicate / tp / pp / cp / ep, matching ``ParallelDims``'s
        interpretation. Mirrors ``ParallelDims._validate``'s divisibility
        check so a mis-sized launch fails here with a config-level message
        rather than deep inside mesh construction.
        """
        fixed = self.non_dp_degrees()
        if self.data_parallel_shard_degree == -1:
            if world_size % fixed != 0:
                raise ValueError(
                    f"world_size={world_size} not divisible by "
                    f"dp_replicate*tp*pp*cp*ep={fixed}"
                )
            return world_size // fixed
        dp_shard = self.data_parallel_shard_degree
        if dp_shard * fixed != world_size:
            raise ValueError(
                f"dp_shard*dp_replicate*tp*pp*cp*ep = {dp_shard * fixed} "
                f"!= world_size={world_size}"
            )
        return dp_shard

    backend: str = "nccl"
    """Distributed backend: nccl (CUDA), gloo (CPU), or hccl (Ascend)."""

    def __post_init__(self):
        for name in (
            "tensor_parallel_degree",
            "pipeline_parallel_degree",
            "context_parallel_degree",
            "expert_parallel_degree",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if (
            self.data_parallel_shard_degree < 1
            and self.data_parallel_shard_degree != -1
        ):
            raise ValueError(
                "data_parallel_shard_degree must be >= 1 or -1 (derive), got "
                f"{self.data_parallel_shard_degree}"
            )
        if self.context_parallel_load_balancer == "":
            raise ValueError(
                "context_parallel_load_balancer cannot be an empty string. "
                "Use None to disable load balancing."
            )
        allowed = frozenset({None, "headtail", "ptrr"})
        if self.context_parallel_load_balancer not in allowed:
            raise ValueError(
                "parallelism.context_parallel_load_balancer must be one of: "
                f"None, 'headtail', 'ptrr' "
                f"(got {self.context_parallel_load_balancer!r})"
            )
        allowed_strategies = frozenset({"kv_allgather", "ulysses"})
        if self.context_parallel_strategy not in allowed_strategies:
            raise ValueError(
                "parallelism.context_parallel_strategy must be one of: "
                f"'kv_allgather', 'ulysses' "
                f"(got {self.context_parallel_strategy!r})"
            )
        if (
            self.context_parallel_strategy == "ulysses"
            and self.context_parallel_load_balancer is not None
        ):
            raise ValueError(
                "parallelism.context_parallel_strategy='ulysses' requires "
                "context_parallel_load_balancer=None: the all-to-all "
                "reassembles the sequence by concatenating rank shards in rank "
                "order, which only the contiguous split satisfies. "
                f"(got {self.context_parallel_load_balancer!r})"
            )
        if self.enable_fsdp_symm_mem and (
            not torch.cuda.is_available()
            or (
                torch.version.hip is None
                and torch.cuda.get_device_capability() < (9, 0)
            )
        ):
            raise ValueError(
                "For NVIDIA GPUs, parallelism.enable_fsdp_symm_mem is only supported "
                "for compute capability 9.0 or newer."
            )
        if self.backend not in {"nccl", "gloo", "hccl"}:
            raise ValueError(
                f"backend must be one of {{nccl, gloo, hccl}}, got {self.backend}"
            )

        # Import lazily so loading configs.py does not pull in pipelining.
        from torch.distributed.pipelining.schedules import get_schedule_class

        try:
            get_schedule_class(self.pipeline_parallel_schedule)
        except ValueError as e:
            raise ValueError(
                "Invalid parallelism.pipeline_parallel_schedule "
                f"{self.pipeline_parallel_schedule!r}: {e}"
            ) from e

    # Short aliases for the torchtitan-spelled degree fields.
    @property
    def dp(self) -> int:
        return self.data_parallel_shard_degree

    @property
    def tp(self) -> int:
        return self.tensor_parallel_degree

    @property
    def pp(self) -> int:
        return self.pipeline_parallel_degree

    @property
    def cp(self) -> int:
        return self.context_parallel_degree

    @property
    def ep(self) -> int:
        return self.expert_parallel_degree


@dataclass(kw_only=True)
class LRSchedulerConfig:
    """The WSD schedule's knobs.

    The runtime side -- the ``LambdaLR`` this builds -- is
    ``hpmesh.components.lr_scheduler.LRScheduler``.

    ``decay_ratio`` is the switch that matters. At its default of 0 there is no
    decay phase, so the factor is 1.0 throughout (after any warmup) and the
    learning rate is exactly the one the optimizer was built with -- which is
    what makes the default run comparable to every measurement taken at a
    constant lr. Setting it to a fraction appends a decay covering that fraction
    of ``total_steps``; whatever is left over after warmup and decay is the
    stable phase, at the peak learning rate.
    """

    warmup_steps: int = field(
        default=0,
        metadata={"help": "Steps to linearly ramp the learning rate from 0."},
    )
    total_steps: int | None = field(
        default=None,
        metadata={
            "help": "Length of the schedule. Defaults to --steps. Set it to "
            "decouple the curve from the run length, so a short debugging run "
            "sees the same lrs the full run would."
        },
    )
    decay_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of total_steps spent decaying the learning rate. "
            "0 (the default) never decays, holding the rate at its peak. A "
            "value below 1 leaves the intervening steps at the peak rate "
            "(WSD)."
        },
    )
    decay_type: Literal["linear", "sqrt", "cosine"] = field(
        default="linear",
        metadata={"help": "Shape of the decay phase. Ignored when decay_ratio=0."},
    )
    min_lr_factor: float = field(
        default=0.0,
        metadata={
            "help": "Floor of the decay, as a fraction of the base learning "
            "rate. 0 decays all the way to zero. Ignored when decay_ratio=0."
        },
    )

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.total_steps is not None and self.total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {self.total_steps}")
        if not 0.0 <= self.decay_ratio <= 1.0:
            raise ValueError(f"decay_ratio must be in [0, 1], got {self.decay_ratio}")
        if not 0.0 <= self.min_lr_factor < 1.0:
            raise ValueError(
                f"min_lr_factor must be in [0, 1), got {self.min_lr_factor}"
            )


@dataclass
class OptimizerConfig:
    """Optimizer (adamw only for now -- the learning path needs just one).

    The schedule lives here rather than in ``TrainingConfig`` because it
    scales this group's learning rate: a lr with no schedule is the degenerate
    case of one, and splitting them would let the two be set independently.
    """

    learning_rate: float = field(default=3e-4, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay"})
    lr_scheduler_config: LRSchedulerConfig = field(
        default_factory=LRSchedulerConfig,
        metadata={"help": "Learning-rate schedule (see components/lr_scheduler)."},
    )

    @property
    def lr_scheduler(self) -> LRSchedulerConfig:
        """The schedule's config.

        A property backed by ``lr_scheduler_config`` so the flat
        ``cfg.lr_scheduler`` spelling works at the call site, while the parser
        still sees a real field to generate flags from. See
        ``TrainingConfig.checkpoint`` for the same shape.
        """
        return self.lr_scheduler_config


@dataclass(kw_only=True)
class CheckpointConfig:
    """What the checkpoint manager keeps, where, and when.

    ``HfArgumentParser`` cannot nest a dataclass behind a named flag -- a nested
    field surfaces as one opaque ``--checkpoint CHECKPOINT`` string -- so this is
    a flat dataclass with every field becoming its own flag (``--enable``,
    ``--interval``, ...). Its ``__post_init__`` therefore runs on the parsed
    values, which is where the cross-field checks belong.

    No ``slots=True``, deliberately: the re-created class breaks the zero-arg
    ``super()`` cell in a subclass's ``__post_init__``. ``ParallelConfig``
    omits it for the same reason.
    """

    enable: bool = False
    """Whether to enable checkpointing."""

    folder: str = "checkpoint"
    """Checkpoint folder, relative to the trainer dump folder."""

    interval: int = 500
    """Checkpointing interval in steps."""

    initial_load_path: str | None = None
    """Optional checkpoint path used when the output checkpoint folder is empty."""

    initial_load_model_only: bool = True
    """Whether an initial checkpoint restores only model state.

    Only consulted on the initial-load path, i.e. when ``initial_load_path``
    names a checkpoint; with no initial checkpoint there is nothing to load
    either way.
    """

    initial_load_in_hf: bool = False
    """Whether the initial checkpoint uses Hugging Face safetensors."""

    initial_load_in_hf_quantized: bool = False
    """Whether the initial Hugging Face checkpoint uses quantized keys."""

    last_save_model_only: bool = True
    """Whether the final checkpoint contains only model state."""

    last_save_in_hf: bool = False
    """Whether the final model-only checkpoint uses Hugging Face safetensors."""

    export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    """Model dtype used by a final model-only checkpoint."""

    keep_latest_k: int = 10
    """Number of recent checkpoints to retain, or zero to retain all."""

    purge_exempt: Callable[[int], bool] | None = None
    """Optional predicate that exempts checkpoint steps from purging."""

    load_step: int = -1
    """Load the checkpoint at the specified step. If -1, load the latest one."""

    exclude_from_loading: list[str] = field(default_factory=list)
    """Non-model state keys excluded from loading."""

    enable_first_step_checkpoint: bool = False
    """Whether to save immediately after the first training step."""

    create_seed_checkpoint: bool = False
    """Whether to initialize and save an unsharded seed checkpoint."""

    load_only: bool = False
    """Whether to permit loads while disabling all saves."""

    async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
    """DCP save mode: synchronous, threaded async, or pinned-memory async.

    Only the DCP backend reads it; the torch_checkpointing one saves
    synchronously. Kept here anyway because it is a checkpointing policy like
    every other field on this class, and splitting one field off into a
    subclass per backend is what this file's single-config design avoids.
    """

    def __post_init__(self) -> None:
        if not self.folder.strip():
            raise ValueError("The 'folder' field cannot be empty.")
        if self.interval < 1:
            raise ValueError("Checkpoint interval needs to be at least 1 step.")
        if self.load_step < -1:
            raise ValueError("load_step must be -1 or non-negative.")
        if self.keep_latest_k < 0:
            raise ValueError("keep_latest_k cannot be negative.")
        if self.keep_latest_k == 1:
            raise ValueError(
                "We need to maintain at least 2 checkpoint replicas, "
                "as the last one may be in the process of being saved."
            )
        if MODEL in self.exclude_from_loading:
            raise ValueError(f"{MODEL} key shouldn't be in exclude_from_loading.")
        if (
            OPTIMIZER in self.exclude_from_loading
            and LR_SCHEDULER not in self.exclude_from_loading
        ):
            raise ValueError(
                f"{LR_SCHEDULER} must be excluded when {OPTIMIZER} is excluded."
            )

        if self.initial_load_path:
            self.initial_load_path = self.initial_load_path.strip()
            if not (
                self.initial_load_path.startswith("/")
                or filesystem.is_remote(self.initial_load_path)
            ):
                raise ValueError(
                    "initial_load_path must be an absolute path or a remote "
                    f"URI (e.g. gs://...): {self.initial_load_path}"
                )
        if self.initial_load_in_hf and not self.initial_load_model_only:
            raise ValueError("initial_load_in_hf requires initial_load_model_only.")
        if self.initial_load_in_hf_quantized and not (
            self.initial_load_in_hf and self.initial_load_path
        ):
            raise ValueError(
                "initial_load_in_hf_quantized requires initial_load_in_hf "
                "and initial_load_path."
            )
        if self.last_save_in_hf and not self.last_save_model_only:
            raise ValueError("last_save_in_hf requires last_save_model_only=True.")

        async_lowered = self.async_mode.lower()
        if async_lowered not in ("disabled", "async", "async_with_pinned_mem"):
            raise ValueError(f"Invalid async_mode: {async_lowered}")
        self.async_mode = async_lowered

        # Remote (fsspec) checkpoint IO supports only the native DCP format. HF
        # safetensors read/write to a remote URI is not implemented, so reject
        # the combination up front instead of failing deep inside DCP.
        if self.last_save_in_hf and filesystem.is_remote(self.folder):
            raise ValueError(
                "last_save_in_hf is not supported with a remote "
                f"checkpoint.folder: {self.folder}"
            )
        if (
            self.initial_load_in_hf
            and self.initial_load_path
            and filesystem.is_remote(self.initial_load_path)
        ):
            raise ValueError(
                "initial_load_in_hf is not supported with a remote "
                f"initial_load_path: {self.initial_load_path}"
            )

        if self.load_only and self.enable_first_step_checkpoint:
            logger.warning(
                "checkpoint.load_only is True; enable_first_step_checkpoint "
                "will be ignored."
            )
        # Note torchtitan's sibling warning for ``initial_load_model_only``
        # without an ``initial_load_path`` is deliberately not ported: hpmesh
        # builds a default Config on every run (``--help`` included), so that
        # warning would fire on runs that never load anything.


@dataclass(kw_only=True)
class MetricsConfig:
    """What the metrics processor reports, and where.

    ``log_freq`` and ``tag`` are this module's own: the first is a CLI concern
    (the processor reads its window length from here), the second names a logging
    key. Both live with the config rather than the processor because the config
    is what the command line describes.
    """

    log_freq: int = 1
    """Console log frequency, in steps. Also the TensorBoard/WandB frequency --
    one window feeds both."""

    enable_tensorboard: bool = False
    """Whether to write TensorBoard event files."""

    disable_color_printing: bool = False
    """Whether to drop colour from the console line."""

    save_tb_folder: str = "tb"
    """TensorBoard folder, relative to the dump folder."""

    save_for_all_ranks: bool = False
    """Whether every rank logs, rather than only the metrics rank."""

    enable_wandb: bool = False
    """Whether to stream metrics to Weights & Biases."""

    tag: str | None = None
    """Prefix applied to every recorded key in TensorBoard/WandB. The console
    line is not prefixed: it is read live, never merged."""

    def __post_init__(self) -> None:
        if self.log_freq <= 0:
            raise ValueError("metrics.log_freq must be greater than 0.")


@dataclass(kw_only=True)
class ProfilerConfig:
    """What the profiler collects, and when."""

    enable_profiling: bool = False
    """Whether to collect Kineto traces."""

    save_traces_folder: str = "profiling/traces"
    """Trace location, relative to the base folder."""

    profile_freq: int = 10
    """How often to collect a trace, in iterations."""

    profiler_repeat: int | None = None
    """How many times to repeat the profiling cycle. ``None`` repeats forever,
    which is ``torch.profiler.schedule``'s own default."""

    profiler_skip_first: int | None = None
    """How many iterations to skip before the schedule starts."""

    profiler_skip_first_wait: int | None = None
    """How many waits to skip at the start of the first cycle."""

    profiler_active: int = 1
    """Iterations the profiler is active for, per cycle."""

    profiler_warmup: int = 3
    """Warmup iterations before the active ones in each cycle.

    Warmup discards its results, so it is what keeps the first active iteration
    from being dominated by lazy initialization.
    """

    enable_memory_snapshot: bool = False
    """Whether to write allocator memory snapshots."""

    memory_snapshot_freq: int | None = None
    """Snapshot frequency, in iterations. Defaults to ``profile_freq``."""

    save_memory_snapshot_folder: str = "profiling/memory_snapshot"
    """Snapshot location, relative to the base folder.

    Spelled out rather than derived from ``save_traces_folder``: the two are
    independent knobs, and a derived default would silently follow a custom
    trace folder into an unexpected place.
    """

    memory_snapshot_max_entries: int = 1_000_000
    """Alloc/free events kept per snapshot.

    The allocator history is a ring buffer, so this bounds how far back a
    snapshot can see, and with it the dump's size and cost.
    """

    def __post_init__(self) -> None:
        if self.enable_profiling and self.profile_freq < (
            self.profiler_warmup + self.profiler_active
        ):
            raise ValueError(
                "profiler.profile_freq must be greater than or equal to "
                "profiler_warmup + profiler_active."
            )


# HfArgumentParser cannot turn a nested dataclass into a set of flags -- it
# collapses it to one opaque `--<field>` argument and never consults the fields
# inside. The `<x>_config` field is that escape hatch: the config's own group is
# parsed separately in train.py and grafted back on here, which is also where
# its __post_init__ re-runs against the parsed values.
#
# That is what keeps every class in this file named `*Config`. Renaming one is
# not cosmetic -- HfArgumentParser takes the flag stem from the FIELD, so the
# field names below are the CLI: `metrics_config` is what makes this class's
# fields reach the user as `--log_freq` and `--enable_tensorboard`. A class
# named `MetricsArguments` with a field `metrics_arguments` would move every one
# of them behind a `--metrics_arguments` prefix, which is a breaking change.
#
# A plain `checkpoint` field would be nicer to read, but a dataclass field and
# the class it types cannot share a name.


@dataclass(kw_only=True)
class DataloaderConfig:
    """Where the micro-batches come from.

    ``random`` (the default) keeps the synthetic corpus and needs no assets, so
    the default run is unchanged and reproducible offline. Any other value
    names a recipe from ``datasets.hf.text.DATASETS``, or the built-in
    ``local_jsonl`` -- which is deliberately NOT in that dict, because its
    corpus path is a runtime argument rather than a constant.

    Every non-``random`` dataset builds its graph on Grain, which needs a
    tokenizer, so ``tokenizer_path`` is required there and unused otherwise.
    """

    dataset: str = field(
        default="random",
        metadata={
            "help": "Corpus selector: 'random' (synthetic, no assets) | "
            "'local_jsonl' | a key of datasets.hf.text.DATASETS"
        },
    )
    tokenizer_path: str | None = field(
        default=None,
        metadata={
            "help": "Directory holding the tokenizer. Required unless --dataset random."
        },
    )
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Corpus path. Required for --dataset local_jsonl."},
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Globally shuffle before sharding across DP ranks"},
    )
    streaming_shuffle_buffer_size: int = field(
        default=1_000,
        metadata={"help": "Streaming rows retained per rank for approximate shuffle"},
    )
    num_prefetch_batches: int = field(
        default=2,
        metadata={"help": "Collated batches queued per rank for the trainer"},
    )
    max_num_documents: int | None = field(
        default=None,
        metadata={
            "help": "Cap on documents packed into one row. None leaves the "
            "frontier unconstrained."
        },
    )

    def __post_init__(self) -> None:
        if self.dataset != "random" and not self.tokenizer_path:
            raise ValueError(
                f"tokenizer_path is required for dataset '{self.dataset}'. "
                "Only 'random' runs without a tokenizer."
            )
        if self.dataset == "local_jsonl" and not self.dataset_path:
            raise ValueError("dataset_path is required for dataset 'local_jsonl'")
        if self.dataset not in ("random", "local_jsonl") and (
            self.dataset not in DATASETS
        ):
            raise ValueError(
                f"unknown dataset {self.dataset!r}. Expected 'random', "
                f"'local_jsonl', or one of: {sorted(DATASETS)}"
            )
        if self.max_num_documents is not None and self.max_num_documents <= 0:
            raise ValueError("max_num_documents must be positive")


@dataclass
class TrainingConfig:
    """Training loop hyperparameters and reproducibility."""

    global_batch_size: int = field(
        default=8, metadata={"help": "Sequences per step across ALL DP ranks"}
    )
    max_seq_len: int = field(default=64, metadata={"help": "Sequence length"})
    steps: int = field(default=20, metadata={"help": "Number of optimizer steps"})
    seed: int = field(default=42, metadata={"help": "Base RNG seed"})
    compile: bool = field(default=False, metadata={"help": "torch.compile the model"})
    deterministic: bool = field(
        default=True,
        metadata={
            "help": "Deterministic algorithms -- required for bit-exact comparison"
        },
    )
    max_norm: float = field(
        default=1.0,
        metadata={
            "help": "Gradient-norm clip threshold. A non-positive value disables "
            "clipping but still reports grad_norm."
        },
    )
    dump_folder: str = field(
        default="./outputs",
        metadata={
            "help": "Root directory for this run's outputs. The checkpoint, "
            "TensorBoard and profiling folders are resolved against it."
        },
    )
    checkpoint_config: CheckpointConfig = field(
        default_factory=CheckpointConfig,
        metadata={"help": "Checkpointing (see components/checkpointer)."},
    )
    dataloader_config: DataloaderConfig = field(
        default_factory=DataloaderConfig,
        metadata={"help": "Micro-batch source (see datasets/)."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "Metrics reporting (see components/metrics)."},
    )
    profiler_config: ProfilerConfig = field(
        default_factory=ProfilerConfig,
        metadata={"help": "Profiling (see components/profiler)."},
    )

    @property
    def checkpoint(self) -> CheckpointConfig:
        """The checkpoint manager's config.

        A property backed by ``checkpoint_config`` rather than a field of its own
        so the flat ``cfg.checkpoint`` spelling works at the call site. A
        property is not a dataclass field, so the parser never turns it into a
        flag.
        """
        return self.checkpoint_config

    @property
    def metrics(self) -> MetricsConfig:
        """The metrics processor's config. See ``checkpoint`` for the shape."""
        return self.metrics_config

    @property
    def dataloader(self) -> DataloaderConfig:
        """The micro-batch source's config. See ``checkpoint`` for the shape."""
        return self.dataloader_config

    @property
    def profiler(self) -> ProfilerConfig:
        """The profiler's config. See ``checkpoint`` for the shape."""
        return self.profiler_config

    def __post_init__(self) -> None:
        if self.global_batch_size < 1:
            raise ValueError(
                f"global_batch_size must be >= 1, got {self.global_batch_size}"
            )
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")


@dataclass
class HybridMeshConfig:
    """Single entry point: composes the argument groups (no multiple inheritance).

    Each group validates itself in its own __post_init__ (run by default_factory).
    Nested groups (``parallel``, ``training``) are reachable both as themselves and
    through the flat property view below.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        # Cross-group check: CP must divide the sequence length.
        if self.training.max_seq_len % self.parallel.cp != 0:
            raise ValueError(
                f"max_seq_len ({self.training.max_seq_len}) must be divisible by "
                f"cp ({self.parallel.cp})"
            )

    # -- Flat view: lets mesh/trainer read cfg.dp / cfg.lr / ... uniformly. --
    @property
    def hf_model(self) -> str:
        return self.model.model_name_or_path

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.model.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.model.intermediate_size

    @property
    def num_hidden_layers(self) -> int:
        return self.model.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.model.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.model.num_key_value_heads

    @property
    def dp(self) -> int:
        return self.parallel.data_parallel_shard_degree

    @property
    def tp(self) -> int:
        return self.parallel.tensor_parallel_degree

    @property
    def pp(self) -> int:
        return self.parallel.pipeline_parallel_degree

    @property
    def cp(self) -> int:
        return self.parallel.context_parallel_degree

    @property
    def ep(self) -> int:
        return self.parallel.expert_parallel_degree

    @property
    def lr(self) -> float:
        return self.optimizer.learning_rate

    @property
    def weight_decay(self) -> float:
        return self.optimizer.weight_decay

    @property
    def lr_scheduler_config(self) -> LRSchedulerConfig:
        return self.optimizer.lr_scheduler_config

    @property
    def global_batch_size(self) -> int:
        return self.training.global_batch_size

    @property
    def max_seq_len(self) -> int:
        return self.training.max_seq_len

    @property
    def steps(self) -> int:
        return self.training.steps

    @property
    def seed(self) -> int:
        return self.training.seed

    @property
    def compile(self) -> bool:
        return self.training.compile

    @property
    def deterministic(self) -> bool:
        return self.training.deterministic

    @property
    def pipeline_parallel_schedule(self) -> str:
        return self.parallel.pipeline_parallel_schedule

    @property
    def log_freq(self) -> int:
        # Owned by the metrics group: the same number gates both the console
        # line and the TensorBoard/WandB write, so one knob controls them.
        return self.training.metrics.log_freq

    @property
    def max_norm(self) -> float:
        return self.training.max_norm

    @property
    def dump_folder(self) -> str:
        return self.training.dump_folder

    @property
    def checkpoint(self) -> CheckpointConfig:
        return self.training.checkpoint

    @property
    def metrics(self) -> MetricsConfig:
        return self.training.metrics

    @property
    def dataloader(self) -> DataloaderConfig:
        return self.training.dataloader

    @property
    def profiler(self) -> ProfilerConfig:
        return self.training.profiler

    def derive_dp(self, world_size: int) -> int:
        """Flat passthrough so callers use cfg.derive_dp(world_size) uniformly."""
        return self.parallel.derive_dp(world_size)

    def auto_fill_model(self) -> None:
        """Fill model arch fields from a HF hub config when given a hub id.

        Called explicitly (not in __post_init__) so offline runs never hit the network.
        """
        mp = self.model.model_name_or_path
        if mp.count("/") != 1:
            return  # offline architecture name; explicit sizes are authoritative
        try:
            hf_config = AutoConfig.from_pretrained(mp)
        except Exception as e:  # noqa: BLE001 - warn, keep explicit values
            logger.warning("Could not load AutoConfig for '%s': %s", mp, e)
            return
        m = self.model
        m.vocab_size = getattr(hf_config, "vocab_size", m.vocab_size)
        m.hidden_size = getattr(hf_config, "hidden_size", m.hidden_size)
        m.intermediate_size = getattr(
            hf_config, "intermediate_size", m.intermediate_size
        )
        m.num_hidden_layers = getattr(
            hf_config, "num_hidden_layers", m.num_hidden_layers
        )
        m.num_attention_heads = getattr(
            hf_config, "num_attention_heads", m.num_attention_heads
        )
        m.num_key_value_heads = getattr(
            hf_config, "num_key_value_heads", m.num_key_value_heads
        )
