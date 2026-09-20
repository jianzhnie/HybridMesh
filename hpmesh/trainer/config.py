"""hpmesh configuration: grouped dataclasses composed into one HybridMeshConfig.

Design notes (see README): grouped by concern (model / parallel / optimizer /
training), then COMPOSED -- not mixed in via multiple inheritance -- so each
group's __post_init__ validation runs automatically via default_factory, with no
fragile manual chaining. The single entry point is ``HybridMeshConfig``; CLI/YAML
parsing is done by ``transformers.HfArgumentParser`` in ``hpmesh.train``.

Kept deliberately small for the learning path: one optimizer (adamw), one LR
schedule, deterministic seeding. Add knobs only when a learning step needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from transformers import AutoConfig

from hpmesh.components.checkpointer.dcp import CheckpointManager
from hpmesh.components.metrics import MetricsProcessor
from hpmesh.components.profiler import Profiler
from hpmesh.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "CheckpointArguments",
    "HybridMeshConfig",
    "MetricsArguments",
    "ModelArguments",
    "OptimizerArguments",
    "ParallelArguments",
    "ProfilerArguments",
    "TrainingArguments",
]


@dataclass
class ModelArguments:
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
class ParallelismConfig:
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
        # Import lazily so loading configs.py does not pull in pipelining.
        from torch.distributed.pipelining.schedules import get_schedule_class

        try:
            get_schedule_class(self.pipeline_parallel_schedule)
        except ValueError as e:
            raise ValueError(
                "Invalid parallelism.pipeline_parallel_schedule "
                f"{self.pipeline_parallel_schedule!r}: {e}"
            ) from e


@dataclass(kw_only=True, slots=True)
class ParallelArguments(ParallelismConfig):
    """CLI-facing view of ``ParallelismConfig``.

    Subclasses the shared config (rather than re-declaring the same degrees) so
    hpmesh has ONE source of truth for parallelism: ``ParallelDims.from_config``
    reads the inherited ``*_degree`` fields directly, which is what the torchtitan
    mesh builder expects. On top of that the trainer CLI needs a distributed
    ``backend``.

    Short aliases (``tp`` / ``pp`` / ``cp`` / ``ep`` / ``dp``) are exposed as
    properties so callers and tests can use the same names as ``HybridMeshConfig``
    without going through the long ``*_degree`` spelling.
    """

    backend: str = "nccl"
    """Distributed backend: nccl (CUDA), gloo (CPU), or hccl (Ascend)."""

    def __post_init__(self) -> None:
        # Explicit two-arg super(): ``slots=True`` rebuilds the class, so the
        # zero-arg form's ``__class__`` cell would point at the discarded one.
        super(ParallelArguments, self).__post_init__()
        if self.backend not in {"nccl", "gloo", "hccl"}:
            raise ValueError(
                f"backend must be one of {{nccl, gloo, hccl}}, got {self.backend}"
            )

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


@dataclass
class OptimizerArguments:
    """Optimizer (adamw only for now -- the learning path needs just one)."""

    learning_rate: float = field(default=3e-4, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay"})


@dataclass(kw_only=True)
class CheckpointArguments(CheckpointManager.Config):
    """CLI view of the checkpoint config.

    Subclasses the manager's own ``Config`` (rather than re-declaring the same
    fields) so hpmesh has ONE source of truth for checkpointing: the manager
    reads the inherited fields directly, exactly as ``ParallelDims.from_config``
    reads the inherited ``*_degree`` fields off ``ParallelArguments``.

    Why an empty subclass is needed at all: ``HfArgumentParser`` cannot nest a
    dataclass group behind a named flag -- a nested field surfaces as a single
    opaque ``--checkpoint CHECKPOINT`` string argument. Being its own dataclass
    gives the parser the flat field list to generate real flags from
    (``--checkpoint_folder``, ``--checkpoint_interval``, ...). It also runs
    ``Config.__post_init__`` validation on the parsed values.

    The defaults are inherited, not restated, so a change to the manager's
    default reaches the CLI.

    No ``slots=True`` (the manager's ``Config`` omits it too): rebuilding the
    class would break the zero-arg ``super()`` in ``Config.__post_init__`` once
    this is its subclass, and the fields are config, not a hot inner loop.
    """


@dataclass(kw_only=True)
class MetricsArguments(MetricsProcessor.Config):
    """CLI view of the metrics config.

    Built like ``CheckpointArguments``: subclass the component's own ``Config``
    so the defaults and the validation live in one place, and let ``train.py``
    parse it as its own group rather than as a nested field.
    """

    log_freq: int = field(
        default=1,
        metadata={
            "help": "Console log frequency, in steps. Also the TensorBoard/WandB "
            "frequency -- one window feeds both."
        },
    )
    tag: str | None = field(
        default=None,
        metadata={
            "help": "Prefix applied to every recorded key in TensorBoard/WandB. The "
            "console line is not prefixed: it is read live, never merged."
        },
    )

    def __post_init__(self) -> None:
        # Explicit two-arg super(): see ParallelArguments.
        super().__post_init__()


@dataclass(kw_only=True)
class ProfilerArguments(Profiler.Config):
    """CLI view of the profiler config.

    Built like ``CheckpointArguments`` and ``MetricsArguments``: subclass the
    component's own ``Config`` so the defaults and the validation live in one
    place, and let ``train.py`` parse it as its own group.
    """

    def __post_init__(self) -> None:
        # Explicit two-arg super(): see ParallelArguments.
        super().__post_init__()


# HfArgumentParser cannot turn a nested dataclass into a set of flags -- it
# collapses it to one opaque `--<field>` argument and never consults the fields
# inside. The `<x>_config` field is that escape hatch, and each config's own
# group is parsed separately in train.py and grafted back on. Naming the field
# for the dataclass it holds keeps the parser's `--help` from advertising a
# value nobody should set. A plain `checkpoint` field would be nicer to read,
# but a dataclass field and the class it types cannot share a name.


@dataclass
class TrainingArguments:
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
    checkpoint_config: CheckpointArguments = field(
        default_factory=CheckpointArguments,
        metadata={"help": "Checkpointing (see components/checkpointer)."},
    )
    metrics_config: MetricsArguments = field(
        default_factory=MetricsArguments,
        metadata={"help": "Metrics reporting (see components/metrics)."},
    )
    profiler_config: ProfilerArguments = field(
        default_factory=ProfilerArguments,
        metadata={"help": "Profiling (see components/profiler)."},
    )

    @property
    def checkpoint(self) -> CheckpointArguments:
        """The checkpoint manager's config.

        A property backed by ``checkpoint_config`` rather than a field of its own
        so the flat ``cfg.checkpoint`` spelling works at the call site. A
        property is not a dataclass field, so the parser never turns it into a
        flag.
        """
        return self.checkpoint_config

    @property
    def metrics(self) -> MetricsArguments:
        """The metrics processor's config. See ``checkpoint`` for the shape."""
        return self.metrics_config

    @property
    def profiler(self) -> ProfilerArguments:
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

    model: ModelArguments = field(default_factory=ModelArguments)
    parallel: ParallelArguments = field(default_factory=ParallelArguments)
    optimizer: OptimizerArguments = field(default_factory=OptimizerArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)

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
    def checkpoint(self) -> CheckpointArguments:
        return self.training.checkpoint

    @property
    def metrics(self) -> MetricsArguments:
        return self.training.metrics

    @property
    def profiler(self) -> ProfilerArguments:
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
