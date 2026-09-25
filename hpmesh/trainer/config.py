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
``components/optimizer/lr_scheduler.py`` -- for two reasons. A config that
carries its own builder suggests the built object is one of its fields, and it
is not: building takes arguments the config does not have (which rank am I, how
many tokens per batch). And a builder has to name the type it produces, which
for a config nested under a component would mean importing the component into
this module while the component imports this one back.

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

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from transformers import AutoConfig

from hpmesh.utils import filesystem
from hpmesh.utils.checkpoint_keys import LR_SCHEDULER, MODEL, OPTIMIZER
from hpmesh.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "CheckpointConfig",
    "CompileConfig",
    "DataloaderConfig",
    "EMAConfig",
    "ValidationConfig",
    "HybridMeshConfig",
    "LRSchedulerConfig",
    "MetricsConfig",
    "ModelConfig",
    "OptimizerConfig",
    "ParallelConfig",
    "ParamGroupConfig",
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
    experts_implementation: str = field(
        default="native",
        metadata={
            "help": "HF experts forward kernel for MoE models: 'native' keeps "
            "the model's built-in kernel; 'grouped_mm' / 'batched_mm' / 'eager' "
            "require a model with a settable experts implementation and raise "
            "otherwise (never silently substituted). Irrelevant under EP>1, "
            "where the swap replaces the whole MoE block."
        },
    )
    compute_dtype: str | None = field(
        default=None,
        metadata={
            "help": "Forward matmul dtype for the lm_head only: 'float32', "
            "'float16', or 'bfloat16'. None (default) leaves the output "
            "projection in the model's own dtype. The stored weight keeps its "
            "dtype -- the cast happens inside the forward, so logits are "
            "scored at the requested precision without an upcast copy of the "
            "weight living in the optimizer."
        },
    )
    arch_overrides: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Extra architecture settings, by name, for fields this "
            "config does not name. The six fields above cover every dense "
            "decoder; anything else a model needs -- an MoE's "
            "n_routed_experts / num_experts_per_tok, MLA's q_lora_rank -- has "
            "no field here. Offline only: they are applied on top of the six "
            "above when building the HF config, and ignored for a hub id or a "
            "local checkpoint directory, where the saved config wins. Supplied "
            "programmatically (a dict does not survive the CLI parser)."
        },
    )


@dataclass(kw_only=True, slots=True)
class ParallelConfig:
    """The parallelism sizes.

    The field names and semantics are torchtitan's, spelled ``*_size`` rather
    than ``*_degree``. ``ParallelDims.from_config`` reads these six fields by
    name, so the spelling here and there has to stay in step. Short aliases
    (``tp`` / ``pp`` / ``cp`` / ``ep`` / ``dp``) are exposed as properties at
    the end of the class, so callers and tests can use the same names
    ``HybridMeshConfig`` does without going through the long spelling.
    """

    data_parallel_replicate_size: int = 1
    """
    The `data_parallel_replicate_size` argument specifies the degree of
    data parallelism for weight replication. When this value is greater
    than 1, weights will be replicated across `data_parallel_replicate_size`
    ranks. If `data_parallel_shard_size` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is DDP (Distributed Data Parallelism).
    1 means disabled.
    """

    data_parallel_shard_size: int = -1
    """
    The `data_parallel_shard_size` argument specifies the degree of data
    parallelism for weight sharding. When this value is greater than 1, weights
    will be sharded across `data_parallel_shard_size` ranks. If
    `data_parallel_replicate_size` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is FSDP (Fully Sharded Data Parallelism).
    -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
    only `data_parallel_shard_size` can be negative. 1 means disabled.
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
    FSDP modules after `fully_shard` has been applied.
    """

    fsdp_symm_mem_scope: Literal["all", "dense"] = "all"
    """
    Which FSDP modules get symmetric-memory comms when
    ``enable_fsdp_symm_mem`` is on: "all" covers every FSDP module; "dense"
    skips modules flagged ``moe_enabled`` (an MoE transformer block is one
    FSDP module, so its attention parameters are skipped along with its
    experts). Symmetric memory is not always beneficial for the expert
    (sparse) FSDP modules, hence the narrower option. Ignored when
    ``enable_fsdp_symm_mem`` is False.
    """

    tensor_parallel_size: int = 1
    """Tensor Parallelism degree. 1 means disabled."""

    enable_sequence_parallel: bool = True
    """Sequence parallelism as part of tensor parallelism.

    Not a switch here. hpmesh's TP is the sequence-parallel formulation only:
    ``parallel/tensor_parallel/tp.py`` gathers and scatters the sequence inside
    the fused GEMMs, and the trainer shards the batch so ``T / tp`` tokens
    enter each rank. There is no replicated-activation realization to fall
    back to, so ``False`` has nothing to select and is rejected rather than
    silently ignored -- see ``ParallelConfig.__post_init__``.

    Unlike the other fields here this one is hpmesh's own: it is not forwarded
    to any upstream API, so it is not a name this config has to keep in step.
    """

    pipeline_parallel_size: int = 1
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
    pipeline_parallel_size. If not specified, the layers per stage will be
    inferred from the model, schedule, and pipeline_parallel_size.
    """

    pipeline_parallel_schedule: str = "1F1B"
    # Supported schedules (see schedules.py#L2161 for the list):
    # https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py  # noqa: E501
    """
    Specify the Pipeline Parallel schedule to use. The schedule must be
    compatible with the split points and stages_per_rank.
    Looped schedules (e.g. Interleaved1F1B) require specifying
    pipeline_parallel_size = number of ranks,
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
    is disabled (`pipeline_parallel_size = 1`, the default).
    """

    context_parallel_size: int = 1
    """Context parallelism degree. 1 means disabled."""

    context_parallel_strategy: str = "kv_allgather"
    """
    CP attention redistribution strategy. Options:
    - "kv_allgather": all-gather K/V; Q stays token-sharded
    - "ulysses": all-to-all between the token and head shards, so attention
      runs on the full sequence with num_heads / cp heads per rank. Requires
      num_attention_heads and num_key_value_heads divisible by cp, and
      context_parallel_load_balancer=None: every rank attends the full
      sequence in whatever order the all-to-all delivers, and flex cannot be
      handed a mask over a different order than the tokens it is attending.
    """

    context_parallel_load_balancer: str | None = "headtail"
    """
    Load balancer type for context parallelism. Options:
    - "headtail": Use HeadTailLoadBalancer for SDPA
    - "ptrr": accepted for config compatibility, but NOT implemented -- it is
      rejected with NotImplementedError when the CP mesh is built
      (``parallel/context_parallel/input_shard.py``), because it needs a
      BlockMask to derive its schedule from and hpmesh's kernel does not
      consume one. Use "headtail" or None.
    - None: Disable load balancing
    """
    expert_parallel_size: int = 1
    """
    Expert parallelism degree. 1 means disabled. No effect for non-MoE models.

    Mesh constraint: the dense region (dp_shard * cp * tp) and sparse region
    (efsdp * ep) cover the same ranks, so dp_shard * cp * tp == efsdp * ep.
    EP borrows ranks from FSDP and TP: efsdp = dp_shard * cp * tp / ep.
    pp and dp_replicate are outer dimensions unaffected by this constraint.

    Not composable with tensor_parallel_size > 1 (rejected in
    ``__post_init__``): MoE-under-TP shards each expert's hidden dim, EP
    shards the expert count, and the two-dimensional combination is not
    implemented.
    """

    router_aux_loss_coef: float | None = None
    """
    Coefficient of the per-forward MoE load-balance loss, for HF models whose
    config does not carry one.

    The swap reads ``router_aux_loss_coef`` off the HF config when it is there
    (Qwen3Moe has it). DeepSeek-V3 does not -- its config has no aux-loss field
    at all -- so without this the balance loss its design calls for
    (DeepSeek-V3 Sec 2.1.2, the sequence-wise complementary loss) would never be
    instantiated. ``None`` keeps the HF config's value, or no loss when it has
    none; setting it overrides for every MoE layer.
    """

    moe_quantile_balancing: bool = False
    """
    Replace the sign-based load-balancing bias with quantile-balanced routing
    (Kimi K3 Sec 2.3.3): the router observes a biased Top-(K+1) cutoff per
    token, accumulates a required-bias histogram, and an optimizer pre-hook
    re-solves ``expert_bias_E`` as the ``top_k / num_experts`` quantile once
    per step. Mutually exclusive with the sign-based update -- the swap forces
    ``load_balance_coeff`` off. Requires sigmoid router scores, no
    group-limited routing, and ep > 1 (the swap is what installs the router).
    """

    def non_dp_sizes(self) -> int:
        """Product of fixed world-mesh degrees: dp_replicate*tp*pp*cp.

        EP is not an additional world dimension. It tiles the existing
        ``dp_shard * cp * tp`` sparse region, matching ``ParallelDims`` and
        TorchTitan's mesh algebra.
        """
        return (
            self.data_parallel_replicate_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.context_parallel_size
        )

    def derive_dp(self, world_size: int) -> int:
        """Resolve ``data_parallel_shard_size`` against world_size.

        ``-1`` means "derive from world_size": the leftover ranks after
        dp_replicate / tp / pp / cp / ep, matching ``ParallelDims``'s
        interpretation. Mirrors ``ParallelDims._validate``'s divisibility
        check so a mis-sized launch fails here with a config-level message
        rather than deep inside mesh construction.
        """
        fixed = self.non_dp_sizes()
        if self.data_parallel_shard_size == -1:
            if world_size % fixed != 0:
                raise ValueError(
                    f"world_size={world_size} not divisible by "
                    f"dp_replicate*tp*pp*cp={fixed}"
                )
            return world_size // fixed
        dp_shard = self.data_parallel_shard_size
        if dp_shard * fixed != world_size:
            raise ValueError(
                f"dp_shard*dp_replicate*tp*pp*cp = {dp_shard * fixed} "
                f"!= world_size={world_size}"
            )
        return dp_shard

    train_timeout_seconds: int = 100
    """Timeout, in seconds, applied to every process group once training starts.

    The process groups are created with a deliberately long timeout, because
    startup -- model build, the first collective, compile -- is what actually
    takes minutes on a large run. Left at that value, a later hang is
    indistinguishable from a slow start: the job waits out the startup timeout,
    which can be half an hour. The trainer therefore lowers every group's
    timeout to this after the first completed train step, at which point the
    startup work is known to be behind it.

    Default matches torchtitan's ``comm.train_timeout_seconds``.
    """

    def __post_init__(self):
        if self.train_timeout_seconds <= 0:
            raise ValueError(
                "train_timeout_seconds must be greater than 0, got "
                f"{self.train_timeout_seconds}"
            )
        for name in (
            "data_parallel_replicate_size",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "context_parallel_size",
            "expert_parallel_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if not self.enable_sequence_parallel:
            # The flag used to be read by no line of the package, so ``False``
            # was accepted and silently behaved as ``True``. It cannot simply be
            # honoured instead: hpmesh has one TP realization, and it *is* the
            # sequence-parallel one (see the field's docstring). Refuse the
            # configuration rather than train something other than what was
            # asked for.
            raise NotImplementedError(
                "parallelism.enable_sequence_parallel=false is not supported: "
                "hpmesh's tensor parallelism is sequence-parallel by "
                "construction (the fused TP GEMMs gather/scatter the sequence "
                "and the batch is sharded T/tp). There is no "
                "replicated-activation TP path to fall back to, so this flag "
                "has nothing to disable. Leave it true, or set "
                "tensor_parallel_size=1 to drop TP."
            )
        if self.tensor_parallel_size > 1 and self.expert_parallel_size > 1:
            # The two MoE strategies do not compose yet: apply_tp shards each
            # expert's hidden dim F across the TP group (tensor_parallel/tp.py,
            # MoE-under-TP), while the EP swap slices the expert count E across
            # the EP group and installs all-to-all dispatch. Combining them is
            # a two-dimensional expert sharding (upstream's EDP-style mesh)
            # that neither path is written or verified against, and apply_tp
            # runs BEFORE apply_ep, so the swap would either see TP-sharded
            # weights it cannot slice correctly or find no HF block at all.
            # Refuse at the config, before any mesh is built.
            raise NotImplementedError(
                "tensor_parallel_size > 1 with expert_parallel_size > 1 is not "
                "supported: MoE-under-TP (experts sharded on the F dim) and EP "
                "(experts sharded on the E count) do not compose in hpmesh. "
                "Run MoE models with tp>1 and ep=1 (MoE-under-TP) or ep>1 and "
                "tp=1 (EP)."
            )
        if self.data_parallel_shard_size < 1 and self.data_parallel_shard_size != -1:
            raise ValueError(
                "data_parallel_shard_size must be >= 1 or -1 (derive), got "
                f"{self.data_parallel_shard_size}"
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
        if self.context_parallel_load_balancer == "ptrr":
            # It is in `allowed` only so that a config naming it is recognized
            # rather than reported as a typo. Rejecting it here rather than in
            # the mesh builder moves the failure from the first forward -- after
            # process groups, model build and dataloader construction -- to the
            # config, where the message is still about the flag.
            raise NotImplementedError(
                "parallelism.context_parallel_load_balancer='ptrr' is not "
                "implemented in hpmesh: it derives its schedule from a "
                "BlockMask, which hpmesh's CP kernel does not consume. Use "
                "'headtail' or None."
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
                "context_parallel_load_balancer=None: every rank attends the "
                "full sequence in whatever order the all-to-all delivers, and "
                "a load balancer's rearrangement would make that a permuted "
                "corpus. Nothing raises: the attention is over the wrong "
                "order of the right tokens, so the loss stays finite and the "
                "run trains a different model. "
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
        if self.fsdp_symm_mem_scope not in ("all", "dense"):
            raise ValueError(
                "parallelism.fsdp_symm_mem_scope must be one of: 'all', 'dense' "
                f"(got {self.fsdp_symm_mem_scope!r})"
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
        return self.data_parallel_shard_size

    @property
    def tp(self) -> int:
        return self.tensor_parallel_size

    @property
    def pp(self) -> int:
        return self.pipeline_parallel_size

    @property
    def cp(self) -> int:
        return self.context_parallel_size

    @property
    def ep(self) -> int:
        return self.expert_parallel_size


@dataclass(kw_only=True)
class LRSchedulerConfig:
    """The WSD schedule's knobs.

    The runtime side -- the ``LambdaLR`` this builds -- is
    ``hpmesh.components.optimizer.LRSchedulersContainer``.

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
class ParamGroupConfig:
    """One parameter group and the optimizer that owns it.

    A list of these, in order, is what ``OptimizerConfig.param_groups`` means:
    each parameter is claimed by the first entry whose ``pattern`` matches its
    FQN, so a catch-all ``.*`` belongs last.

    ``optimizer_name`` and ``optimizer_kwargs`` define the group's optimizer
    completely -- there is no inheritance from ``OptimizerConfig``'s flat
    scalars, which supply the *default* catch-all group and nothing else. A
    config that omits ``lr`` here therefore fails in the optimizer constructor
    rather than silently adopting a value set somewhere else.

    There is no CLI flag for this: ``HfArgumentParser`` builds one flag per
    dataclass *field*, and a list of nested dataclasses has no flag spelling.
    Groups are set from code, by constructing an ``OptimizerConfig`` with
    ``param_groups=[...]``. See that class's ``__post_init__`` for how the flat
    scalars become a group when this list is left empty.
    """

    pattern: str
    """Regex matched against parameter FQNs, e.g. ``r".*\\.bias$"``, ``r".*"``."""

    optimizer_name: str
    """Optimizer class for this group's parameters: ``"Adam"`` or ``"AdamW"``."""

    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the optimizer constructor. Must include everything
    required (``lr`` above all); nothing is filled in from the enclosing config.
    Entries override the run-wide implementation kwargs, so a group can ask for
    ``fused=False`` where the run default is fused."""


@dataclass
class OptimizerConfig:
    """Optimizer (adamw only for now -- the learning path needs just one).

    The schedule lives here rather than in ``TrainingConfig`` because it
    scales this group's learning rate: a lr with no schedule is the degenerate
    case of one, and splitting them would let the two be set independently.
    """

    learning_rate: float = field(default=3e-4, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay"})
    betas: list[float] = field(
        default_factory=lambda: [0.9, 0.999],
        metadata={
            "help": "AdamW (beta1, beta2). Pass as two values: "
            "--betas 0.9 0.95. The default is torch's; torchtitan's reference "
            "LLM recipe uses (0.9, 0.95), which is a NUMERIC change."
        },
    )
    eps: float = field(
        default=1e-8,
        metadata={"help": "AdamW epsilon (denominator floor)."},
    )
    implementation: Literal["fused", "foreach", "for-loop"] = field(
        default="fused",
        metadata={
            "help": "Optimizer kernel. 'fused' is CUDA-only and falls back to "
            "the for-loop kernel elsewhere; on CPU all three are bit-identical. "
            "torchtitan's default."
        },
    )
    param_groups: list[ParamGroupConfig] = field(
        default_factory=list,
        metadata={
            "help": "Per-parameter-group optimizers. Empty (the default) means "
            "one catch-all group built from the flat scalars above. No CLI flag "
            "-- nested dataclass lists cannot be parsed; set it from code."
        },
    )
    lr_scheduler_config: LRSchedulerConfig = field(
        default_factory=LRSchedulerConfig,
        metadata={"help": "Learning-rate schedule (see components/optimizer)."},
    )

    def __post_init__(self) -> None:
        # ``list`` is what the parser can build from two CLI values, but the
        # optimizer wants a tuple and the field must not be mutable: a reused
        # ``HfArgumentParser`` hands every instance the SAME default list (the
        # factory runs once, not per instance), so an in-place edit would leak
        # across runs. Normalizing here makes that unreachable, and the length
        # check is what turns a typo like ``--betas 0.9`` into an error rather
        # than a one-element betas that torch rejects deep in a step.
        betas = tuple(self.betas)
        if len(betas) != 2:
            raise ValueError(
                f"betas must have exactly 2 entries (beta1, beta2), got "
                f"{len(betas)}: {betas}. Pass both: --betas 0.9 0.95."
            )
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"betas must each be in [0, 1), got {betas}")
        self.betas = betas

        # The degenerate grouping: no explicit param_groups means one catch-all
        # group over every trainable parameter, built from the flat scalars.
        # Synthesized here rather than in the container so there is exactly one
        # description of the default, and so a caller that reads
        # ``cfg.param_groups`` sees the groups a run actually uses.
        if not self.param_groups:
            self.param_groups = [
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": self.learning_rate,
                        "weight_decay": self.weight_decay,
                        "betas": self.betas,
                        "eps": self.eps,
                    },
                )
            ]

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
class EMAConfig:
    """Online EMA of model weights (see ``components/optimizer/ema.py``).

    Set ``training.ema_config`` to one of these to turn EMA on; the default
    ``None`` means no EMA is built and the run pays nothing for it -- there is
    no CLI flag (a nested dataclass does not survive ``HfArgumentParser``;
    ``ParamGroupConfig`` is programmatic-only for the same reason). The field
    names and semantics are upstream's.
    """

    decay: float | None = None
    """Fixed decay per firing: ``ema = decay * ema + (1 - decay) * param``.
    If None (default), computed dynamically from ``half_life_fraction``."""

    half_life_fraction: float = 0.05
    """Used when ``decay`` is None:
    ``decay = 2 ** (-1 / (half_life_fraction * num_updates))``. Keeps roughly
    the most recent ``half_life_fraction`` share of updates dominant."""

    start_step: int = 0
    """Last trainer step before EMA tracking begins, so the first update fires
    at ``start_step + update_every_n_steps``."""

    step_bias: int = 0
    """Offset added to the firing count when computing ``num_updates``, for
    renumbering a new training phase without resetting EMA aging. Measured in
    EMA firings, not raw steps. A normal resume needs no bias."""

    update_every_n_steps: int = 1
    """Only fire the EMA update every N real optimizer steps."""

    buffer_patterns: list[str] = field(default_factory=list)
    """Regex patterns (``re.search``, against buffer FQNs) selecting which
    buffers also get an EMA tracked -- e.g. an MoE's ``expert_bias_E``. Empty
    (default): no buffers tracked."""

    def __post_init__(self) -> None:
        if self.update_every_n_steps < 1:
            raise ValueError("ema.update_every_n_steps must be greater than 0.")
        if not math.isfinite(self.half_life_fraction):
            raise ValueError("ema.half_life_fraction must be finite.")
        if self.half_life_fraction <= 0:
            raise ValueError("ema.half_life_fraction must be greater than 0.")
        if self.step_bias < 0:
            raise ValueError(
                "ema.step_bias must not be negative; it is added to the firing "
                "count, and a non-positive count has no decay."
            )
        if self.decay is not None and not (
            math.isfinite(self.decay) and 0 <= self.decay < 1
        ):
            raise ValueError(
                "ema.decay must be finite and in [0, 1); "
                "decay=1 never updates the EMA."
            )
        # A fixed decay replaces the half-life schedule outright, so a
        # half_life_fraction set alongside it would do nothing.
        default_half_life = (
            type(self).__dataclass_fields__["half_life_fraction"].default
        )
        if self.decay is not None and self.half_life_fraction != default_half_life:
            logger.warning(
                "ema.half_life_fraction=%s is ignored because ema.decay=%s is "
                "set; the decay is then fixed and the half-life schedule is "
                "never used. Leave decay unset to use half_life_fraction.",
                self.half_life_fraction,
                self.decay,
            )


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
    names a recipe from ``datasets.text.text.DATASETS`` or
    ``datasets.multimodal.mm_datasets.MM_DATASETS``, or the built-in
    ``local_jsonl`` --
    which is deliberately NOT in either dict, because its corpus path is a
    runtime argument rather than a constant.

    Every non-``random`` dataset builds its graph on Grain, which needs a
    tokenizer, so ``tokenizer_path`` is required there and unused otherwise.
    Multimodal recipes additionally need the optional dependencies
    torchvision/Pillow (and av for video) and the five ``mm_*_token`` strings
    below, which must exist as added tokens in that tokenizer.
    """

    dataset: str = field(
        default="random",
        metadata={
            "help": "Corpus selector: 'random' (synthetic, no assets) | "
            "'local_jsonl' | a key of datasets.text.text.DATASETS | a key of "
            "datasets.multimodal.mm_datasets.MM_DATASETS (needs torchvision)"
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
    prompt_field: str = field(
        default="prompt",
        metadata={"help": "Prompt field used by dataset=local_jsonl_sft."},
    )
    response_field: str = field(
        default="response",
        metadata={"help": "Assistant field used by dataset=local_jsonl_sft."},
    )
    chat_renderer: str | None = field(
        default=None,
        metadata={
            "help": "Multi-turn SFT via the optional renderers package: name "
            "of its renderer config class (e.g. 'Qwen3RendererConfig'). Only "
            "valid with dataset=local_jsonl_sft, whose rows must then carry a "
            "messages list; replaces the single-turn chat-template path. "
            "Needs `pip install renderers==0.1.11`."
        },
    )
    messages_field: str = field(
        default="messages",
        metadata={
            "help": "Row field holding the multi-turn conversation. Used only "
            "when chat_renderer is set."
        },
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
    packing: Literal["concat_then_split", "first_fit"] = field(
        default="concat_then_split",
        metadata={
            "help": "Text packing recipe. 'concat_then_split' concatenates "
            "documents and chunks them into fixed-length rows; 'first_fit' "
            "lays documents of up to the context window into bins instead, "
            "which sustains single-document rows. Ignored for multimodal "
            "recipes, which pack whole documents regardless."
        },
    )
    num_packing_bins: int = field(
        default=8,
        metadata={
            "help": "Candidate rows 'first_fit' keeps open. More bins can "
            "reduce padding, but buffer more documents. Ignored otherwise."
        },
    )
    max_num_documents: int | None = field(
        default=None,
        metadata={
            "help": "Cap on documents packed into one row. None leaves the "
            "frontier unconstrained."
        },
    )
    mm_image_token: str = field(
        default="<|image_pad|>",
        metadata={"help": "Image placeholder token. Multimodal recipes only."},
    )
    mm_video_token: str = field(
        default="<|video_pad|>",
        metadata={"help": "Video placeholder token. Multimodal recipes only."},
    )
    mm_vision_start_token: str = field(
        default="<|vision_start|>",
        metadata={
            "help": "Token opening a vision placeholder run. Multimodal recipes only."
        },
    )
    mm_vision_end_token: str = field(
        default="<|vision_end|>",
        metadata={
            "help": "Token closing a vision placeholder run. Multimodal recipes only."
        },
    )
    mm_pad_token: str = field(
        default="<|endoftext|>",
        metadata={"help": "Padding token. Multimodal recipes only."},
    )

    def __post_init__(self) -> None:
        if self.dataset != "random" and not self.tokenizer_path:
            raise ValueError(
                f"tokenizer_path is required for dataset '{self.dataset}'. "
                "Only 'random' runs without a tokenizer."
            )
        if self.dataset in {"local_jsonl", "local_jsonl_sft"} and not self.dataset_path:
            raise ValueError(f"dataset_path is required for dataset {self.dataset!r}")
        if self.dataset == "local_jsonl_sft" and (
            not self.prompt_field.strip() or not self.response_field.strip()
        ):
            raise ValueError("local_jsonl_sft field names cannot be empty")
        if self.chat_renderer is not None:
            if self.dataset != "local_jsonl_sft":
                raise ValueError(
                    f"chat_renderer requires dataset='local_jsonl_sft', got "
                    f"{self.dataset!r}"
                )
            if not self.messages_field.strip():
                raise ValueError(
                    "messages_field cannot be empty when chat_renderer is set"
                )
        # Membership in ``datasets.text.text.DATASETS`` /
        # ``datasets.multimodal.mm_datasets.MM_DATASETS`` is checked by
        # ``datasets/build.py`` at build time, not here: reading the registries
        # would import the datasets package into the config layer.
        if self.max_num_documents is not None and self.max_num_documents <= 0:
            raise ValueError("max_num_documents must be positive")
        # Validated here even though only 'first_fit' reads it: the field is
        # always parsed, so a bad value would otherwise be accepted silently
        # under the default recipe and only fail after switching to first_fit.
        if self.num_packing_bins <= 0:
            raise ValueError("num_packing_bins must be positive")


@dataclass(kw_only=True)
class SelectiveACConfig:
    """Settings for ``activation_checkpoint_mode='selective'``.

    Ported from torchtitan's ``SelectiveAC.Config``: ``preserve_rng_state``,
    ``determinism_check`` and ``debug`` are the same knobs with the same
    defaults. The save set itself is not configurable (it is
    ``activation_checkpoint._get_default_save_ops``); these are the per-run
    knobs around it.

    Two deviations from upstream, both forced by what hpmesh runs:

    * ``force_recompute_mm_shapes_by_fqns`` defaults to empty rather than
      ``["moe.router.gate"]``. That default assumes the torchtitan ``Module``
      protocol, whose MoE routers are ``nn.Linear`` as ``moe.router.gate``. In
      HF models the router living at ``mlp.gate`` is a container (e.g.
      ``Qwen3MoeTopKRouter``), not a ``Linear``, so the pattern matches nothing
      on an HF MoE and the setting silently does nothing; on a dense HF model
      there is no router at all. Set it to a real ``*.q_proj``/``*.o_proj``
      style name to use it -- a pattern that matches a non-``Linear`` raises
      rather than being ignored, so a wrong guess is loud.
    * The shared save set drops ``aten.topk`` (see that function's docstring:
      HF's routers mutate topk's output in place, which torch's selective
      checkpoint rejects outright). The consequence is that a selective run
      over an HF MoE recomputes its topk and so inherits topk's
      non-determinism on kernels that have any.

    ``preserve_rng_state`` is separate from the flat argument on ``apply_ac``
    -- full and selective each carry their own, as upstream's policy classes
    do, because the two policies make different demands on the recompute's RNG.
    """

    force_recompute_mm_shapes_by_fqns: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Fully-qualified-name substrings selecting the nn.Linear "
            "modules whose weight shapes are recomputed unconditionally. Note "
            "this selects *shapes*, not modules: any matmul with a matching "
            "(in, out) weight is recomputed, wherever it appears. Defaults to "
            "empty -- see the class docstring for why upstream's "
            "'moe.router.gate' default does not carry over."
        },
    )
    preserve_rng_state: bool = field(
        default=True,
        metadata={
            "help": "Stash and restore the RNG state around each checkpointed "
            "region so the backward-time recompute redraws the same random "
            "values, at some speed cost. Set false only if the checkpointed "
            "region is known to hold no random state."
        },
    )
    determinism_check: str = field(
        default="default",
        metadata={
            "help": "Determinism function torch's checkpoint uses to compare "
            "the recompute against the original. 'default' checks tensors that "
            "have no data-invariant structure; 'none' disables."
        },
    )
    debug: bool = field(
        default=False,
        metadata={
            "help": "Capture activation-checkpointing debug information. "
            "Slower; see torch.utils.checkpoint's documentation for details."
        },
    )


@dataclass(kw_only=True)
class MemoryBudgetACConfig:
    """Settings for ``activation_checkpoint_mode='memory_budget'``.

    Ported from torchtitan's ``MemoryBudgetAC.Config``. That policy carries no
    checkpointing code of its own: it sets one process-global,
    ``torch._functorch.config.activation_memory_budget``, and lets the compile
    partitioner trade compute for memory inside each compiled region -- so the
    mode only means anything when ``training.compile`` is on, and selecting it
    without compile is a config error, exactly as upstream validates.

    Upstream's ``visualize_memory_budget_pareto`` is not ported: it dumps SVGs
    into a trainer dump folder, a concept hpmesh's AC path does not have.
    """

    memory_budget: float = field(
        default=0.5,
        metadata={
            "help": "How much the compile partitioner trades compute for "
            "memory: 0.0 is the activation memory of full checkpointing over "
            "the compiled region, 1.0 the default runtime-optimized strategy. "
            "Must be in [0, 1]."
        },
    )

    def __post_init__(self) -> None:
        if not 0 <= self.memory_budget <= 1:
            raise ValueError(
                "memory_budget must be finite and between 0 and 1, got "
                f"{self.memory_budget}"
            )


@dataclass(kw_only=True)
class CompileConfig:
    """Settings for ``training.compile=True`` -- how the model is compiled.

    Ported from torchtitan's ``CompileConfig``, minus its ``components``
    list (hpmesh compiles the model only; the loss has no compile path).
    The defaults are exactly hpmesh's historical behavior -- one whole-model
    ``torch.compile(model, backend="inductor")`` -- so an existing run that
    never touches this config is bitwise unchanged. Each non-default knob is
    independent of the others.
    """

    per_block: bool = field(
        default=False,
        metadata={
            "help": "Compile each decoder layer separately (fullgraph=True) "
            "instead of the model as a whole: the repeated block structure "
            "is traced once and its graph reused, so compile time scales "
            "with one block rather than the depth. False (default) keeps "
            "the whole-model compile."
        },
    )
    backend: str = field(
        default="inductor",
        metadata={
            "help": "torch.compile backend. 'inductor' (default) or "
            "'aot_eager'; on a flex-attention model 'aot_eager' is wrapped "
            "in regional_inductor so the flex regions still lower to "
            "inductor. Any other backend on a flex model is an error, "
            "since flex has no non-inductor lowering."
        },
    )
    enable_async_tensor_parallel: bool = field(
        default=False,
        metadata={
            "help": "Pipeline tensor-parallel collectives with the GEMMs "
            "inside compiled regions (Inductor's micro-pipeline pass). "
            "Requires training.compile=True, tensor_parallel_size > 1, and "
            "a torch carrying torch._inductor.config._micro_pipeline_tp "
            "plus symmetric-memory registration; every missing piece is a "
            "loud error, never a silent skip."
        },
    )

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("compile.backend cannot be empty.")


@dataclass(kw_only=True)
class ValidationConfig:
    """The validation (eval) loop's knobs (see ``Trainer.validate``).

    Set ``training.validation_config`` to one of these to turn validation on;
    the default ``None`` means no validation runs and the training loop is
    bit-identical to before. Programmatic-only, like ``ema_config``: a nested
    dataclass does not survive ``HfArgumentParser``.

    Validation replays the training loss forward-only: summed next-token
    cross-entropy over the whole pass, divided by the global valid-token count
    reduced across DP, so the number is independent of how the batches were
    split across ranks. It updates no parameters and touches no checkpoint
    state.
    """

    freq: int = 10
    """Validate every this many steps (step 1 always validates)."""

    steps: int = -1
    """Batches per validation pass. -1 consumes the finite dataset once
    (the loader is built with repeat=False). Ranks then stop independently,
    so -1 requires data-parallel degree 1 and a finite (non-random) dataset;
    both are rejected at trainer build time rather than hanging the pass."""

    dataset: str | None = None
    """Corpus override for validation, same vocabulary as
    ``dataloader.dataset``. None (the default) validates on the training
    corpus."""

    def __post_init__(self) -> None:
        if self.freq <= 0:
            raise ValueError(f"validation.freq must be positive, got {self.freq}")
        if not (self.steps > 0 or self.steps == -1):
            raise ValueError(
                f"validation.steps must be positive or -1, got {self.steps}"
            )


@dataclass
class TrainingConfig:
    """Training loop hyperparameters and reproducibility."""

    global_batch_size: int = field(
        default=8,
        metadata={
            "help": "Sequences per step across ALL DP ranks. Under pipeline "
            "parallelism this is the step total, not the per-iteration read: "
            "each rank reads global_batch_size / dp_world_size rows once, and "
            "those rows are then sliced into num_pp_microbatches pipeline "
            "micro-batches rather than repeated reads. torchtitan instead reads "
            "num_pp_microbatches separate batches per accumulation group, so "
            "the same global_batch_size there consumes num_pp_microbatches "
            "times more tokens per step; multiply here to match it."
        },
    )
    max_seq_len: int = field(default=64, metadata={"help": "Sequence length"})
    steps: int = field(default=20, metadata={"help": "Number of optimizer steps"})
    seed: int = field(default=42, metadata={"help": "Base RNG seed"})
    compile: bool = field(default=False, metadata={"help": "torch.compile the model"})
    compile_config: CompileConfig = field(
        default_factory=CompileConfig,
        metadata={
            "help": "How the model is compiled (per-block vs whole-model, "
            "backend, async TP). Read only under compile=True; its defaults "
            "reproduce the plain whole-model compile."
        },
    )
    activation_checkpoint_mode: str = field(
        default="none",
        metadata={
            "help": "Activation checkpointing: 'none' (off), 'full' (recompute "
            "each decoder layer during backward), 'selective' (per-op: save "
            "the expensive ops, recompute the rest -- tune it with "
            "selective_ac), or 'memory_budget' (let the compile partitioner "
            "trade compute for memory -- tune it with memory_budget_ac, "
            "requires compile=True). Wraps layers after TP/EP/CP and before "
            "compile/FSDP."
        },
    )
    selective_ac: SelectiveACConfig = field(
        default_factory=SelectiveACConfig,
        metadata={
            "help": "Selective activation checkpointing. Read only under "
            "activation_checkpoint_mode='selective'."
        },
    )
    memory_budget_ac: MemoryBudgetACConfig = field(
        default_factory=MemoryBudgetACConfig,
        metadata={
            "help": "Memory-budget activation checkpointing. Read only under "
            "activation_checkpoint_mode='memory_budget'."
        },
    )
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
    gc_freq: int = field(
        default=50,
        metadata={
            "help": "Run a cyclic garbage collection every this many steps. The "
            "training loop takes the collector over from CPython so it fires at a "
            "step boundary instead of mid-forward."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Micro-batches accumulated per optimizer update. Each is a "
            "full forward/backward; the step's gradients are summed and the "
            "reported loss is the sum over all of them divided by the global "
            "valid-token count, so the number stays comparable across settings. "
            "Contrast num_pp_microbatches, which splits one batch's pipeline "
            "schedule rather than training on more data."
        },
    )
    chunked_loss_num_chunks: int = field(
        default=1,
        metadata={
            "help": "Split the lm_head + cross-entropy computation into this "
            "many sequence chunks, cutting peak logits memory from O(T*V) to "
            "O(T*V/chunks) -- the key memory lever for large-vocabulary models. "
            "1 (the default) disables chunking. Not supported with pipeline "
            "parallelism."
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
    ema_config: EMAConfig | None = field(
        default=None,
        metadata={
            "help": "Online EMA of model weights (see components/optimizer). "
            "None (the default) disables it. No CLI flag -- set it from code, "
            "like optimizer.param_groups."
        },
    )
    validation_config: ValidationConfig | None = field(
        default=None,
        metadata={
            "help": "Periodic validation pass (see Trainer.validate). None "
            "(the default) disables it. No CLI flag -- set it from code, like "
            "ema_config."
        },
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

    @property
    def ema(self) -> EMAConfig | None:
        """The weight EMA's config, or None when EMA is off.

        Unlike the other nested configs there is no always-on default: an EMA
        doubles the weight memory a run carries, so it exists only when the
        run asks for it.
        """
        return self.ema_config

    @property
    def validation(self) -> ValidationConfig | None:
        """The validation loop's config, or None when validation is off.

        Opt-in like ``ema``: a validation pass re-reads the corpus and runs a
        full forward over it, so it exists only when the run asks for it.
        """
        return self.validation_config

    def __post_init__(self) -> None:
        if self.global_batch_size < 1:
            raise ValueError(
                f"global_batch_size must be >= 1, got {self.global_batch_size}"
            )
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be >= 1, got "
                f"{self.gradient_accumulation_steps}"
            )
        if self.chunked_loss_num_chunks < 1:
            raise ValueError(
                "chunked_loss_num_chunks must be >= 1 (1 disables chunking), "
                f"got {self.chunked_loss_num_chunks}"
            )
        if self.activation_checkpoint_mode == "region":
            raise NotImplementedError(
                "training.activation_checkpoint_mode='region' (upstream "
                "RegionAC) needs torch_remat and model-declared remat "
                "regions, which hpmesh has no equivalent of; see "
                "parallel/activation_checkpoint.py's docstring."
            )
        if self.activation_checkpoint_mode not in (
            "none",
            "full",
            "selective",
            "memory_budget",
        ):
            raise ValueError(
                "training.activation_checkpoint_mode must be one of: 'none', "
                "'full', 'selective', 'memory_budget' (got "
                f"{self.activation_checkpoint_mode!r})"
            )
        if self.activation_checkpoint_mode == "memory_budget" and not self.compile:
            raise ValueError(
                "training.activation_checkpoint_mode='memory_budget' requires "
                "training.compile=True: the budget is consumed by the compile "
                "partitioner, so without compile it would silently do nothing."
            )


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
        # Async TP is a compiled-TP optimization: without compile there is no
        # inductor pass to pipeline the collectives, and without TP there are
        # no collectives. Reject both halves here so the error names the flag
        # rather than surfacing deep inside the compile step.
        if self.training.compile_config.enable_async_tensor_parallel:
            if not self.training.compile:
                raise ValueError(
                    "training.compile_config.enable_async_tensor_parallel "
                    "requires training.compile=True: async TP is an inductor "
                    "pass over compiled regions, so without compile it would "
                    "silently do nothing."
                )
            if self.parallel.tp < 2:
                raise ValueError(
                    "training.compile_config.enable_async_tensor_parallel "
                    "requires tensor_parallel_size > 1 (got "
                    f"{self.parallel.tp}): it pipelines the TP collectives, "
                    "and there are none at tp=1."
                )

    # -- Flat view: lets the trainer read cfg.lr / cfg.steps / ... uniformly. --
    # The parallel degrees (dp/tp/pp/cp/ep) are deliberately NOT here: the
    # parallel layer takes ``cfg.parallel`` (a ParallelConfig) directly, so a
    # second flat spelling of the same numbers could only drift apart.
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
    def arch_overrides(self) -> dict[str, Any]:
        return self.model.arch_overrides

    @property
    def experts_implementation(self) -> str:
        return self.model.experts_implementation

    @property
    def compute_dtype(self) -> str | None:
        return self.model.compute_dtype

    @property
    def lr(self) -> float:
        return self.optimizer.learning_rate

    @property
    def weight_decay(self) -> float:
        return self.optimizer.weight_decay

    @property
    def betas(self) -> tuple[float, float]:
        return self.optimizer.betas

    @property
    def eps(self) -> float:
        return self.optimizer.eps

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
    def deterministic(self) -> bool:
        return self.training.deterministic

    @property
    def pipeline_parallel_schedule(self) -> str:
        return self.parallel.pipeline_parallel_schedule

    @property
    def max_norm(self) -> float:
        return self.training.max_norm

    @property
    def dump_folder(self) -> str:
        return self.training.dump_folder

    @property
    def gc_freq(self) -> int:
        return self.training.gc_freq

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.training.gradient_accumulation_steps

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

    @property
    def validation(self) -> ValidationConfig | None:
        return self.training.validation

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
