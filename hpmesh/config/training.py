"""Training-loop config and the nested configs grafted onto it."""

from __future__ import annotations

from dataclasses import dataclass, field

from hpmesh.config.checkpoint import CheckpointConfig
from hpmesh.config.data import DataloaderConfig
from hpmesh.config.optimizer import EMAConfig
from hpmesh.errors import ConfigError, EnvironmentUnsupportedError


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
            raise ConfigError("metrics.log_freq must be greater than 0.")


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
            raise ConfigError(
                "profiler.profile_freq must be greater than or equal to "
                "profiler_warmup + profiler_active."
            )


# HfArgumentParser cannot turn a nested dataclass into a set of flags -- it
# collapses it to one opaque `--<field>` argument and never consults the fields
# inside. The `<x>_config` field is that escape hatch: the config's own group is
# parsed separately in train.py and grafted back on here, which is also where
# its __post_init__ re-runs against the parsed values.
#
# That is what keeps every class in this package named `*Config`. Renaming one is
# not cosmetic -- HfArgumentParser takes the flag stem from the FIELD, so the
# field names below are the CLI: `metrics_config` is what makes this class's
# fields reach the user as `--log_freq` and `--enable_tensorboard`. A class
# named `MetricsArguments` with a field `metrics_arguments` would move every one
# of them behind a `--metrics_arguments` prefix, which is a breaking change.
#
# A plain `checkpoint` field would be nicer to read, but a dataclass field and
# the class it types cannot share a name.

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
            raise ConfigError(
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
            raise ConfigError("compile.backend cannot be empty.")


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
            raise ConfigError(f"validation.freq must be positive, got {self.freq}")
        if not (self.steps > 0 or self.steps == -1):
            raise ConfigError(
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
            raise ConfigError(
                f"global_batch_size must be >= 1, got {self.global_batch_size}"
            )
        if self.max_seq_len < 1:
            raise ConfigError(f"max_seq_len must be >= 1, got {self.max_seq_len}")
        if self.steps < 1:
            raise ConfigError(f"steps must be >= 1, got {self.steps}")
        if self.gradient_accumulation_steps < 1:
            raise ConfigError(
                "gradient_accumulation_steps must be >= 1, got "
                f"{self.gradient_accumulation_steps}"
            )
        if self.chunked_loss_num_chunks < 1:
            raise ConfigError(
                "chunked_loss_num_chunks must be >= 1 (1 disables chunking), "
                f"got {self.chunked_loss_num_chunks}"
            )
        if self.activation_checkpoint_mode == "region":
            raise EnvironmentUnsupportedError(
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
            raise ConfigError(
                "training.activation_checkpoint_mode must be one of: 'none', "
                "'full', 'selective', 'memory_budget' (got "
                f"{self.activation_checkpoint_mode!r})"
            )
        if self.activation_checkpoint_mode == "memory_budget" and not self.compile:
            raise ConfigError(
                "training.activation_checkpoint_mode='memory_budget' requires "
                "training.compile=True: the budget is consumed by the compile "
                "partitioner, so without compile it would silently do nothing."
            )
