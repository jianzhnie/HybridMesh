"""The combination support matrix: one source of truth for what composes.

"Which parallel combinations are supported, refused, or conditional" used to
live in three places at once -- ``config/parallel.py``'s ``__post_init__``,
the assembly-time guards in the ``apply_*`` functions, and the EP swap's
layout probes -- and drifted on every upstream alignment. Every combination
hpmesh has an opinion about is a row here, with its verdict, its rationale
(and unlock condition), and the phase that can decide it:

* ``config`` rows are decidable from the config alone. They carry their
  predicate and message; the ``__post_init__`` of the owning config calls the
  row's check at its original position (first-error ordering is unchanged),
  and ``check_config`` runs them all for consistency tests and docs.
* ``assembly`` rows need runtime information (the model, the resolved
  ``ParallelDims``, the dataset name). The trigger condition stays at the
  guard site; the verdict -- exception type and exact message -- is the row's
  ``reject(...)`` here, so the site cannot quietly disagree with the matrix.
* ``probe`` rows need the HF model's layout (the EP swap's duck-typed
  probes). Same split: the probe triggers, the row rejects.

This is deliberately not a rules engine: rows are named entries with a check
or a reject, full stop. Field-level validation (sizes, allowed values) is not
combination knowledge and stays in the configs; capability probing (torch
knobs) lives in ``accelerator/capabilities.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hpmesh.errors import (
    ConfigError,
    EnvironmentUnsupportedError,
    UnsupportedCombinationError,
)

__all__ = [
    "ENTRIES",
    "Entry",
    "check_config",
    "check_root",
    "check_training",
]

Phase = Literal["config", "assembly", "probe"]


@dataclass(frozen=True)
class Entry:
    """One matrix row: a combination verdict and where it is enforced."""

    name: str
    phase: Phase
    error: type[Exception]
    reason: str
    guard: str


ENTRIES: dict[str, Entry] = {}


def _entry(name: str, phase: Phase, error: type[Exception], reason: str, guard: str):
    """Register a matrix row; decorates the function that enforces it."""
    ENTRIES[name] = Entry(name, phase, error, reason, guard)

    def wrap(fn):
        return fn

    return wrap


# == config phase: ParallelConfig ============================================
# Called from ``ParallelConfig.__post_init__`` at the positions the inline
# guards used to hold, so first-error ordering is unchanged.


@_entry(
    "sequence_parallel_required",
    "config",
    UnsupportedCombinationError,
    "hpmesh has one TP realization and it is the sequence-parallel one; "
    "False has nothing to select. Leave it true, or set tp=1.",
    "config/parallel.py::__post_init__",
)
def sequence_parallel_required(cfg) -> None:
    if not cfg.enable_sequence_parallel:
        raise UnsupportedCombinationError(
            "parallelism.enable_sequence_parallel=false is not supported: "
            "hpmesh's tensor parallelism is sequence-parallel by "
            "construction (the fused TP GEMMs gather/scatter the sequence "
            "and the batch is sharded T/tp). There is no "
            "replicated-activation TP path to fall back to, so this flag "
            "has nothing to disable. Leave it true, or set "
            "tensor_parallel_size=1 to drop TP."
        )


@_entry(
    "tp_ep_cp",
    "config",
    UnsupportedCombinationError,
    "tp x ep is supported (TP dense, EP owns the routed experts); adding CP "
    "on top is unverified -- the token-count reductions and dispatcher "
    "layouts have not been exercised together.",
    "config/parallel.py::__post_init__",
)
def tp_ep_cp(cfg) -> None:
    if (
        cfg.tensor_parallel_size > 1
        and cfg.expert_parallel_size > 1
        and cfg.context_parallel_size > 1
    ):
        raise UnsupportedCombinationError(
            "tensor_parallel_size > 1 with expert_parallel_size > 1 and "
            "context_parallel_size > 1 is not supported: tp x ep x cp is "
            "unverified. Run tp x ep with context_parallel_size=1, or ep x "
            "cp with tensor_parallel_size=1."
        )


@_entry(
    "deepep_hybridep",
    "config",
    EnvironmentUnsupportedError,
    "CUDA-only kernels plus torchtitan's distributed/deepep/ wrappers, which "
    "hpmesh does not vendor. Unlock: vendor the wrappers, add the CUDA-only "
    "dependency as an optional extra, re-validate numerics on a CUDA device.",
    "config/parallel.py::__post_init__",
)
def deepep_hybridep(cfg) -> None:
    if cfg.ep_token_dispatcher in ("deepep", "hybridep"):
        raise EnvironmentUnsupportedError(
            f"ep_token_dispatcher={cfg.ep_token_dispatcher!r} is a "
            "registered gap, not a supported backend: it is CUDA-only and "
            "requires the deep_ep/hybridep kernels plus torchtitan's "
            "distributed/deepep/ wrappers, which hpmesh does not vendor "
            "(environment not covered; see docs/hpmesh_upstream_map.md "
            "table D). Unlock conditions: vendor the wrappers, add the "
            "CUDA-only dependency as an optional extra, and re-validate "
            "numerics on a CUDA device. Use 'alltoall' meanwhile."
        )


@_entry(
    "dispatcher_requires_ep",
    "config",
    UnsupportedCombinationError,
    "The EP swap is the only place a token dispatcher is installed, and it "
    "does not run at ep=1.",
    "config/parallel.py::__post_init__",
)
def dispatcher_requires_ep(cfg) -> None:
    if cfg.ep_token_dispatcher != "alltoall" and cfg.expert_parallel_size == 1:
        raise UnsupportedCombinationError(
            f"ep_token_dispatcher={cfg.ep_token_dispatcher!r} has no "
            "effect at expert_parallel_size=1: the EP swap is the only "
            "place a token dispatcher is installed and it does not run at "
            "ep=1. Set expert_parallel_size > 1, or keep 'alltoall'."
        )


@_entry(
    "ptrr_load_balancer",
    "config",
    UnsupportedCombinationError,
    "ptrr derives its schedule from a BlockMask, which hpmesh's CP kernel "
    "does not consume. Use 'headtail' or None.",
    "config/parallel.py::__post_init__ (backstop: "
    "context_parallel/input_shard.py)",
)
def ptrr_load_balancer(cfg) -> None:
    if cfg.context_parallel_load_balancer == "ptrr":
        raise UnsupportedCombinationError(
            "parallelism.context_parallel_load_balancer='ptrr' is not "
            "implemented in hpmesh: it derives its schedule from a "
            "BlockMask, which hpmesh's CP kernel does not consume. Use "
            "'headtail' or None."
        )


@_entry(
    "ulysses_no_load_balancer",
    "config",
    UnsupportedCombinationError,
    "Every rank attends the full sequence in whatever order the all-to-all "
    "delivers; a load balancer's rearrangement would make that a permuted "
    "corpus, and nothing would raise.",
    "config/parallel.py::__post_init__",
)
def ulysses_no_load_balancer(cfg) -> None:
    if (
        cfg.context_parallel_strategy == "ulysses"
        and cfg.context_parallel_load_balancer is not None
    ):
        raise UnsupportedCombinationError(
            "parallelism.context_parallel_strategy='ulysses' requires "
            "context_parallel_load_balancer=None: every rank attends the "
            "full sequence in whatever order the all-to-all delivers, and "
            "a load balancer's rearrangement would make that a permuted "
            "corpus. Nothing raises: the attention is over the wrong "
            "order of the right tokens, so the loss stays finite and the "
            "run trains a different model. "
            f"(got {cfg.context_parallel_load_balancer!r})"
        )


# == config phase: TrainingConfig ============================================


@_entry(
    "region_ac",
    "config",
    EnvironmentUnsupportedError,
    "RegionAC needs torch_remat and model-declared remat regions, which "
    "hpmesh has no equivalent of. Unlock: add the torch_remat dependency "
    "plus a region-declaration channel on HF decoder layers.",
    "config/training.py::TrainingConfig.__post_init__ (backstop: "
    "parallel/activation_checkpoint.py::apply_ac)",
)
def region_ac(training) -> None:
    if training.activation_checkpoint_mode == "region":
        raise EnvironmentUnsupportedError(
            "training.activation_checkpoint_mode='region' (upstream "
            "RegionAC) needs torch_remat and model-declared remat "
            "regions, which hpmesh has no equivalent of; see "
            "parallel/activation_checkpoint.py's docstring."
        )


@_entry(
    "memory_budget_requires_compile",
    "config",
    ConfigError,
    "The memory budget is consumed by the compile partitioner, so without "
    "compile it would silently do nothing.",
    "config/training.py::TrainingConfig.__post_init__",
)
def memory_budget_requires_compile(training) -> None:
    if training.activation_checkpoint_mode == "memory_budget" and not training.compile:
        raise ConfigError(
            "training.activation_checkpoint_mode='memory_budget' requires "
            "training.compile=True: the budget is consumed by the compile "
            "partitioner, so without compile it would silently do nothing."
        )


# == config phase: HybridMeshConfig cross-group ==============================


@_entry(
    "cp_divides_seq_len",
    "config",
    ConfigError,
    "Cross-group check: CP must divide the sequence length.",
    "config/root.py::HybridMeshConfig.__post_init__",
)
def cp_divides_seq_len(root) -> None:
    if root.training.max_seq_len % root.parallel.cp != 0:
        raise ConfigError(
            f"max_seq_len ({root.training.max_seq_len}) must be divisible by "
            f"cp ({root.parallel.cp})"
        )


@_entry(
    "async_tp_requires_compile",
    "config",
    ConfigError,
    "Async TP is an inductor pass over compiled regions; without compile it "
    "would silently do nothing.",
    "config/root.py::HybridMeshConfig.__post_init__",
)
def async_tp_requires_compile(root) -> None:
    if (
        root.training.compile_config.enable_async_tensor_parallel
        and not root.training.compile
    ):
        raise ConfigError(
            "training.compile_config.enable_async_tensor_parallel "
            "requires training.compile=True: async TP is an inductor "
            "pass over compiled regions, so without compile it would "
            "silently do nothing."
        )


@_entry(
    "async_tp_requires_tp",
    "config",
    ConfigError,
    "Async TP pipelines the TP collectives, and there are none at tp=1.",
    "config/root.py::HybridMeshConfig.__post_init__",
)
def async_tp_requires_tp(root) -> None:
    if (
        root.training.compile_config.enable_async_tensor_parallel
        and root.training.compile
        and root.parallel.tp < 2
    ):
        raise ConfigError(
            "training.compile_config.enable_async_tensor_parallel "
            "requires tensor_parallel_size > 1 (got "
            f"{root.parallel.tp}): it pipelines the TP collectives, "
            "and there are none at tp=1."
        )


_CONFIG_SCOPES = {
    "parallel": (
        sequence_parallel_required,
        tp_ep_cp,
        deepep_hybridep,
        dispatcher_requires_ep,
        ptrr_load_balancer,
        ulysses_no_load_balancer,
    ),
    "training": (region_ac, memory_budget_requires_compile),
    "root": (cp_divides_seq_len, async_tp_requires_compile, async_tp_requires_tp),
}


def check_config(parallel) -> None:
    """Every config-phase row over a ``ParallelConfig``, in matrix order."""
    for check in _CONFIG_SCOPES["parallel"]:
        check(parallel)


def check_training(training) -> None:
    """Every config-phase row over a ``TrainingConfig``, in matrix order."""
    for check in _CONFIG_SCOPES["training"]:
        check(training)


def check_root(root) -> None:
    """Every config-phase row over a ``HybridMeshConfig``, in matrix order."""
    for check in _CONFIG_SCOPES["root"]:
        check(root)


# == assembly phase: verdicts called from the guard sites ====================


@_entry(
    "pp_activation_checkpoint",
    "assembly",
    UnsupportedCombinationError,
    "AC belongs between apply_tp and compile in the per-chunk pipeline, "
    "which does not accept it yet.",
    "parallel/parallelize_hf.py::parallelize_hf_transformers",
)
def pp_activation_checkpoint() -> None:
    raise UnsupportedCombinationError(
        "activation checkpointing is not wired through the pp > 1 path: "
        "it belongs between apply_tp and compile in the per-chunk "
        "pipeline below, which does not accept it yet."
    )


@_entry(
    "pp_validation",
    "assembly",
    UnsupportedCombinationError,
    "The pipeline schedule is driven through its training seam (the loss is "
    "computed and backwarded inside the schedule step); there is no "
    "eval-only pipeline path.",
    "trainer/validation.py::_check_validation_feasibility",
)
def pp_validation() -> None:
    raise UnsupportedCombinationError(
        "validation with pipeline parallelism is not supported: "
        "hpmesh drives the pipeline schedule through its training "
        "seam, where the last stage's loss is computed and backwarded "
        "inside the schedule step. There is no eval-only pipeline "
        "path; run validation with pipeline_parallel_size=1."
    )


@_entry(
    "validation_once_requires_dp1",
    "assembly",
    ConfigError,
    "steps=-1 stops each rank when its own shard is exhausted; with DP > 1 "
    "the ranks can exhaust at different iterations and hang on the pass's "
    "collectives.",
    "trainer/validation.py::_check_validation_feasibility",
)
def validation_once_requires_dp1(dp_world_size: int) -> None:
    raise ConfigError(
        "validation.steps=-1 runs one finite pass over the dataset "
        "(the loader is built with repeat=False). With data-parallel "
        f"degree > 1 ({dp_world_size}), ranks can exhaust at different "
        "iterations and hang on the validation collectives. Set "
        "validation.steps to a positive count so every rank runs the "
        "same number of batches, or run with data-parallel degree 1."
    )


@_entry(
    "validation_once_requires_finite_corpus",
    "assembly",
    ConfigError,
    "steps=-1 against the synthetic corpus has no exhaustion at all: the "
    "random source is infinite, so 'one finite pass' never ends.",
    "trainer/validation.py::_check_validation_feasibility",
)
def validation_once_requires_finite_corpus() -> None:
    raise ConfigError(
        "validation.steps=-1 consumes the dataset once, but the "
        "'random' corpus is an infinite synthetic source that never "
        "exhausts. Set validation.steps to a positive count, or name a "
        "finite validation dataset."
    )


@_entry(
    "ep_checkpoint",
    "assembly",
    UnsupportedCombinationError,
    "Expert weights are rank-heterogeneous plain tensors and the current "
    "checkpoint backends treat them as replicated. Unlock: EP-aware expert "
    "state serialization.",
    "trainer/trainer.py::Trainer.__init__",
)
def ep_checkpoint(ep: int) -> None:
    raise UnsupportedCombinationError(
        f"expert_parallel_size={ep} with checkpointing "
        "is not supported: expert weights are rank-heterogeneous plain "
        "tensors and the current checkpoint backends treat them as "
        "replicated. Disable checkpointing until EP-aware expert state "
        "serialization is implemented."
    )


@_entry(
    "chunked_loss_pp",
    "assembly",
    UnsupportedCombinationError,
    "Under PP the last stage's loss runs inside the schedule on "
    "materialized logits; rewiring that seam for hidden states plus a "
    "per-chunk backward is a PP-side change.",
    "trainer/trainer.py::Trainer.__init__",
)
def chunked_loss_pp(chunks: int, pp: int) -> None:
    raise UnsupportedCombinationError(
        f"chunked_loss_num_chunks={chunks} with "
        f"pipeline_parallel_size={pp} is not "
        "supported: the pipeline last stage's loss runs inside the "
        "schedule on materialized logits. Run chunked loss without "
        "pipeline parallelism."
    )


@_entry(
    "pp_cp_ep",
    "assembly",
    UnsupportedCombinationError,
    "CP shards the batch the schedule consumes and EP swaps MoE blocks per "
    "chunk; neither path is wired through the pipeline.",
    "parallel/pipeline_parallel/apply.py::apply_pp",
)
def pp_cp_ep() -> None:
    raise UnsupportedCombinationError(
        "pp > 1 does not compose with cp > 1 or ep > 1 yet: CP shards the "
        "batch the schedule consumes and EP swaps MoE blocks per chunk, and "
        "neither path is wired through the pipeline. Run them separately."
    )


@_entry(
    "pp_real_corpus",
    "assembly",
    UnsupportedCombinationError,
    "A packed real corpus supplies per-token positions, and the pipeline "
    "body does not thread them through the schedule.",
    "parallel/pipeline_parallel/apply.py::apply_pp",
)
def pp_real_corpus() -> None:
    raise UnsupportedCombinationError(
        "pp > 1 supports only the synthetic 'random' corpus: a packed real "
        "corpus supplies per-token positions, and the pipeline body does "
        "not thread them through the schedule."
    )


@_entry(
    "pp_weight_tying",
    "assembly",
    UnsupportedCombinationError,
    "The split puts the embedding on the first stage and the head on the "
    "last, and each stage's deep copy would train an independent copy of "
    "the shared weight.",
    "parallel/pipeline_parallel/apply.py::apply_pp",
)
def pp_weight_tying() -> None:
    raise UnsupportedCombinationError(
        "pp > 1 with tied word embeddings is not supported: the split puts "
        "the embedding on the first stage and the head on the last, and "
        "each stage's deep copy would train an independent copy of the "
        "shared weight."
    )


@_entry(
    "shared_expert_tp",
    "assembly",
    UnsupportedCombinationError,
    "The TP plan shards a shared expert with the dense colwise/rowwise "
    "realizers; composing those with the MoE sequence-boundary collectives "
    "is unverified. Use tp=1, or ep>1 (the EP swap handles shared experts).",
    "parallel/tensor_parallel/apply.py::apply_tp",
)
def shared_expert_tp(module_path: str, block: object) -> None:
    raise UnsupportedCombinationError(
        f"TP over {module_path} ({type(block).__name__}): the block "
        "has a shared expert, which the plan shards with the dense "
        "colwise/rowwise realizers. Composing those with the MoE "
        "sequence-boundary collectives is unverified; use tp=1, or "
        "ep > 1 (the EP swap handles shared experts)."
    )


@_entry(
    "tp_moe_specs_without_block",
    "assembly",
    UnsupportedCombinationError,
    "The plan declares MoE TP specs but no HF MoE block was found; running "
    "TP with the experts silently replicated is refused.",
    "parallel/tensor_parallel/apply.py::apply_tp",
)
def tp_moe_specs_without_block(tp: int, model: object) -> None:
    raise UnsupportedCombinationError(
        f"apply_tp with tp={tp}: the plan declares MoE TP specs "
        "but no HF MoE block was found on "
        f"{type(model).__name__}. Refusing to run TP with the experts "
        "silently replicated."
    )


@_entry(
    "tp_moe_non_tensor_output",
    "assembly",
    UnsupportedCombinationError,
    "The boundary reduce-scatter has no defined place to run when the MoE "
    "block returns something other than a bare hidden-states tensor.",
    "parallel/tensor_parallel/tp.py::_TPMoeSequenceBoundary.forward",
)
def tp_moe_non_tensor_output(boundary: object, out: object) -> None:
    raise UnsupportedCombinationError(
        f"TP over {type(boundary).__name__}: the MoE block returned "
        f"{type(out).__name__}, not a bare hidden-states tensor. The "
        "boundary reduce-scatter has no defined place to run; refusing "
        "rather than dropping part of the output."
    )


@_entry(
    "quantile_requires_ep",
    "assembly",
    UnsupportedCombinationError,
    "Quantile balancing is installed by the EP swap, which ep=1 never runs "
    "-- there is no hpmesh MoE to balance.",
    "parallel/expert_parallel/apply.py::apply_ep",
)
def quantile_requires_ep() -> None:
    raise UnsupportedCombinationError(
        "moe_quantile_balancing is installed by the EP swap, which "
        "ep=1 never runs -- there is no hpmesh MoE to balance. Run "
        "with ep > 1 to use it."
    )


@_entry(
    "ptrr_load_balancer_backstop",
    "assembly",
    UnsupportedCombinationError,
    "Assembly-time backstop for the config-phase 'ptrr_load_balancer' row: "
    "a caller that bypasses the config still hits the same refusal.",
    "parallel/context_parallel/input_shard.py::_cp_load_balancer",
)
def ptrr_load_balancer_backstop() -> None:
    raise UnsupportedCombinationError(
        "'ptrr' load balancing builds its schedule from a BlockMask and is "
        "not wired in hpmesh yet; use 'headtail' or None."
    )


# == probe phase: verdicts called from the EP swap's layout probes ===========


@_entry(
    "gpt_oss_layout",
    "probe",
    UnsupportedCombinationError,
    "GPT-OSS carries per-expert bias vectors, a transposed (E, D, 2F) "
    "layout, and a hardcoded clamped sigmoid-GLU activation; hpmesh's "
    "GroupedExperts has no slot for them. Unlock: a bias-bearing expert "
    "module with its own activation seam.",
    "parallel/expert_parallel/probe.py::_fused_experts_of",
)
def gpt_oss_layout(experts: object) -> None:
    raise UnsupportedCombinationError(
        f"{type(experts).__name__} carries per-expert bias vectors, which "
        "hpmesh's GroupedExperts has no slot for. Only GPT-OSS has them, "
        "and it differs further: its gate_up_proj is transposed to "
        "(E, D, 2F) and its activation is a hardcoded clamped sigmoid-GLU "
        "rather than a module. Support needs a bias-bearing expert module "
        "with its own activation seam, not a wider copy here."
    )


@_entry(
    "group_limited_greedy",
    "probe",
    UnsupportedCombinationError,
    "DeepSeek-V2's group_limited_greedy scores a group by its single best "
    "expert (max); the implemented rule sums the group's top-2 "
    "(DeepSeek-V3/GLM4), and routing with the wrong rule picks different "
    "experts. Unlock: a group-scoring option in TokenChoiceTopKRouter.",
    "parallel/expert_parallel/probe.py::_read_expert_groups",
)
def group_limited_greedy() -> None:
    raise UnsupportedCombinationError(
        "DeepSeek-V2's group_limited_greedy scores a group by its single "
        "best expert (max); the implemented rule sums the group's top-2 "
        "(DeepSeek-V3/GLM4). Routing this checkpoint with that rule picks "
        "different experts, so refusing is the point -- add a group-scoring "
        "option to TokenChoiceTopKRouter to support it."
    )


@_entry(
    "router_bias",
    "probe",
    UnsupportedCombinationError,
    "RouterGateLinear has no slot for a router bias. Every supported family "
    "is bias-free, so this fires only on a family the probe does not know.",
    "parallel/expert_parallel/convert.py::_convert_block",
)
def router_bias(router_gate: object) -> None:
    raise UnsupportedCombinationError(
        f"{type(router_gate).__name__} carries a router bias, which "
        "RouterGateLinear has no slot for. Every supported family "
        "(Qwen3Moe, OLMoE, Mixtral, DeepSeek-V2/V3, GLM4) is bias-free, "
        "so this fires only on a family the probe does not know."
    )


@_entry(
    "quantile_requires_sigmoid",
    "probe",
    UnsupportedCombinationError,
    "The quantile scheme is defined over sigmoid scores (the histogram "
    "range derives from their [0, 1] bound).",
    "parallel/expert_parallel/convert.py::_convert_block",
)
def quantile_requires_sigmoid(score_func: str, block: object) -> None:
    raise UnsupportedCombinationError(
        f"quantile-balanced routing requires sigmoid router scores, "
        f"got {score_func!r} for {type(block).__name__}."
    )


@_entry(
    "quantile_no_group_limit",
    "probe",
    UnsupportedCombinationError,
    "Quantile-balanced routing selects a free Top-(K+1) over all experts; "
    "group-limited routing is incompatible with it (a single group is no "
    "restriction and is accepted).",
    "parallel/expert_parallel/convert.py::_convert_block",
)
def quantile_no_group_limit(block: object) -> None:
    raise UnsupportedCombinationError(
        f"quantile-balanced routing selects a free Top-(K+1) over all "
        f"experts; {type(block).__name__}'s group-limited routing is "
        "incompatible with it. (A single group is no restriction and "
        "is accepted.)"
    )


@_entry(
    "shared_expert_gate",
    "probe",
    UnsupportedCombinationError,
    "Qwen2Moe's shared_expert_gate multiplies where MoE.shared_experts "
    "only adds.",
    "parallel/expert_parallel/convert.py::_convert_block",
)
def shared_expert_gate(block: object) -> None:
    raise UnsupportedCombinationError(
        f"{type(block).__name__} gates its shared expert "
        "(shared_expert_gate); MoE's shared_experts is additive only. "
        "Qwen3Moe has no shared expert, so this is unreachable there."
    )


@_entry(
    "shared_expert_tp_ep",
    "probe",
    UnsupportedCombinationError,
    "tp x ep over a shared-expert block: the TP plan shards it with the "
    "dense realizers, and composing those with the swapped MoE's "
    "sequence-sharded dispatch layout is unverified.",
    "parallel/expert_parallel/convert.py::_convert_block",
)
def shared_expert_tp_ep(block: object) -> None:
    raise UnsupportedCombinationError(
        f"tp x ep over {type(block).__name__}: the block has a shared "
        "expert, which the TP plan shards with the dense colwise/rowwise "
        "realizers. Composing those with the swapped MoE's sequence-"
        "sharded dispatch layout is unverified; run shared-expert models "
        "with tp=1 (EP handles the shared expert) or ep=1."
    )
