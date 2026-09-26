"""The combination support matrix: one source of truth for what composes.

"Which parallel combinations are supported, refused, or conditional" used to
live in three places at once -- ``config/parallel.py``'s ``__post_init__``,
the assembly-time guards in the ``apply_*`` functions, and the EP swap's
layout probes -- and drifted on every upstream alignment. Every combination
hpmesh has an opinion about is a plain function below plus one row in the
``ENTRIES`` table at the bottom of this file: the function carries the verdict
(exception type, exact message) and the rationale (its docstring); the row
records the phase that can decide it and the guard's location.

* ``config`` rows are decidable from the config alone. The owning config's
  ``__post_init__`` calls the function at its original position
  (first-error ordering is unchanged); ``check_config`` /
  ``check_training`` / ``check_root`` filter the table by ``scope`` for
  consistency tests and docs.
* ``assembly`` rows need runtime information (the model, the resolved
  ``ParallelDims``, the dataset name). The trigger condition stays at the
  guard site; the verdict is the function here, so the site cannot quietly
  disagree with the matrix.
* ``probe`` rows need the HF model's layout (the EP swap's duck-typed
  probes). Same split: the probe triggers, the row's function rejects.

This is deliberately not a rules engine: plain functions, plus one flat
table. Field-level validation (sizes, allowed values) is not combination
knowledge and stays in the configs; capability probing (torch knobs) lives in
``accelerator/capabilities.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from hpmesh.errors import (
    ConfigError,
    EnvironmentUnsupportedError,
    UnsupportedCombinationError,
)

__all__ = [
    "ENTRIES",
    "Row",
    "check_config",
    "check_root",
    "check_training",
]

Phase = Literal["config", "assembly", "probe"]


@dataclass(frozen=True)
class Row:
    """One matrix row. ``name``/``reason`` derive from the function."""

    fn: Callable
    phase: Phase
    error: type[Exception]
    guard: str
    scope: str | None = None

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def reason(self) -> str:
        return (self.fn.__doc__ or "").strip()


# == config phase: ParallelConfig ========================================


def sequence_parallel_required(cfg) -> None:
    """hpmesh has one TP realization and it is the sequence-parallel one; False has
    nothing to select. Leave it true, or set tp=1.
    """
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



def tp_ep_cp(cfg) -> None:
    """tp x ep is supported (TP dense, EP owns the routed experts); adding CP on top is
    unverified -- the token-count reductions and dispatcher layouts have not been
    exercised together.
    """
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



def deepep_hybridep(cfg) -> None:
    """CUDA-only kernels plus torchtitan's distributed/deepep/ wrappers, which hpmesh
    does not vendor. Unlock: vendor the wrappers, add the CUDA-only dependency as
    an optional extra, re-validate numerics on a CUDA device.
    """
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



def dispatcher_requires_ep(cfg) -> None:
    """The EP swap is the only place a token dispatcher is installed, and it does not
    run at ep=1.
    """
    if cfg.ep_token_dispatcher != "alltoall" and cfg.expert_parallel_size == 1:
        raise UnsupportedCombinationError(
            f"ep_token_dispatcher={cfg.ep_token_dispatcher!r} has no "
            "effect at expert_parallel_size=1: the EP swap is the only "
            "place a token dispatcher is installed and it does not run at "
            "ep=1. Set expert_parallel_size > 1, or keep 'alltoall'."
        )



def ptrr_load_balancer(cfg) -> None:
    """ptrr derives its schedule from a BlockMask, which hpmesh's CP kernel does not
    consume. Use 'headtail' or None.
    """
    if cfg.context_parallel_load_balancer == "ptrr":
        raise UnsupportedCombinationError(
            "parallelism.context_parallel_load_balancer='ptrr' is not "
            "implemented in hpmesh: it derives its schedule from a "
            "BlockMask, which hpmesh's CP kernel does not consume. Use "
            "'headtail' or None."
        )



def ulysses_no_load_balancer(cfg) -> None:
    """Every rank attends the full sequence in whatever order the all-to-all delivers; a
    load balancer's rearrangement would make that a permuted corpus, and nothing
    would raise.
    """
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



# == config phase: TrainingConfig ========================================


def region_ac(training) -> None:
    """RegionAC needs torch_remat and model-declared remat regions, which hpmesh has no
    equivalent of. Unlock: add the torch_remat dependency plus a region-
    declaration channel on HF decoder layers.
    """
    if training.activation_checkpoint_mode == "region":
        raise EnvironmentUnsupportedError(
            "training.activation_checkpoint_mode='region' (upstream "
            "RegionAC) needs torch_remat and model-declared remat "
            "regions, which hpmesh has no equivalent of; see "
            "parallel/activation_checkpoint.py's docstring."
        )



def memory_budget_requires_compile(training) -> None:
    """The memory budget is consumed by the compile partitioner, so without compile it
    would silently do nothing.
    """
    if training.activation_checkpoint_mode == "memory_budget" and not training.compile:
        raise ConfigError(
            "training.activation_checkpoint_mode='memory_budget' requires "
            "training.compile=True: the budget is consumed by the compile "
            "partitioner, so without compile it would silently do nothing."
        )



# == config phase: HybridMeshConfig cross-group =======================


def cp_divides_seq_len(root) -> None:
    """Cross-group check: CP must divide the sequence length.
    """
    if root.training.max_seq_len % root.parallel.cp != 0:
        raise ConfigError(
            f"max_seq_len ({root.training.max_seq_len}) must be divisible by "
            f"cp ({root.parallel.cp})"
        )



def async_tp_requires_compile(root) -> None:
    """Async TP is an inductor pass over compiled regions; without compile it would
    silently do nothing.
    """
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



def async_tp_requires_tp(root) -> None:
    """Async TP pipelines the TP collectives, and there are none at tp=1.
    """
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



# == assembly phase: verdicts called from the guard sites =================


def pp_activation_checkpoint() -> None:
    """AC belongs between apply_tp and compile in the per-chunk pipeline, which does not
    accept it yet.
    """
    raise UnsupportedCombinationError(
        "activation checkpointing is not wired through the pp > 1 path: "
        "it belongs between apply_tp and compile in the per-chunk "
        "pipeline below, which does not accept it yet."
    )



def pp_validation() -> None:
    """The pipeline schedule is driven through its training seam (the loss is computed
    and backwarded inside the schedule step); there is no eval-only pipeline
    path.
    """
    raise UnsupportedCombinationError(
        "validation with pipeline parallelism is not supported: "
        "hpmesh drives the pipeline schedule through its training "
        "seam, where the last stage's loss is computed and backwarded "
        "inside the schedule step. There is no eval-only pipeline "
        "path; run validation with pipeline_parallel_size=1."
    )



def validation_once_requires_dp1(dp_world_size: int) -> None:
    """steps=-1 stops each rank when its own shard is exhausted; with DP > 1 the ranks
    can exhaust at different iterations and hang on the pass's collectives.
    """
    raise ConfigError(
        "validation.steps=-1 runs one finite pass over the dataset "
        "(the loader is built with repeat=False). With data-parallel "
        f"degree > 1 ({dp_world_size}), ranks can exhaust at different "
        "iterations and hang on the validation collectives. Set "
        "validation.steps to a positive count so every rank runs the "
        "same number of batches, or run with data-parallel degree 1."
    )



def validation_once_requires_finite_corpus() -> None:
    """steps=-1 against the synthetic corpus has no exhaustion at all: the random source
    is infinite, so 'one finite pass' never ends.
    """
    raise ConfigError(
        "validation.steps=-1 consumes the dataset once, but the "
        "'random' corpus is an infinite synthetic source that never "
        "exhausts. Set validation.steps to a positive count, or name a "
        "finite validation dataset."
    )



def ep_checkpoint(ep: int) -> None:
    """Expert weights are rank-heterogeneous plain tensors and the current checkpoint
    backends treat them as replicated. Unlock: EP-aware expert state
    serialization.
    """
    raise UnsupportedCombinationError(
        f"expert_parallel_size={ep} with checkpointing "
        "is not supported: expert weights are rank-heterogeneous plain "
        "tensors and the current checkpoint backends treat them as "
        "replicated. Disable checkpointing until EP-aware expert state "
        "serialization is implemented."
    )



def chunked_loss_pp(chunks: int, pp: int) -> None:
    """Under PP the last stage's loss runs inside the schedule on materialized logits;
    rewiring that seam for hidden states plus a per-chunk backward is a PP-side
    change.
    """
    raise UnsupportedCombinationError(
        f"chunked_loss_num_chunks={chunks} with "
        f"pipeline_parallel_size={pp} is not "
        "supported: the pipeline last stage's loss runs inside the "
        "schedule on materialized logits. Run chunked loss without "
        "pipeline parallelism."
    )



def pp_cp_ep() -> None:
    """CP shards the batch the schedule consumes and EP swaps MoE blocks per chunk;
    neither path is wired through the pipeline.
    """
    raise UnsupportedCombinationError(
        "pp > 1 does not compose with cp > 1 or ep > 1 yet: CP shards the "
        "batch the schedule consumes and EP swaps MoE blocks per chunk, and "
        "neither path is wired through the pipeline. Run them separately."
    )



def pp_real_corpus() -> None:
    """A packed real corpus supplies per-token positions, and the pipeline body does not
    thread them through the schedule.
    """
    raise UnsupportedCombinationError(
        "pp > 1 supports only the synthetic 'random' corpus: a packed real "
        "corpus supplies per-token positions, and the pipeline body does "
        "not thread them through the schedule."
    )



def pp_weight_tying() -> None:
    """The split puts the embedding on the first stage and the head on the last, and
    each stage's deep copy would train an independent copy of the shared weight.
    """
    raise UnsupportedCombinationError(
        "pp > 1 with tied word embeddings is not supported: the split puts "
        "the embedding on the first stage and the head on the last, and "
        "each stage's deep copy would train an independent copy of the "
        "shared weight."
    )



def shared_expert_tp(module_path: str, block: object) -> None:
    """The TP plan shards a shared expert with the dense colwise/rowwise realizers;
    composing those with the MoE sequence-boundary collectives is unverified. Use
    tp=1, or ep>1 (the EP swap handles shared experts).
    """
    raise UnsupportedCombinationError(
        f"TP over {module_path} ({type(block).__name__}): the block "
        "has a shared expert, which the plan shards with the dense "
        "colwise/rowwise realizers. Composing those with the MoE "
        "sequence-boundary collectives is unverified; use tp=1, or "
        "ep > 1 (the EP swap handles shared experts)."
    )



def tp_moe_specs_without_block(tp: int, model: object) -> None:
    """The plan declares MoE TP specs but no HF MoE block was found; running TP with the
    experts silently replicated is refused.
    """
    raise UnsupportedCombinationError(
        f"apply_tp with tp={tp}: the plan declares MoE TP specs "
        "but no HF MoE block was found on "
        f"{type(model).__name__}. Refusing to run TP with the experts "
        "silently replicated."
    )



def tp_moe_non_tensor_output(boundary: object, out: object) -> None:
    """The boundary reduce-scatter has no defined place to run when the MoE block
    returns something other than a bare hidden-states tensor.
    """
    raise UnsupportedCombinationError(
        f"TP over {type(boundary).__name__}: the MoE block returned "
        f"{type(out).__name__}, not a bare hidden-states tensor. The "
        "boundary reduce-scatter has no defined place to run; refusing "
        "rather than dropping part of the output."
    )



def quantile_requires_ep() -> None:
    """Quantile balancing is installed by the EP swap, which ep=1 never runs -- there is
    no hpmesh MoE to balance.
    """
    raise UnsupportedCombinationError(
        "moe_quantile_balancing is installed by the EP swap, which "
        "ep=1 never runs -- there is no hpmesh MoE to balance. Run "
        "with ep > 1 to use it."
    )



def ptrr_load_balancer_backstop() -> None:
    """Assembly-time backstop for the config-phase 'ptrr_load_balancer' row: a caller
    that bypasses the config still hits the same refusal.
    """
    raise UnsupportedCombinationError(
        "'ptrr' load balancing builds its schedule from a BlockMask and is "
        "not wired in hpmesh yet; use 'headtail' or None."
    )



# == probe phase: verdicts called from the EP swap's layout probes =======


def gpt_oss_layout(experts: object) -> None:
    """GPT-OSS carries per-expert bias vectors, a transposed (E, D, 2F) layout, and a
    hardcoded clamped sigmoid-GLU activation; hpmesh's GroupedExperts has no slot
    for them. Unlock: a bias-bearing expert module with its own activation seam.
    """
    raise UnsupportedCombinationError(
        f"{type(experts).__name__} carries per-expert bias vectors, which "
        "hpmesh's GroupedExperts has no slot for. Only GPT-OSS has them, "
        "and it differs further: its gate_up_proj is transposed to "
        "(E, D, 2F) and its activation is a hardcoded clamped sigmoid-GLU "
        "rather than a module. Support needs a bias-bearing expert module "
        "with its own activation seam, not a wider copy here."
    )



def group_limited_greedy() -> None:
    """DeepSeek-V2's group_limited_greedy scores a group by its single best expert
    (max); the implemented rule sums the group's top-2 (DeepSeek-V3/GLM4), and
    routing with the wrong rule picks different experts. Unlock: a group-scoring
    option in TokenChoiceTopKRouter.
    """
    raise UnsupportedCombinationError(
        "DeepSeek-V2's group_limited_greedy scores a group by its single "
        "best expert (max); the implemented rule sums the group's top-2 "
        "(DeepSeek-V3/GLM4). Routing this checkpoint with that rule picks "
        "different experts, so refusing is the point -- add a group-scoring "
        "option to TokenChoiceTopKRouter to support it."
    )



def router_bias(router_gate: object) -> None:
    """RouterGateLinear has no slot for a router bias. Every supported family is bias-
    free, so this fires only on a family the probe does not know.
    """
    raise UnsupportedCombinationError(
        f"{type(router_gate).__name__} carries a router bias, which "
        "RouterGateLinear has no slot for. Every supported family "
        "(Qwen3Moe, OLMoE, Mixtral, DeepSeek-V2/V3, GLM4) is bias-free, "
        "so this fires only on a family the probe does not know."
    )



def quantile_requires_sigmoid(score_func: str, block: object) -> None:
    """The quantile scheme is defined over sigmoid scores (the histogram range derives
    from their [0, 1] bound).
    """
    raise UnsupportedCombinationError(
        f"quantile-balanced routing requires sigmoid router scores, "
        f"got {score_func!r} for {type(block).__name__}."
    )



def quantile_no_group_limit(block: object) -> None:
    """Quantile-balanced routing selects a free Top-(K+1) over all experts; group-
    limited routing is incompatible with it (a single group is no restriction and
    is accepted).
    """
    raise UnsupportedCombinationError(
        f"quantile-balanced routing selects a free Top-(K+1) over all "
        f"experts; {type(block).__name__}'s group-limited routing is "
        "incompatible with it. (A single group is no restriction and "
        "is accepted.)"
    )



def shared_expert_gate(block: object) -> None:
    """Qwen2Moe's shared_expert_gate multiplies where MoE.shared_experts only adds.
    """
    raise UnsupportedCombinationError(
        f"{type(block).__name__} gates its shared expert "
        "(shared_expert_gate); MoE's shared_experts is additive only. "
        "Qwen3Moe has no shared expert, so this is unreachable there."
    )



def shared_expert_tp_ep(block: object) -> None:
    """tp x ep over a shared-expert block: the TP plan shards it with the dense
    realizers, and composing those with the swapped MoE's sequence-sharded
    dispatch layout is unverified.
    """
    raise UnsupportedCombinationError(
        f"tp x ep over {type(block).__name__}: the block has a shared "
        "expert, which the TP plan shards with the dense colwise/rowwise "
        "realizers. Composing those with the swapped MoE's sequence-"
        "sharded dispatch layout is unverified; run shared-expert models "
        "with tp=1 (EP handles the shared expert) or ep=1."
    )



# == the table ====================================================================


ENTRIES: tuple[Row, ...] = (
    Row(sequence_parallel_required, "config", UnsupportedCombinationError,
        'config/parallel.py::__post_init__', scope='parallel'),
    Row(tp_ep_cp, "config", UnsupportedCombinationError,
        'config/parallel.py::__post_init__', scope='parallel'),
    Row(deepep_hybridep, "config", EnvironmentUnsupportedError,
        'config/parallel.py::__post_init__', scope='parallel'),
    Row(dispatcher_requires_ep, "config", UnsupportedCombinationError,
        'config/parallel.py::__post_init__', scope='parallel'),
    Row(ptrr_load_balancer, "config", UnsupportedCombinationError,
        'config/parallel.py::__post_init__ '
        '(backstop: context_parallel/input_shard.py)', scope='parallel'),
    Row(ulysses_no_load_balancer, "config", UnsupportedCombinationError,
        'config/parallel.py::__post_init__', scope='parallel'),
    Row(region_ac, "config", EnvironmentUnsupportedError,
        'config/training.py::TrainingConfig.__post_init__ '
        '(backstop: parallel/activation_checkpoint.py::apply_ac)', scope='training'),
    Row(memory_budget_requires_compile, "config", ConfigError,
        'config/training.py::TrainingConfig.__post_init__', scope='training'),
    Row(cp_divides_seq_len, "config", ConfigError,
        'config/root.py::HybridMeshConfig.__post_init__', scope='root'),
    Row(async_tp_requires_compile, "config", ConfigError,
        'config/root.py::HybridMeshConfig.__post_init__', scope='root'),
    Row(async_tp_requires_tp, "config", ConfigError,
        'config/root.py::HybridMeshConfig.__post_init__', scope='root'),
    Row(pp_activation_checkpoint, "assembly", UnsupportedCombinationError,
        'parallel/parallelize.py::parallelize_hf_transformers'),
    Row(pp_validation, "assembly", UnsupportedCombinationError,
        'trainer/validate.py::_check_validation_feasibility'),
    Row(validation_once_requires_dp1, "assembly", ConfigError,
        'trainer/validate.py::_check_validation_feasibility'),
    Row(validation_once_requires_finite_corpus, "assembly", ConfigError,
        'trainer/validate.py::_check_validation_feasibility'),
    Row(ep_checkpoint, "assembly", UnsupportedCombinationError,
        'trainer/trainer.py::Trainer.__init__'),
    Row(chunked_loss_pp, "assembly", UnsupportedCombinationError,
        'trainer/trainer.py::Trainer.__init__'),
    Row(pp_cp_ep, "assembly", UnsupportedCombinationError,
        'parallel/pipeline_parallel/apply.py::apply_pp'),
    Row(pp_real_corpus, "assembly", UnsupportedCombinationError,
        'parallel/pipeline_parallel/apply.py::apply_pp'),
    Row(pp_weight_tying, "assembly", UnsupportedCombinationError,
        'parallel/pipeline_parallel/apply.py::apply_pp'),
    Row(shared_expert_tp, "assembly", UnsupportedCombinationError,
        'parallel/tensor_parallel/apply.py::apply_tp'),
    Row(tp_moe_specs_without_block, "assembly", UnsupportedCombinationError,
        'parallel/tensor_parallel/apply.py::apply_tp'),
    Row(tp_moe_non_tensor_output, "assembly", UnsupportedCombinationError,
        'parallel/tensor_parallel/tp.py::_TPMoeSequenceBoundary.forward'),
    Row(quantile_requires_ep, "assembly", UnsupportedCombinationError,
        'parallel/expert_parallel/apply.py::apply_ep'),
    Row(ptrr_load_balancer_backstop, "assembly", UnsupportedCombinationError,
        'parallel/context_parallel/input_shard.py::_cp_load_balancer'),
    Row(gpt_oss_layout, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/probe.py::_fused_experts_of'),
    Row(group_limited_greedy, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/probe.py::_read_expert_groups'),
    Row(router_bias, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/convert.py::_convert_block'),
    Row(quantile_requires_sigmoid, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/convert.py::_convert_block'),
    Row(quantile_no_group_limit, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/convert.py::_convert_block'),
    Row(shared_expert_gate, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/convert.py::_convert_block'),
    Row(shared_expert_tp_ep, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/convert.py::_convert_block'),
)


def check_config(parallel) -> None:
    """Every config-phase row over a ``ParallelConfig``, in table order."""
    for row in ENTRIES:
        if row.phase == "config" and row.scope == "parallel":
            row.fn(parallel)


def check_training(training) -> None:
    """Every config-phase row over a ``TrainingConfig``, in table order."""
    for row in ENTRIES:
        if row.phase == "config" and row.scope == "training":
            row.fn(training)


def check_root(root) -> None:
    """Every config-phase row over a ``HybridMeshConfig``, in table order."""
    for row in ENTRIES:
        if row.phase == "config" and row.scope == "root":
            row.fn(root)
