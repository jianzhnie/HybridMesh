"""Cross-layer combination verdicts: the single source for assembly/probe guards.

The support matrix's scope is the combinations that need MORE than the config
to decide -- assembly time (the model, the resolved ``ParallelDims``, the
dataset name) and probe time (the HF model's layout, via the EP swap's
duck-typed probes). Each is a plain function below plus one row in the
``ENTRIES`` table at the bottom: the function carries the verdict (exception
type, exact message) and the rationale (its docstring); the row records the
phase and the guard's location. The trigger condition stays at the guard
site; the verdict lives here, so the site cannot quietly disagree.

Config-phase combination checks are NOT here: they live in the owning
config's ``__post_init__`` (``config/parallel.py``, ``config/training.py``,
``config/root.py``), alongside every other field validation. The division of
labor is documented in docs/hybridmesh_design.md (support-boundary section).

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
    UnsupportedCombinationError,
)

__all__ = [
    "ENTRIES",
    "Row",
]

Phase = Literal["config", "assembly", "probe"]


@dataclass(frozen=True)
class Row:
    """One matrix row. ``name``/``reason`` derive from the function."""

    fn: Callable
    phase: Phase
    error: type[Exception]
    guard: str

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def reason(self) -> str:
        return (self.fn.__doc__ or "").strip()


# == config phase: ParallelConfig ========================================




















# == config phase: TrainingConfig ========================================








# == config phase: HybridMeshConfig cross-group =======================











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
    Row(pp_activation_checkpoint, "assembly", UnsupportedCombinationError,
        'parallel/parallelize.py::parallelize_hf_transformers'),
    Row(pp_validation, "assembly", UnsupportedCombinationError,
        'trainer/validate.py::check_validation_feasibility'),
    Row(validation_once_requires_dp1, "assembly", ConfigError,
        'trainer/validate.py::check_validation_feasibility'),
    Row(validation_once_requires_finite_corpus, "assembly", ConfigError,
        'trainer/validate.py::check_validation_feasibility'),
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
        'parallel/tensor_parallel/tp.py::TPMoeSequenceBoundary.forward'),
    Row(quantile_requires_ep, "assembly", UnsupportedCombinationError,
        'parallel/expert_parallel/apply.py::apply_ep'),
    Row(ptrr_load_balancer_backstop, "assembly", UnsupportedCombinationError,
        'parallel/context_parallel/input_shard.py::_cp_load_balancer'),
    Row(gpt_oss_layout, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/probe.py::fused_experts_of'),
    Row(group_limited_greedy, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/probe.py::read_expert_groups'),
    Row(router_bias, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/convert.py::convert_block'),
    Row(quantile_requires_sigmoid, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/convert.py::convert_block'),
    Row(quantile_no_group_limit, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/convert.py::convert_block'),
    Row(shared_expert_gate, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/convert.py::convert_block'),
    Row(shared_expert_tp_ep, "probe", UnsupportedCombinationError,
        'parallel/expert_parallel/convert.py::convert_block'),
)
