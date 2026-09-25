"""Parallelism sizes config (TP/CP/EP/PP/FSDP and related validation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


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

    Not composable with context_parallel_size > 1 when tensor_parallel_size >
    1 (rejected in ``__post_init__``, the tp x ep x cp combination is
    unverified). tp x ep itself IS supported: TP shards only the dense parts
    and EP owns the routed experts.
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

    ep_token_dispatcher: str = "alltoall"
    """
    EP dispatch backend the MoE swap installs. Options:

    - "alltoall": default. ``AllToAllTokenDispatcher`` -- local reorder plus
      all-to-all collectives over the EP group.
    - "torchao": ``TorchAOTokenDispatcher``, same dispatch but each local
      expert's token group is padded to ``ep_torchao_pad_multiple`` for
      FP8/MXFP8 quantized grouped GEMMs. Requires the optional ``torchao``
      package; constructing the dispatcher without it raises ImportError
      with an install hint. Numerics unverified (no torchao/CUDA on the
      development machine), awaiting a CUDA-target re-run.
    - "deepep" / "hybridep": registered gaps, refused here with
      NotImplementedError. Both are CUDA-only and drive their kernels through
      torchtitan's ``distributed/deepep/`` wrappers, which are not vendored;
      ``alltoall`` satisfies the same dispatch/combine contract meanwhile.

    Requires expert_parallel_size > 1 (the EP swap is the only place a
    dispatcher is installed, and it does not run at ep=1).
    """

    ep_torchao_pad_multiple: int = 16
    """
    Token-group padding multiple for ``ep_token_dispatcher="torchao"``:
    16 for FP8, 32 for MXFP8 quantized grouped GEMM kernels. Ignored by the
    other backends.
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
        if (
            self.tensor_parallel_size > 1
            and self.expert_parallel_size > 1
            and self.context_parallel_size > 1
        ):
            # tp x ep is supported: TP shards the dense parts, EP owns the
            # routed experts (E-shard), the router stays replicated -- the
            # upstream alignment. Adding CP on top is unverified (the CP
            # attention kernel is orthogonal, but the token-count reductions
            # and dispatcher layouts have not been exercised together), so it
            # is refused here rather than silently composing.
            raise NotImplementedError(
                "tensor_parallel_size > 1 with expert_parallel_size > 1 and "
                "context_parallel_size > 1 is not supported: tp x ep x cp is "
                "unverified. Run tp x ep with context_parallel_size=1, or ep x "
                "cp with tensor_parallel_size=1."
            )
        if self.data_parallel_shard_size < 1 and self.data_parallel_shard_size != -1:
            raise ValueError(
                "data_parallel_shard_size must be >= 1 or -1 (derive), got "
                f"{self.data_parallel_shard_size}"
            )
        allowed_dispatchers = ("alltoall", "torchao", "deepep", "hybridep")
        if self.ep_token_dispatcher not in allowed_dispatchers:
            raise ValueError(
                "parallelism.ep_token_dispatcher must be one of: "
                f"{allowed_dispatchers} (got {self.ep_token_dispatcher!r})"
            )
        if self.ep_token_dispatcher in ("deepep", "hybridep"):
            # Registered gap, not a typo: both backends are CUDA-only (they
            # cannot even be installed on a CPU/NPU box) and their dispatch /
            # combine drive torchtitan's `distributed/deepep/` wrappers around
            # the deep_ep / hybridep kernels, which hpmesh does not vendor.
            # Porting them means vendoring those wrappers AND validating on a
            # CUDA target device; neither is done, so refuse the configuration
            # rather than silently run all-to-all. The default 'alltoall'
            # dispatcher satisfies the same dispatch/combine contract.
            raise NotImplementedError(
                f"ep_token_dispatcher={self.ep_token_dispatcher!r} is a "
                "registered gap, not a supported backend: it is CUDA-only and "
                "requires the deep_ep/hybridep kernels plus torchtitan's "
                "distributed/deepep/ wrappers, which hpmesh does not vendor "
                "(environment not covered; see docs/hpmesh_upstream_map.md "
                "table D). Unlock conditions: vendor the wrappers, add the "
                "CUDA-only dependency as an optional extra, and re-validate "
                "numerics on a CUDA device. Use 'alltoall' meanwhile."
            )
        if self.ep_token_dispatcher != "alltoall" and self.expert_parallel_size == 1:
            raise NotImplementedError(
                f"ep_token_dispatcher={self.ep_token_dispatcher!r} has no "
                "effect at expert_parallel_size=1: the EP swap is the only "
                "place a token dispatcher is installed and it does not run at "
                "ep=1. Set expert_parallel_size > 1, or keep 'alltoall'."
            )
        if self.ep_torchao_pad_multiple < 1:
            raise ValueError(
                "ep_torchao_pad_multiple must be >= 1, got "
                f"{self.ep_torchao_pad_multiple}"
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
