"""Route tokens to experts: reorder locally, all-to-all across EP ranks.

Vendored from torchtitan ``models/common/token_dispatcher.py``. The ``Local`` and
``AllToAll`` dispatchers came across; of the three optional backends:

* ``TorchAOTokenDispatcher`` is ported as an optional-import adapter: it varies
  from ``AllToAllTokenDispatcher`` only in its ``_permute``/``_unpermute``,
  which delegate to torchao's ``permute_and_pad``. torchao is not a dependency;
  constructing the dispatcher without it raises ``ImportError`` with an install
  hint. Its padded-permute numerics are unverified on this (macOS/CPU) machine
  and await a re-run on the CUDA target with the real package.
* ``DeepEPTokenDispatcher`` and ``HybridEPTokenDispatcher`` stay registered
  gaps. They drive DeepEP v2's ``ElasticBuffer`` and HybridEP's kernels through
  torchtitan's ``distributed/deepep/`` wrappers (1155 lines) around the
  CUDA-only ``deep_ep``/``hybridep`` packages; the dispatch/combine surface
  cannot be expressed faithfully without vendoring those wrappers, so selecting
  either backend is refused at config time (``ParallelConfig``) with the unlock
  conditions spelled out.

None of that is a functional gap: none of the backends changes the routing
contract, only how the tokens cross ranks. The dispatch/combine/metadata
interface below is the whole contract, and ``AllToAllTokenDispatcher``
implements it.

What changed from upstream, and why:

* Dispatchers are constructed directly from plain ints; there is no nested
  ``Config`` in between.
* ``spmd.all_to_all`` is gone in favour of ``all_to_all_single``. Upstream
  already writes that call as the compiled/traced branch; it is the native
  collective, it needs no ``spmd_types``, and it is the branch that actually
  runs on a non-NCCL backend. The eager path now uses it too.
* every ``spmd.is_type_checking()`` block is deleted -- they annotate the SPMD
  type checker and have no runtime effect.
* ``maybe_set_sparse_mesh()`` is gone. It swaps the ambient mesh while the
  expert body runs; hpmesh passes process groups explicitly instead.
* the mesh is a plain ``ProcessGroup``, not a ``DeviceMesh``, because all this
  code does with it is launch collectives.

The two dispatchers differ in one way: ``LocalTokenDispatcher`` only reorders
tokens (EP=1), while ``AllToAllTokenDispatcher`` additionally moves them between
ranks (EP>1). The EP one falls back to the local path when it has no group, so
the EP=1 code path is literally the same code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single

from ...accelerator import dist_utils
from .scatter_add import deterministic_scatter_add

__all__ = [
    "EP_DISPATCHER_BACKENDS",
    "TORCHAO_INSTALL_HINT",
    "LocalDispatchMetadata",
    "AllToAllDispatchMetadata",
    "LocalTokenDispatcher",
    "AllToAllTokenDispatcher",
    "TorchAOTokenDispatcher",
]

#: EP dispatch backends selectable via ``ParallelConfig.ep_token_dispatcher``.
#: ``alltoall`` is the default; ``torchao`` is an optional-import adapter;
#: ``deepep``/``hybridep`` are registered gaps refused at config time.
EP_DISPATCHER_BACKENDS = ("alltoall", "torchao", "deepep", "hybridep")

TORCHAO_INSTALL_HINT = (
    "ep_token_dispatcher='torchao' requires the optional `torchao` package, "
    "which is not installed. Install it with `pip install torchao`, or use "
    "the default ep_token_dispatcher='alltoall'. torchao's `permute_and_pad` "
    "pads each expert's token group to a multiple of pad_multiple for "
    "FP8/MXFP8 quantized grouped GEMMs."
)


def _materialize(tensor: torch.Tensor) -> torch.Tensor:
    """Force an async collective result to be ready.

    ``all_to_all_single`` returns immediately, leaving the transfer running; the
    count exchange's output is needed on the host (to build split-size lists)
    before the data all-to-all can be launched, so it must be waited on first.
    """
    wait = getattr(tensor, "wait", None)
    return wait() if callable(wait) else tensor


@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata:
    """What ``LocalTokenDispatcher.dispatch()`` hands to ``combine()``."""

    token_indices_experts_sorted_N: torch.Tensor  # noqa: N815
    topk_scores_experts_sorted_N: torch.Tensor  # noqa: N815


@dataclass(frozen=True, kw_only=True)
class AllToAllDispatchMetadata(LocalDispatchMetadata):
    """The local metadata plus what the all-to-all needs to be reversed."""

    input_shape: tuple  # for _unpermute
    permuted_indices: torch.Tensor  # for _unpermute
    input_splits: list[int]
    output_splits: list[int]


class LocalTokenDispatcher:
    """Token dispatcher for EP=1: local reordering only.

    Not an ``nn.Module`` -- a dispatcher owns no parameters or buffers.

    Args:
        num_experts: total expert count (E).
        top_k: experts each token is routed to (K).
    """

    def __init__(self, num_experts: int, top_k: int) -> None:
        self.num_experts = num_experts
        self.top_k = top_k

    def wire_meshes(self, *, ep_group: dist.ProcessGroup | None) -> None:
        """No-op for the EP=1 dispatcher. Subclasses override."""
        del ep_group

    def _local_reorder(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder tokens by expert assignment for local expert computation.

        Groups tokens by expert index via a stable argsort, so within an expert
        the original token order is preserved. Routing scores are applied to the
        expert outputs in ``combine``, after expert computation.

        Args:
            x_TD: ``(T, D)`` input tokens.
            topk_scores_TK: ``(T, K)`` routing scores.
            topk_expert_ids_TK: ``(T, K)`` expert indices.

        Returns:
            routed_input_ND: ``(N, D)``, N = T*K, tokens in expert-sorted order.
            token_indices_experts_sorted_N: ``(N,)`` token-to-original mapping.
            topk_scores_experts_sorted_N: ``(N,)`` scores in expert-sorted order.
        """
        # N = T*K: one entry per (token, chosen expert) pair.
        token_indices_experts_sorted_N = torch.argsort(
            topk_expert_ids_TK.view(-1), stable=True
        )
        topk_scores_experts_sorted_N = topk_scores_TK.view(-1)[
            token_indices_experts_sorted_N
        ]
        token_indices_experts_sorted_N = token_indices_experts_sorted_N // self.top_k
        routed_input_ND = x_TD[token_indices_experts_sorted_N]

        return (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder tokens by expert assignment for local expert computation.

        Args:
            x_TD: ``(T, D)`` all input tokens.
            topk_scores_TK: ``(T, K)`` routing scores.
            topk_expert_ids_TK: ``(T, K)`` expert indices per token.
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert.

        Returns:
            routed_input_RD: ``(R, D)`` with ``R = sum(num_local_tokens_per_
                expert_E)``, tokens sorted by expert index.
            num_local_tokens_per_expert_E: ``(E,)``, unchanged.
            metadata: for ``combine()``.
        """
        # R = N: with no EP there is no all-to-all, so nothing is dropped.
        (
            routed_input_RD,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)
        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        )
        return routed_input_RD, num_local_tokens_per_expert_E, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Weight the expert outputs by their routing score and scatter them home.

        The score is applied here rather than in ``dispatch`` so it multiplies
        the expert output, which is what the routing weight means.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs.
            metadata: from ``dispatch()``.
            x_TD: ``(T, D)`` original input tokens.

        Returns:
            out_TD: ``(T, D)`` combined output.
        """
        out_TD = torch.zeros_like(x_TD)

        # fp32 for the multiply, then back: the score is a probability and the
        # expert output may be low precision.
        routed_output_RD = (
            routed_output_RD.to(torch.float32)
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(routed_output_RD.dtype)

        dim = x_TD.shape[-1]
        # A token routed to K experts appears K times at its own index, so this
        # scatter_add has duplicate indices -- hence the deterministic variant.
        out_TD = deterministic_scatter_add(
            out_TD,
            metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, dim),
            routed_output_RD,
        )
        return out_TD


class BaseEPTokenDispatcher(LocalTokenDispatcher, ABC):
    """Base for EP dispatchers: owns the EP group and its wiring.

    Args:
        num_experts: experts *per rank* (E / ep_size); this is the "local"
            count, since each rank holds only a slice of the expert set.
        top_k: experts each token is routed to.
    """

    def __init__(self, num_experts: int, top_k: int) -> None:
        super().__init__(num_experts, top_k)
        # ``None`` means EP is off and every method falls back to local.
        self.ep_group: dist.ProcessGroup | None = None

    def wire_meshes(self, *, ep_group: dist.ProcessGroup | None) -> None:
        """Install the EP group used by dispatch/combine."""
        self.ep_group = ep_group
        self.init_buffer()

    def init_buffer(self) -> None:
        """Initialize backend communication buffers, if any."""

    @abstractmethod
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Move each token to the rank that owns the expert it chose."""
        raise NotImplementedError

    @abstractmethod
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: object,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Return expert outputs to the tokens that asked for them."""
        raise NotImplementedError


class AllToAllTokenDispatcher(BaseEPTokenDispatcher):
    """EP>1: reorder locally, then all-to-all tokens to their experts' ranks.

    Dispatch moves a token to whichever rank holds the expert it picked;
    combine reverses that and scatters the results back. Both halves are the
    same collective with the split sizes swapped.
    """

    def _token_count_exchange(
        self,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Tell every rank how many tokens you are sending it per expert.

        Each rank contributes a ``(ep_size, e)`` block of counts; the all-to-all
        gathers one column-block from every rank, giving each rank the counts it
        will receive -- and, transposed, the counts it must send.

        Split sizes are omitted, so every rank sends an equal-size buffer: the
        blocks are all ``(ep_size, e)`` and therefore already aligned.
        """
        assert self.ep_group is not None
        ep_size = dist_utils.get_world_size(self.ep_group)
        return all_to_all_single(
            num_local_tokens_per_expert_E.view(ep_size, -1),
            None,
            None,
            group=self.ep_group,
        )

    def _sync_token_count_exchange(
        self,
        num_local_tokens_per_expert_E: torch.Tensor,
        num_global_tokens_per_local_expert_EP_e: torch.Tensor,
        ep_size: int,
    ) -> tuple[torch.Tensor, list[int], list[int]]:
        """Wait for the counts, then materialize the split-size lists.

        Remote counts (``output_splits``) must be known before the variable-size
        data all-to-all is launched; local counts are ours already and only need
        to reach the host.
        """
        num_global_tokens_per_local_expert_EP_e = _materialize(
            num_global_tokens_per_local_expert_EP_e
        )
        num_global_tokens_per_local_expert_E = (
            num_global_tokens_per_local_expert_EP_e.reshape(-1)
        )
        input_splits = (
            num_local_tokens_per_expert_E.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        # NOTE: this incurs a device-to-host sync.
        output_splits = (
            num_global_tokens_per_local_expert_E.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=False)
        )
        return (
            num_global_tokens_per_local_expert_E,
            input_splits.tolist(),
            output_splits.tolist(),
        )

    def _dispatch_token_exchange(
        self,
        routed_input_ND: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> torch.Tensor:
        """Send routed tokens to the ranks holding their chosen experts."""
        assert self.ep_group is not None
        return all_to_all_single(
            routed_input_ND,
            output_splits,
            input_splits,
            group=self.ep_group,
        )

    def _combine_token_exchange(
        self,
        routed_output_RD: torch.Tensor,
        input_splits: list[int],
        output_splits: list[int],
    ) -> torch.Tensor:
        """Send expert outputs back, reversing the dispatch exchange."""
        assert self.ep_group is not None
        # The split lists are swapped relative to dispatch: what was received
        # there is sent here.
        return all_to_all_single(
            routed_output_RD,
            input_splits,
            output_splits,
            group=self.ep_group,
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, AllToAllDispatchMetadata]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        Args:
            x_TD: ``(T, D)`` local token shard.
            topk_scores_TK: ``(T, K)`` routing scores.
            topk_expert_ids_TK: ``(T, K)`` expert indices.
            num_local_tokens_per_expert_E: ``(E,)`` counts for this shard.

        Returns:
            routed_input_RD: tokens in expert-major order for local experts.
            num_tokens_per_local_expert_e: ``(num_local_experts,)`` counts.
            metadata: for ``combine()``.
        """
        # EP=1: nothing crosses ranks, so defer to the local path.
        if self.ep_group is None:
            return LocalTokenDispatcher.dispatch(
                self,
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )

        ep_size = dist_utils.get_world_size(self.ep_group)
        # _local_reorder yields (N, D), N = T*K; the all-to-all below yields
        # (R, D) with R != N, since each rank ends up with only its own experts'
        # tokens.
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        with torch.no_grad():
            num_global_tokens_per_local_expert_EP_e = self._token_count_exchange(
                num_local_tokens_per_expert_E
            )
            (
                num_global_tokens_per_local_expert_E,
                input_splits_list,
                output_splits_list,
            ) = self._sync_token_count_exchange(
                num_local_tokens_per_expert_E,
                num_global_tokens_per_local_expert_EP_e,
                ep_size,
            )

        routed_input_RD = self._dispatch_token_exchange(
            routed_input_ND,
            output_splits_list,
            input_splits_list,
        )
        # The all-to-all delivers in rank-major order but the experts want
        # expert-major:
        #   (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...   (rank-major)
        #   (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...   (expert-major)
        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        ) = self._permute(
            routed_input_RD,
            num_global_tokens_per_local_expert_E,
        )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    def _permute(
        self,
        routed_input_RD: torch.Tensor,
        num_global_tokens_per_local_expert_E: torch.Tensor,
    ) -> tuple[tuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder received tokens from rank-major to expert-major layout.

        ``num_global_tokens_per_local_expert_E`` arrives as an ``(EP, e)`` count
        matrix; this builds the index map that regroups it by local expert.
        """
        assert self.ep_group is not None
        ep_size = dist_utils.get_world_size(self.ep_group)
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size
        device = num_global_tokens_per_local_expert_E.device
        total = routed_input_RD.shape[0]

        t_mat = num_global_tokens_per_local_expert_E.view(ep_size, e)

        # Where each (r, e) segment starts in the input (rank-major order).
        input_starts = (
            num_global_tokens_per_local_expert_E.cumsum(0)
            - num_global_tokens_per_local_expert_E
        ).view(ep_size, e)

        # Transpose to expert-major (e, EP) and flatten.
        segment_lens = t_mat.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)

        # output[p] = input[input_starts[seg] + (p - output_starts[seg])].
        seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(
            segment_lens, output_size=total
        )
        output_starts = segment_lens.cumsum(0) - segment_lens
        permuted_indices = (
            input_starts[seg_ids]
            + torch.arange(seg_ids.shape[0], device=device)
            - output_starts[seg_ids]
        )

        num_global_tokens_per_local_expert_e = t_mat.sum(0)
        return (
            routed_input_RD.shape,
            routed_input_RD[permuted_indices, :],
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        )

    def _unpermute(
        self,
        routed_output_RD: torch.Tensor,
        input_shape: tuple,
        permuted_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the expert-major reordering."""
        out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
        out_unpermuted_RD[permuted_indices, :] = routed_output_RD
        return out_unpermuted_RD

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Unpermute, all-to-all back, weight by score, and scatter home.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs in expert-major order.
            metadata: from ``dispatch()``.
            x_TD: ``(T, D)`` original input tokens.

        Returns:
            out_TD: ``(T, D)`` combined local output.
        """
        # EP=1: nothing to send back.
        if self.ep_group is None:
            return LocalTokenDispatcher.combine(
                self,
                routed_output_RD,
                metadata,
                x_TD,
            )

        routed_output_RD = self._unpermute(
            routed_output_RD, metadata.input_shape, metadata.permuted_indices
        )
        routed_output_RD = self._combine_token_exchange(
            routed_output_RD,
            metadata.input_splits,
            metadata.output_splits,
        )

        # The tokens are back on their home ranks; weighting by score and
        # scattering home is exactly the local dispatcher's combine.
        return LocalTokenDispatcher.combine(
            self,
            _materialize(routed_output_RD),
            metadata,
            x_TD,
        )


class TorchAOTokenDispatcher(AllToAllTokenDispatcher):
    """All-to-all dispatch with token-group padding for quantized grouped GEMMs.

    Identical to ``AllToAllTokenDispatcher`` except ``_permute``/``_unpermute``
    delegate to torchao's ``permute_and_pad``, which reorders tokens to
    expert-major order AND pads each expert's token group to a multiple of
    ``pad_multiple`` -- the alignment FP8/MXFP8 quantized grouped GEMM kernels
    require (16 for FP8, 32 for MXFP8). The padding shows up in
    ``num_tokens_per_local_expert_e`` and must be stripped by the matching
    ``_unpermute``, so a for-loop ``GroupedExperts`` that does not expect
    padding cannot be paired with it.

    Invariants (unchanged from the all-to-all dispatcher):

    * ``dispatch(x_TD(T, D), topk_scores_TK(T, K), topk_expert_ids_TK(T, K),
      num_local_tokens_per_expert_E(E,))`` returns tokens in expert-major
      order for the local experts, per-local-expert counts (padded), and
      metadata consumed only by ``combine``.
    * ``combine`` inverts dispatch exactly: unpermute (strip padding),
      all-to-all back, weight by score, deterministic scatter home.

    torchao is an optional dependency: constructing this class without it
    raises ``ImportError`` with an install hint. With ``ep_group=None`` it
    skips the all-to-all and applies only the local padded permute (the EP=1
    debug/numerics path upstream supports).

    The padded-permute numerics are unverified on this (macOS/CPU) machine --
    no torchao, no CUDA -- and await a re-run on the CUDA target with the real
    package installed.
    """

    def __init__(self, num_experts: int, top_k: int, pad_multiple: int) -> None:
        super().__init__(num_experts, top_k)
        if pad_multiple < 1:
            raise ValueError(f"pad_multiple must be >= 1, got {pad_multiple}.")
        self.pad_multiple = pad_multiple
        try:
            from torchao.prototype.moe_training.ep.permute import permute_and_pad
        except ImportError as exc:
            raise ImportError(TORCHAO_INSTALL_HINT) from exc
        self._permute_and_pad = permute_and_pad

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, AllToAllDispatchMetadata]:
        """Dispatch, padding each local expert's token group to ``pad_multiple``."""
        if self.ep_group is not None:
            return super().dispatch(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )

        # EP=1: no all-to-all. Locally reorder to expert-sorted order, then
        # apply the padded permute so the quantized grouped GEMM sees groups
        # aligned to pad_multiple.
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_tokens_per_local_expert_padded_e,
        ) = self._permute(routed_input_ND, num_local_tokens_per_expert_E)

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            # Unused in the EP=1 combine path (no all-to-all to reverse).
            input_splits=[],
            output_splits=[],
        )
        return routed_input_RD, num_tokens_per_local_expert_padded_e, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Strip the padding, then run the local score + scatter home."""
        if self.ep_group is not None:
            return super().combine(routed_output_RD, metadata, x_TD)

        # EP=1: _unpermute removes the padding and recovers expert-sorted
        # order; the rest is exactly the local dispatcher's combine.
        assert isinstance(metadata, AllToAllDispatchMetadata)
        routed_output_RD = self._unpermute(
            routed_output_RD, metadata.input_shape, metadata.permuted_indices
        )
        return LocalTokenDispatcher.combine(self, routed_output_RD, metadata, x_TD)

    def _permute(
        self,
        routed_input_RD: torch.Tensor,
        num_global_tokens_per_local_expert_E: torch.Tensor,
    ) -> tuple[tuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Padded expert-major permute via torchao's ``permute_and_pad``."""
        # ep_size=1 when EP is disabled: permute_and_pad then only pads token
        # groups (rank-major == expert-major for a single rank).
        ep_size = (
            1
            if self.ep_group is None
            else dist_utils.get_world_size(self.ep_group)
        )
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size

        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_padded_e,
            _group_offsets,
        ) = self._permute_and_pad(
            routed_input_RD,
            num_global_tokens_per_local_expert_E,
            ep_size,
            e,
            self.pad_multiple,
        )
        return (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_padded_e,
        )

    def _unpermute(
        self,
        routed_output_RD: torch.Tensor,
        input_shape: tuple,
        permuted_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the padded permute and strip the sentinel row it added."""
        return super()._unpermute(routed_output_RD, input_shape, permuted_indices)[:-1]
