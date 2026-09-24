"""The CP flex kernel: redistribute q/k/v across the CP group, then run flex.

``apply_cp`` attaches one of these to every decoder layer's attention module
as ``_titan_flex_kernel``; ``hf_wrapper._flex_attention_hf`` then routes the
layer's attention call through it. q/k/v arrive HF-shaped --
``(batch, heads, seq, dim)`` -- with the sequence already sharded along dim 2
by ``shard_batch_for_cp``. Two strategies redistribute them:

* ``"kv_allgather"`` all-gathers K/V along the sequence dim, so flex runs the
  local query shard against the full-length keys. Q and the output stay
  token-sharded.
* ``"ulysses"`` all-to-all's every input from a token shard into a head shard
  -- ``(b, h, s/cp, d) -> (b, h/cp, s, d)`` -- runs flex on the full sequence
  with ``heads / cp`` heads, then all-to-all's the output back. Compute and
  memory both shard, at the cost of two all-to-alls per layer.

Neither redistribution depends on an ambient SPMD context. The kv_allgather
gather is torch's own ``flex_cp_allgather`` custom op (the one torchtitan's
HF-backend ``_wrap_flex_kernel_cp`` uses), held by process-group name; the
ulysses all-to-alls drive ``all_to_all_single`` on the group itself. Both are
captured at attach time.

Masks under ulysses: attention runs on the FULL sequence, so a Q-sharded
BlockMask does not apply -- its Q axis is the local shard's length. Two mask
shapes therefore reach the kernel. For a single causal document the wrapper
hands over its Q-sharded CP mask (it builds one mask shape for every CP
strategy) and the kernel rebuilds the full-length causal mask itself -- exact
because that mask is a function of the sequence length alone. For a packed
corpus (``block_causal``) the wrapper hands over the full-length document mask
UNSHARDED: the all-to-all reassembles the whole token stream on every rank
before attention, so the document structure (the varlen metadata: which tokens
share a document) needs no sharding -- the same reason upstream's ulysses
``cp_shard`` lifts ``attention_masks`` out of the sharded inputs and reattaches
it untouched. The kernel tells the two apart by the mask's Q length (see
:meth:`CPFlexKernel._forward_ulysses`).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from hpmesh.models.common.masks import create_attention_mask, get_causal_mask_mod
from hpmesh.utils.batch_invariant import is_in_batch_invariant_mode

__all__ = ["CPFlexKernel"]

# q/k/v arrive HF-shaped: (batch, heads, seq, dim). The sequence axis the CP
# input sharding split is dim 2 of this layout.
_SEQ_DIM = 2
_HEAD_DIM = 1

_KNOWN_STRATEGIES = ("kv_allgather", "ulysses")


def _cp_all_to_all(
    x_BHSD: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """One CP all-to-all: split ``scatter_dim`` across the group, join ``gather_dim``.

    Chunk ``r`` of the scatter dim goes to rank ``r``; the pieces that come
    back are concatenated along the gather dim in rank order. With
    ``scatter_dim=1, gather_dim=2`` this is the ulysses token-to-head swap
    ``(b, h, s/cp, d) -> (b, h/cp, s, d)``; swapping the two dims is the exact
    inverse, which is also each direction's backward.
    """
    cp = group.size()
    shape = x_BHSD.shape
    if shape[scatter_dim] % cp != 0:
        raise ValueError(
            f"Ulysses all-to-all scatters dim {scatter_dim} of size "
            f"{shape[scatter_dim]} across cp={cp} ranks, which does not divide "
            "evenly. For the head axis this means the model's head count must "
            "be a multiple of the CP degree."
        )
    split = (
        shape[:scatter_dim] + (cp, shape[scatter_dim] // cp) + shape[scatter_dim + 1 :]
    )
    # Chunk r of the scatter dim is bound for rank r, so it leads.
    x = x_BHSD.reshape(split).movedim(scatter_dim, 0).contiguous()
    y = torch.empty_like(x)
    dist.all_to_all_single(y, x, group=group)
    # y[r] is rank r's piece; merge the group axis into the gather dim so the
    # pieces concatenate in rank order.
    y = y.movedim(0, gather_dim)
    merged = (
        y.shape[:gather_dim]
        + (y.shape[gather_dim] * y.shape[gather_dim + 1],)
        + y.shape[gather_dim + 2 :]
    )
    return y.reshape(merged)


def _reject_unrepresentable_attention_kwargs(kwargs: dict) -> None:
    """Refuse attention modifiers the CPU eager fallback cannot express.

    The CUDA branch hands every kwarg to HF's ``flex_attention_forward``, whose
    ``score_mod`` applies ``softcap`` and its post-hoc ``s_aux`` renormalization.
    The CPU branch cannot: it calls ``torch.nn.attention.flex_attention.
    flex_attention`` directly, whose signature has no ``softcap`` at all
    (``score_mod``/``block_mask``/``scale``/``enable_gqa``/``return_lse``/
    ``kernel_options``/``return_aux``), so there is nothing to forward to and the
    modification would be silently dropped -- a different model, computed
    confidently.

    Neither is reachable through the flags hpmesh reads today: nothing sets
    ``attn_logit_softcapping`` (a Gemma-2/3 config field) or an attention-sink
    ``s_aux``. That is exactly why this is a refusal and not a fallback: the
    failure would be silent, and silence is indistinguishable from "this model
    has no softcap".
    """
    blocking = sorted(
        name for name in ("softcap", "s_aux") if kwargs.get(name) is not None
    )
    if blocking:
        raise NotImplementedError(
            f"CP's CPU eager attention fallback cannot represent {blocking}: "
            "torch's flex_attention has no softcap/attention-sink parameter, so "
            "the CUDA branch's HF score_mod would not be reproduced here and the "
            "attention would silently compute a different model. Run this model "
            "on CUDA (where HF's flex_attention_forward applies them), or set "
            f"cp=1. Got {[(k, kwargs[k]) for k in blocking]}."
        )


def _run_flex(module, q, k, v, block_mask, kwargs) -> torch.Tensor:
    """Run flex attention on already-redistributed q/k/v.

    Returns the output in ``(batch, heads, seq, dim)`` on both backends. On
    CUDA this goes through HF's ``flex_attention_forward``, whose ``score_mod``
    applies ``softcap``/``s_aux``, and whose ``(batch, seq, heads, dim)``
    output is transposed back. On CPU, transformers routes flex through
    torch.compile, whose inductor flex lowering has no CPU target (this is why
    the wrapper picks sdpa off CUDA), so this runs torch's eager
    ``flex_attention`` instead -- the fallback exists so CP is exercisable on
    CPU-only machines, e.g. the gloo equivalence tests -- after refusing the
    attention kwargs it cannot express.
    """
    if q.is_cuda:
        from transformers.integrations.flex_attention import (
            flex_attention_forward,
        )

        out, _ = flex_attention_forward(module, q, k, v, block_mask, **kwargs)
        return out.transpose(1, 2)
    _reject_unrepresentable_attention_kwargs(kwargs)
    from torch.nn.attention.flex_attention import flex_attention

    return flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=kwargs.get("scaling"),
        enable_gqa=True,
    )


class _SeqToHead(torch.autograd.Function):
    """``(b, h, s/cp, d) -> (b, h/cp, s, d)``; the backward is the inverse swap."""

    @staticmethod
    def forward(ctx, x_BHSD, group):
        ctx.group = group
        return _cp_all_to_all(x_BHSD, group, scatter_dim=_HEAD_DIM, gather_dim=_SEQ_DIM)

    @staticmethod
    def backward(ctx, grad_BHSD):
        return (
            _cp_all_to_all(
                grad_BHSD, ctx.group, scatter_dim=_SEQ_DIM, gather_dim=_HEAD_DIM
            ),
            None,
        )


class _HeadToSeq(torch.autograd.Function):
    """``(b, h/cp, s, d) -> (b, h, s/cp, d)``; the backward is the inverse swap."""

    @staticmethod
    def forward(ctx, x_BHSD, group):
        ctx.group = group
        return _cp_all_to_all(x_BHSD, group, scatter_dim=_SEQ_DIM, gather_dim=_HEAD_DIM)

    @staticmethod
    def backward(ctx, grad_BHSD):
        return (
            _cp_all_to_all(
                grad_BHSD, ctx.group, scatter_dim=_HEAD_DIM, gather_dim=_SEQ_DIM
            ),
            None,
        )


class CPFlexKernel(nn.Module):
    """Flex attention with q/k/v redistributed across the CP group.

    Args:
        cp_mesh: the CP axis mesh. Its process group is captured at
            construction, so attaching the kernel is also when a missing or
            mis-sized CP axis fails.
        strategy: ``"kv_allgather"`` (all-gather K/V, Q stays token-sharded) or
            ``"ulysses"`` (all-to-all onto the head axis, attention runs
            full-length with ``heads / cp`` heads per rank).
    """

    def __init__(self, *, cp_mesh: DeviceMesh, strategy: str = "kv_allgather") -> None:
        super().__init__()
        if strategy not in _KNOWN_STRATEGIES:
            raise NotImplementedError(
                f"CP strategy {strategy!r} is not wired; supported strategies "
                f"are {_KNOWN_STRATEGIES}."
            )
        self.strategy = strategy
        self._cp_group = cp_mesh.get_group()
        if strategy == "kv_allgather":
            try:
                from torch.distributed.tensor.experimental._context_parallel._cp_custom_ops import (  # noqa: E501
                    flex_cp_allgather,
                )
            except ImportError as e:  # pragma: no cover - torch version guard
                raise ImportError(
                    "CPFlexKernel relies on torch's private ``flex_cp_allgather`` "
                    "(torch.distributed.tensor.experimental._context_parallel."
                    "_cp_custom_ops), which this torch build does not provide. It "
                    "is present in torch 2.6+; upgrade torch, or set cp=1."
                ) from e
            self._flex_cp_allgather = flex_cp_allgather
            self._cp_pg_name = c10d._get_process_group_name(cp_mesh.get_group())
        # Ulysses rebuilds the full-length causal mask per forward (see the
        # module docstring); cache it per (length, device, batch-invariant
        # mode). The cache lives on the kernel instance -- one per attention
        # layer -- so the rebuild happens once per length per layer, not once
        # per forward.
        self._full_masks: dict = {}

    def forward(self, query, key, value, *, module, block_mask=None, **kwargs):
        """Redistribute q/k/v per the strategy, run flex, redistribute back.

        For ``kv_allgather``, ``block_mask`` is the Q-sharded BlockMask (local
        Q, full KV) that pairs with the gathered K/V; for ``ulysses`` it is
        either dropped and rebuilt (single causal document) or used as-is
        (packed corpus, full-length -- see :meth:`_forward_ulysses`). Returns
        just the attention output tensor -- ``hf_wrapper._flex_attention_hf``
        appends the ``None`` LSE itself.
        """
        if self.strategy == "ulysses":
            return self._forward_ulysses(
                query, key, value, module=module, block_mask=block_mask, **kwargs
            )
        key, value = self._flex_cp_allgather(
            key.contiguous(), value.contiguous(), _SEQ_DIM, self._cp_pg_name
        )
        out = _run_flex(module, query, key, value, block_mask, kwargs)
        # HF's interface contract is (batch, seq, heads, dim).
        return out.transpose(1, 2)

    def _forward_ulysses(self, query, key, value, *, module, block_mask=None, **kwargs):
        """Swap the token shard for a head shard, attend full-length, swap back.

        The mask is chosen by what the wrapper handed over, and the two cases
        are distinguished by the mask's Q length against the post-swap (full)
        sequence length:

        * a FULL-LENGTH mask is a packed corpus's document mask
          (``block_causal``): every rank attends the full sequence after the
          all-to-all, so the unsharded document structure applies directly.
          Passing it Q-sharded instead would index the wrong queries -- this is
          the varlen/packed path, where "varlen metadata" in hpmesh's flex
          integration is the document structure baked into the BlockMask.
        * anything shorter (or nothing) is the single-document case, whose
          incoming mask is the wrapper's Q-sharded causal CP mask; the
          full-length causal one is rebuilt from the length alone (see
          :meth:`_full_length_causal_mask`).

        The decision is a pure function of shapes the input sharding fixes
        identically on every rank (the wrapper prebuilds the packed mask
        whenever ``attn_mask_type`` says packed, and both masks are built over
        the full pre-shard sequence), so all ranks take the same branch and no
        rank can stall the all-to-alls on a local surprise.
        """
        q = _SeqToHead.apply(query.contiguous(), self._cp_group)
        k = _SeqToHead.apply(key.contiguous(), self._cp_group)
        v = _SeqToHead.apply(value.contiguous(), self._cp_group)
        if block_mask is None or block_mask.seq_lengths[0] != q.shape[_SEQ_DIM]:
            block_mask = self._full_length_causal_mask(q)
        out = _run_flex(module, q, k, v, block_mask, kwargs)
        out = _HeadToSeq.apply(out.contiguous(), self._cp_group)
        return out.transpose(1, 2)  # HF's interface contract is (b, s/cp, h, d)

    def _full_length_causal_mask(self, q_BHSD: torch.Tensor):
        """The full-sequence causal BlockMask, built once per length, device,
        and batch-invariant mode.

        Under ulysses, q/k/v arrive at flex with the full sequence, so the mask
        is the unsharded causal one -- the same mask the wrapper builds
        internally before Q-sharding it for kv_allgather. ``separate_full_blocks``
        tracks the wrapper's batch-invariant-mode choice, so the ulysses
        decomposition matches every other path's numerics.

        Built from the length alone, which is why this only holds for a
        contiguous split: the causal mask depends on the *order* of the tokens,
        and every rank here holds its shard of that one order. A
        load-balancer-rearranged shard would need the mask permuted to match,
        and nothing in this signature says which rearrangement that was --
        ``apply_cp`` refuses ulysses with a load balancer for that reason.
        """
        seq_len = q_BHSD.shape[_SEQ_DIM]
        key = (seq_len, q_BHSD.device, is_in_batch_invariant_mode())
        mask = self._full_masks.get(key)
        if mask is None:
            # The wrapper's own mask builder, with the wrapper's exact causal
            # arguments: same eager/compiled backend choice, same
            # ``separate_full_blocks`` handling, same BLOCK_SIZE.
            mask = create_attention_mask(
                get_causal_mask_mod(),
                1,
                None,
                seq_len,
                seq_len,
                device=q_BHSD.device,
                BLOCK_SIZE=128,
                separate_full_blocks=not is_in_batch_invariant_mode(),
            )
            self._full_masks[key] = mask
        return mask
