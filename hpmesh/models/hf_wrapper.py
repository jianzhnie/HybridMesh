"""A thin wrapper that plugs a HuggingFace model into a training loop.

This is the whole model layer: build a HF ``ForCausalLM`` from its config, give
the training loop a uniform ``(tokens, positions) -> logits`` entry point, and
expose the decoder's parts under the names the rest of the framework expects
(``tok_embeddings`` / ``layers`` / ``norm`` / ``lm_head`` / ``rotary_emb``).

Deliberately thin. An earlier version of this file (torchtitan's
``experiments/transformers_modeling_backend/model.py``) carried a 230-line
monkey-patch of ``PreTrainedModel._init_weights`` that re-derived every weight
with torchtitan's depth-scaled scheme, plus a ``PretrainedConfig`` subclass that
synced torchtitan and HF attribute names. Both existed to make checkpoints
bit-compatible with torchtitan's native models. hpmesh trains HF models with
HF's own initialization, so neither is needed -- ``from_config`` produces a
fully initialized model and that is the one we train.

What replaces them is per-forward work the HF model does not do on its own:

* attention is routed through flex attention so packed-document masking can be
  expressed (see ``get_attention_masks``),
* RoPE is driven by explicit ``positions`` rather than an ``arange``, because
  packed samples reset their positions at each document boundary.

Forward shape: this wrapper is a *decoder wrapper*, not a CausalLM. It runs the
decoder and applies ``lm_head`` itself, so ``self.model.model`` is the bare text
stack. Callers pass flat ``(T,)`` token and position tensors; the batch dim is
added and removed internally. ``forward(..., skip_lm_head=True)`` returns the
hidden states instead, for the trainer's chunked-loss path.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.modeling_utils import AttentionInterface

from ..accelerator import dist_utils
from ..components.loss import next_token_targets
from ..datasets.types import Batch
from ..parallel.compile import maybe_regional_inductor
from ..parallel.context_parallel import (
    shard_attention_mask_for_cp,
    shard_batch_for_cp,
    shard_batch_for_tp,
    shard_padding_mask_for_cp,
    shard_padding_mask_for_tp,
)
from ..parallel.parallel_dims import ParallelDims
from ..utils.batch_invariant import is_in_batch_invariant_mode
from ..utils.logger_utils import get_logger
from .common.cast_linear import TORCH_DTYPE_MAP, to_cast_linear
from .common.masks import (
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from .common.moe import _iter_moe_layers
from .hf_factory import _ATTN_IMPLEMENTATION, resolve_model_class, unwrap_text_config

logger = get_logger(__name__)

__all__ = ["HFTransformerModel"]



def _flex_supported() -> str:
    """The attention implementation this machine can actually run.

    Flex attention lowers through inductor, and inductor has no CPU target, so
    ``torch.compile(flex_attention, ...)`` raises ``NotImplementedError`` off a
    CUDA device. That is a property of the machine, not of the run, so it is
    decided here rather than requested by the config.

    ``"sdpa"`` is a fallback in *backend*, not in *arithmetic*: on the path this
    wrapper takes (causal, no packing) both compute the same thing, for a reason
    worth spelling out. The wrapper deliberately passes no ``attention_mask``
    down to the decoder, and HF's sdpa path ignores ``is_causal`` whenever a mask
    is present, deriving causality from the mask instead. Handing it the
    ``BlockMask`` would therefore silently disable masking. Leaving the mask
    unset lets sdpa default to causal -- the same thing the flex causal mod
    applies. What is genuinely lost off CUDA is packed-document masking:
    ``get_attention_masks`` still builds a correct ``BlockMask``, it just has no
    flex kernel to run it in. Packed batches must run on CUDA.
    """
    return _ATTN_IMPLEMENTATION if torch.cuda.is_available() else "sdpa"


def flex_attention_hf(module, query, key, value, attention_mask, **kwargs):
    """HF ``AttentionInterface`` entry point for flex attention.

    When a kernel has been attached under the attention module (the parallelism
    layer does this to declare a local SPMD region), route through it so the
    sharding declarations take effect; otherwise run flex directly on the plain
    tensors.
    """
    kernel = getattr(module, "_titan_flex_kernel", None)
    if kernel is None:
        # Mark the flex region so that, when the enclosing model is compiled
        # with a non-inductor backend, regional_inductor scoops just this
        # region into an inductor sub-compile (see parallel/compile.py). A
        # null context on the default inductor / eager paths, so no dead
        # metadata is emitted. Empty configs: inductor defaults -- hpmesh
        # runs HF's flex integration, not its own compiled-flex instance.
        with maybe_regional_inductor({}):
            return flex_attention_forward(
                module, query, key, value, attention_mask, **kwargs
            )
    out = kernel(query, key, value, module=module, block_mask=attention_mask, **kwargs)
    return out, None


def _uses_dsa(config) -> bool:
    """True if the model uses DeepSeek-style sparse attention (DSA).

    DSA models (e.g. GLM-5, model_type 'glm_moe_dsa') run an auxiliary
    "indexer" sub-attention that scores all keys and selects the top-k per
    query, expressing the selection as a dense additive mask. A flex
    ``BlockMask`` has no ``.dim()`` and cannot be added elementwise, so DSA
    needs a dense 4D tensor mask (torchtitan builds one in
    ``_build_dense_attention_mask``). Detected by the DSA-specific
    ``index_topk`` config attr.
    """
    return getattr(config, "index_topk", None) is not None







def first_present(module: nn.Module, names: tuple[str, ...], what: str) -> str:
    """Return the first of ``names`` that ``module`` has.

    HF names the same submodule differently across model families
    (``embed_tokens``/``wte``, ``norm``/``final_layernorm``/``ln_f``), so the
    wrapper probes instead of hardcoding one family's spelling. Resolved once at
    construction: a missing part should fail at build time, not mid-forward.
    """
    for name in names:
        if hasattr(module, name):
            return name
    raise AttributeError(
        f"{type(module).__name__} has no {what} under any of {names}. "
        "Add the model's spelling to the probe in HFTransformerModel.__init__."
    )


def _collapse_batch_dims(
    inputs: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a ``(B, T)`` batch into the ``(B*T,)`` shape the forward takes.

    The wrapper is a single-sequence entry point -- it adds and removes its own
    batch dim around the decoder call. The synthetic source yields one document
    per row of length ``max_seq_len``, so the concatenation is exactly the
    single causal document the fallback attention path expects; RoPE is driven
    per row because positions restart at each row boundary.

    A packed (Grain) batch already arrives as a flat token stream, so for it
    this is the identity -- which is what makes the two sources one code path
    from here on. Used by both the training path (``preprocess_inputs``) and the
    pipeline path, which chunks rows before collapsing; keeping it in one place
    is what makes the two chunkings agree.
    """
    return inputs.reshape(-1), labels.reshape(-1)


def _document_shift(labels: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Next-token targets within a row, ``IGNORE_INDEX`` at each row end.

    The synthetic source hands over labels equal to its inputs, so the shift is
    the model's. Rows are independent documents of length ``seq_len``, so the
    shift has to stay *within* a row: the row-final position would predict the
    next document's first token, which the model had no context for. Those
    positions come back ``IGNORE_INDEX`` and are excluded from the loss.

    The packed source arrives already shifted and already masked at its document
    boundaries, so it never calls this.
    """
    return next_token_targets(labels.reshape(-1), seq_len=seq_len)



class HFTransformerModel(nn.Module):
    """A HF decoder stack behind a uniform training forward.

    Args:
        config: the HF config to instantiate (see ``build_model_config``).
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        config = unwrap_text_config(config)
        if _uses_dsa(config):
            # Fail fast rather than run silently wrong: hpmesh always builds a
            # flex BlockMask, which DSA's indexer and main attention cannot
            # consume (they call ``.dim()`` on the mask and add it to scores).
            # Upstream's dense additive mask path is an unported D-class gap.
            raise NotImplementedError(
                f"{config.model_type}: DeepSeek-style sparse attention (DSA) "
                "models are not supported -- they need a dense additive mask "
                "path that hpmesh does not implement."
            )
        num_heads = getattr(config, "num_attention_heads", None)
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads is not None and num_heads < 1:
            raise ValueError(f"num_attention_heads must be >= 1, got {num_heads}")
        if num_kv_heads is not None and num_kv_heads < 1:
            raise ValueError(f"num_key_value_heads must be >= 1, got {num_kv_heads}")
        if (
            num_heads is not None
            and num_kv_heads is not None
            and num_heads % num_kv_heads != 0
        ):
            raise ValueError(
                f"num_attention_heads ({num_heads}) must be divisible by "
                f"num_key_value_heads ({num_kv_heads})"
            )
        config._attn_implementation = _flex_supported()
        AttentionInterface._global_mapping[_ATTN_IMPLEMENTATION] = flex_attention_hf

        model_cls = resolve_model_class(config)
        # Select the HF experts forward kernel, honoring the explicit request
        # or failing -- never silently substituting a different kernel
        # (torchtitan's semantics). "native" leaves the model's built-in
        # kernel alone; anything else requires a model whose experts
        # implementation is settable (transformers' @use_experts_implementation,
        # probed as a classmethod). Irrelevant under EP>1, where the swap
        # replaces the whole MoE block.
        impl = getattr(config, "experts_implementation", "native")
        if impl != "native":
            can_set = getattr(model_cls, "_can_set_experts_implementation", None)
            if impl not in ("grouped_mm", "batched_mm", "eager"):
                raise ValueError(
                    f"experts_implementation must be one of 'native', "
                    f"'grouped_mm', 'batched_mm', 'eager', got '{impl}'"
                )
            if can_set is None or not can_set():
                raise ValueError(
                    f"{model_cls.__name__} does not support a settable experts "
                    f"implementation, so experts_implementation='{impl}' cannot "
                    "be honored. Set experts_implementation='native' to use "
                    "the HF model's built-in experts kernel."
                )
            config._experts_implementation = impl
        self.model = model_cls(config=config)
        self.model.config._attn_implementation = config._attn_implementation

        # Optional fixed-dtype lm_head (torchtitan's CastLinear semantics):
        # score the vocabulary logits in the requested dtype while the stored
        # weight keeps its own. Swapped in place rather than wrapped so the
        # state-dict FQNs -- and a weight tie with the embedding -- are
        # untouched (see models/common/cast_linear.py for why subclassing
        # beats wrapping).
        compute_dtype = getattr(config, "compute_dtype", None)
        if compute_dtype is not None:
            if compute_dtype not in TORCH_DTYPE_MAP:
                raise ValueError(
                    f"compute_dtype must be one of {sorted(TORCH_DTYPE_MAP)}, "
                    f"got {compute_dtype!r}"
                )
            if self.model.lm_head is not None:
                self.model.lm_head = to_cast_linear(
                    self.model.lm_head, TORCH_DTYPE_MAP[compute_dtype]
                )

        self.cp_mesh = None
        self._cp_load_balancer = None
        self._cp_strategy = "kv_allgather"

        # The decoder is the text stack; lm_head is its sibling on the CausalLM.
        # Stored with object.__setattr__ on purpose: a plain ``self._decoder = ...``
        # would go through nn.Module.__setattr__, which registers it in
        # ``_modules`` and makes every parameter appear twice in the state dict
        # (once under ``model.``, once under ``_decoder.``).
        #
        # The attribute names are resolved once here; the accessors below are
        # properties so a pipeline stage can swap a part out in place.
        object.__setattr__(self, "_decoder", self.model.model)
        object.__setattr__(
            self,
            "_embed_name",
            first_present(self._decoder, ("embed_tokens", "wte"), "token embedding"),
        )
        object.__setattr__(
            self,
            "_norm_name",
            first_present(
                self._decoder, ("norm", "final_layernorm", "ln_f"), "final norm"
            ),
        )

    # -- parts, under the names the rest of the framework uses ------------------

    @property
    def tok_embeddings(self) -> nn.Module:
        return getattr(self._decoder, self._embed_name)

    @tok_embeddings.setter
    def tok_embeddings(self, value: nn.Module) -> None:
        setattr(self._decoder, self._embed_name, value)

    @property
    def layers(self) -> nn.ModuleList:
        return self._decoder.layers

    @layers.setter
    def layers(self, value) -> None:
        self._decoder.layers = value

    @property
    def norm(self) -> nn.Module:
        return getattr(self._decoder, self._norm_name)

    @norm.setter
    def norm(self, value: nn.Module) -> None:
        setattr(self._decoder, self._norm_name, value)

    @property
    def lm_head(self) -> nn.Module | None:
        return getattr(self.model, "lm_head", None)

    @lm_head.setter
    def lm_head(self, value: nn.Module | None) -> None:
        self.model.lm_head = value

    @property
    def rotary_emb(self) -> nn.Module | None:
        return getattr(self._decoder, "rotary_emb", None)

    @rotary_emb.setter
    def rotary_emb(self, value: nn.Module | None) -> None:
        self._decoder.rotary_emb = value

    @property
    def uses_flex_attention(self) -> bool:
        """Whether the decoder routes attention through the flex kernel.

        Read by the compile step: flex has only an inductor lowering, so a
        non-inductor compile backend must either scoop the flex region
        (``aot_eager`` + regional_inductor) or be rejected.
        """
        return self.model.config._attn_implementation == _ATTN_IMPLEMENTATION

    @property
    def enable_weight_tying(self) -> bool:
        """Whether ``lm_head`` and the embedding share one ``Parameter``.

        Read by FSDP, which must not let one Parameter be owned by two FSDP
        units; the two modules are wrapped together when this is true.

        Compared by identity rather than by ``config.tie_word_embeddings``: that
        flag records intent, and a model may carry an unshared head anyway (or
        share one the flag does not mention). Identity is also exactly the check
        FSDP2 itself performs, so the answer here matches what FSDP will do.
        """
        embed, head = self.tok_embeddings, self.lm_head
        if head is None:
            return False
        return getattr(embed, "weight", None) is getattr(head, "weight", None)

    # -- HF integration hooks --------------------------------------------------

    def set_cp_mesh(
        self, mesh, *, load_balancer: str | None = None, strategy: str = "kv_allgather"
    ) -> None:
        """Record the CP mesh so logit dumps can tag their CP coordinate.

        Also records the CP load-balancer type and strategy: the trainer shards
        the batch with the balancer, and the forward's BlockMask handling must
        match the strategy -- kv_allgather attends gathered K/V against a
        Q-sharded mask, while ulysses reassembles the full sequence in the
        all-to-all and needs a packed corpus's document mask full-length.
        """
        self.cp_mesh = mesh
        self._cp_load_balancer = load_balancer
        self._cp_strategy = strategy

    @property
    def tp_plan(self) -> dict[str, str]:
        """HF's TP plan, with patterns rewritten to THIS wrapper's module paths.

        HF states its plan relative to the model it ships -- ``layers.*.q_proj``,
        plus a ``model.``-prefixed variant for families that nest one level
        deeper. The parallel layer, however, walks *this* wrapper, whose
        ``named_modules`` paths all sit under ``model.`` because that is the
        attribute the HF CausalLM is held in. Prefixing every pattern with
        ``model.`` maps one spelling onto the other, so HF's own two variants
        both resolve here and no call site has to know either layout.

        Without this the plan is simply not found (the attribute lives on the
        inner HF model) and ``apply_tp`` silently shards nothing -- a replicated
        run that looks like a working one. That failure is what makes this
        translation load-bearing rather than cosmetic.

        Returns ``{}`` when the model ships no plan. That is left as an empty
        plan rather than an error because "no declared plan" is a real answer:
        ``apply_tp`` then leaves the model replicated instead of guessing.
        """
        plan = getattr(self.model, "_tp_plan", None) or {}
        return {f"model.{pattern}": spec for pattern, spec in plan.items()}

    def named_children(self):
        """Present the decoder's parts as direct children.

        ``nn.Module.named_children`` would yield exactly one child (``self.model``),
        so the parallelism layer, which walks children, would see a single opaque
        blob. Yielding the parts here is what lets it address the decoder's pieces
        (``layers.*`` and friends) directly.

        This does NOT flatten ``state_dict`` keys: the state dict is built from
        ``_modules``, which still holds everything under ``self.model``, so keys
        keep their ``model.`` prefix. Only the child *iteration* is reshaped.
        """
        yield "tok_embeddings", self.tok_embeddings
        yield "layers", self.layers
        yield "norm", self.norm
        if self.lm_head is not None:
            yield "lm_head", self.lm_head
        if self.rotary_emb is not None:
            yield "rotary_emb", self.rotary_emb

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims | None,
        parallelism=None,
        max_context_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Turn a dataloader batch into ``(inputs, labels, extra_kwargs)``.

        The seam torchtitan's trainer calls, and the reason it is on the model
        rather than in the loop: every step below is a statement about *this*
        architecture's input contract, and one of them needs the full-length
        positions that only exist before the sequence is sharded.

        1. **Normalize the batch shape.** The two loaders disagree about what a
           batch is -- the synthetic one yields ``(B, T)`` rows of one document
           each, a Grain one a flat packed stream -- so they are reconciled
           here. ``num_valid_tokens`` is *not* read: the trainer pops it before
           this call, because the loss denominator has to be the pre-shard count
           and reduced across DP before the first backward.
        2. **Collapse the batch dim**, turning ``(B, T)`` into the flat ``(B*T,)``
           the forward takes.
        3. **Build the attention mask**, when the batch carries ``positions``.
           This has to happen before step 5: the document structure of a packed
           batch is a global property, so the mask is built over the FULL
           sequence and only then Q-sharded.
        4. **Shard for context parallelism**, positions included, so RoPE
           follows each token to its rank.
        4b. **Shard for tensor parallelism** (sequence parallelism): the fused
           TP GEMMs all-gather the sequence inside each projection, so the
           forward must be entered holding only this rank's ``T / tp`` slice --
           otherwise the gather concatenates ``tp`` copies of the full sequence
           and every sharded weight gradient comes out ``tp`` times too large.
           Only the token-carrying tensors (``inputs``, ``labels``) are cut:
           after the in-projection gather, attention and RoPE see the assembled
           sequence (the CP shard, or the full sequence with CP off), so
           ``positions`` and the mask keep their CP/full length. This matches
           torchtitan's joint ``(CP, TP)`` sequence sharding
           (``hf_sharding.py``'s ``PartitionSpec(DP, (CP, TP), None)``): the CP
           shard composed with a contiguous TP slice of it is exactly the joint
           (CP outer, TP inner) split.
        5. **Return the leftover dict as ``extra_kwargs``.** Those are splatted
           into ``forward``, so anything left here must be one of its keyword
           parameters -- ``positions``, ``attention_masks`` and
           ``padding_mask``, and nothing else.

        ``positions`` is optional: the synthetic source has none and the forward
        falls back to its own ``arange``, which is right for a single document
        but must not be relied on for a packed one. ``parallelism`` is accepted
        for signature parity with the reference but unused: hpmesh's CP
        load-balancer string is latched onto this wrapper by ``apply_cp`` (see
        ``set_cp_mesh``).

        The reference also takes ``max_num_documents``, to size a fixed-shape
        varlen mask for CUDA graph capture. It is not accepted here because
        there is no path for it to take: hpmesh's CP path goes through
        ``create_attention_mask`` (a flex ``BlockMask``), which rebuilds from
        the positions shard and takes no capacity bound, and the packing
        collator already caps segments at one row. Accepting it would mean a
        parameter the trainer passes and this method silently discards.

        Tensors arrive on the trainer's device; the trainer moves them before
        calling, so this is pure structure and stays device-free.
        """
        del parallelism
        extra_kwargs: dict[str, Any] = {}
        positions = None
        padding_mask = None

        if isinstance(input_dict, Batch):
            # Rows are independent documents of length T.
            labels = _document_shift(
                input_dict.labels, seq_len=input_dict.labels.shape[-1]
            )
            inputs = input_dict.input_ids
        else:
            # Packed stream: the collator already shifted and masked the labels
            # at every document boundary, so for this path the shift is a read.
            inputs, labels = input_dict["input"], input_dict["labels"]
            positions = input_dict.get("positions")
            # True-for-padding, from the collator; consumed by the MoE blocks'
            # load-balancing statistics (see forward's ``padding_mask``).
            padding_mask = input_dict.get("padding_mask")

        inputs, labels = _collapse_batch_dims(inputs, labels)
        if positions is not None:
            positions = positions.reshape(-1)
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1).to(torch.bool)

        # Whether the mask must be prebuilt full-length: a packed corpus
        # (``block_causal``) carries a document structure that is not
        # recoverable from a CP positions shard, so the mask must exist before
        # the shard below. This is decided by ``attn_mask_type``, NOT by
        # scanning ``positions`` for restarts: a restart scan misses the two
        # shapes a real packed corpus produces -- a document that fills the
        # whole row (the packing collator splits overlong documents into
        # single-document rows, whose positions are a plain arange) and a
        # boundary after a length-1 document (positions ``[0, 0, ...]`` have no
        # descending edge). The first would crash the CP forward for lack of a
        # prebuilt mask; the second would disarm the sdpa packed-guard and let
        # attention cross the boundary silently.
        packed = positions is not None and (
            getattr(self.model.config, "attn_mask_type", "causal") == "block_causal"
        )

        # Built before the CP shard, always from the FULL-length positions --
        # which is exactly why this lives here and not in the loop: after the
        # shard below, no rank holds a positions vector that can describe the
        # document structure. When the mask is built there is no need to hand
        # it to the forward: ``_apply_attention`` builds one from ``positions``
        # anyway, so passing it would be a second copy rather than a saving.
        cp_mesh = (
            None if parallel_dims is None else parallel_dims.get_optional_mesh("cp")
        )
        if cp_mesh is None and positions is not None:
            mask = self.get_attention_masks(positions=positions)
            if self.model.config._attn_implementation == _ATTN_IMPLEMENTATION:
                extra_kwargs["attention_masks"] = mask

        if cp_mesh is not None:
            if positions is None:
                # The forward's own ``arange`` default would restart at 0 on
                # every rank; the shard needs positions that describe the whole
                # sequence.
                positions = torch.arange(inputs.numel(), device=inputs.device)
            # A causal-only mask (a single document) can be rebuilt from this
            # rank's positions shard, which is what ``_get_cp_attention_masks``
            # does. Packed cannot: ``positions`` is about to be sharded and the
            # document structure is not recoverable from a shard of it, so the
            # full-length mask is built first. What happens to it next depends
            # on the strategy: kv_allgather attends gathered full-length K/V
            # against local queries, so the mask is Q-sharded to match (the GQA
            # head count still divides by cp -- sharding Q does not change how
            # many Q heads a rank owns); ulysses all-to-all's the FULL sequence
            # onto every rank before attention, so the document mask stays
            # full-length and unsharded -- the varlen semantics, where the
            # document structure is global metadata that the token shard must
            # not cut. Both decisions are config-keyed, hence rank-symmetric.
            if packed:
                attention_masks = self.get_attention_masks(positions=positions)
                if self._cp_strategy != "ulysses":
                    attention_masks = shard_attention_mask_for_cp(
                        attention_masks,
                        cp_mesh,
                        self._cp_load_balancer,
                    )
                if self.model.config._attn_implementation == _ATTN_IMPLEMENTATION:
                    extra_kwargs["attention_masks"] = attention_masks
            inputs, labels, positions = shard_batch_for_cp(
                inputs,
                labels,
                positions,
                cp_mesh,
                load_balancer=self._cp_load_balancer,
            )
            if padding_mask is not None:
                # The mask describes the same token stream, so it takes the
                # same shard (load-balancer rearrangement included).
                padding_mask = shard_padding_mask_for_cp(
                    padding_mask, cp_mesh, self._cp_load_balancer
                )

        tp_mesh = (
            None if parallel_dims is None else parallel_dims.get_optional_mesh("tp")
        )
        if tp_mesh is not None:
            # Sequence parallelism premise (step 4b above): cut the token-
            # carrying tensors along the TP axis. Positions are NOT cut --
            # after the in-projection all-gather, RoPE and attention see the
            # assembled sequence, so they keep the CP-shard (or, with CP off,
            # full) length. Synthesize them full-length when the batch did not
            # carry any: the forward's own ``arange`` default would be sized to
            # the TP-sharded input and restart at 0 on every rank.
            if positions is None:
                positions = torch.arange(inputs.numel(), device=inputs.device)
            if padding_mask is not None:
                # Cut BEFORE the batch tensors: the mask's length still matches
                # the pre-shard ``inputs``, which is what the divisibility
                # check must measure.
                padding_mask = shard_padding_mask_for_tp(padding_mask, tp_mesh)
            inputs, labels = shard_batch_for_tp(inputs, labels, tp_mesh)

        if positions is not None:
            extra_kwargs["positions"] = positions
        if padding_mask is not None:
            extra_kwargs["padding_mask"] = padding_mask
        return inputs, labels, extra_kwargs

    def get_attention_masks(self, positions: torch.Tensor):
        """Build the flex BlockMask for this batch.

        ``attn_mask_type`` selects between plain causal and causal-plus-same-
        document. The latter is the packed path: samples share one sequence, so
        attention must not cross a document boundary (positions reset to 0
        there). Both cases return a BlockMask -- with no mask at all flex would
        compute full attention.
        """
        if getattr(self.model.config, "attn_mask_type", "causal") == "block_causal":
            mask_mod = and_masks(
                get_causal_mask_mod(),
                get_document_mask_mod(positions),
            )
        else:
            mask_mod = get_causal_mask_mod()

        num_tokens = positions.shape[0]
        return create_attention_mask(
            mask_mod,
            1,
            None,
            num_tokens,
            num_tokens,
            device=positions.device,
            BLOCK_SIZE=128,
            separate_full_blocks=not is_in_batch_invariant_mode(),
        )

    def _get_cp_attention_masks(self, positions: torch.Tensor):
        """Build the BlockMask for a CP forward: full-length, strategy-shaped.

        Under CP, ``positions`` is this rank's shard of the sequence (possibly
        load-balancer-rearranged), so it cannot describe the full document
        structure: the mask is built over the FULL sequence. kv_allgather then
        needs it sharded along its Q axis, matching how the CP kernel's
        gathered K/V stay full-length. Ulysses needs it as built: its
        all-to-all reassembles the full sequence before attention, and the
        kernel would only discard a Q-sharded mask and rebuild this very one.
        ``get_attention_masks`` builds the full mask from an arange -- valid
        because only the causal mod is taken here.

        Packed batches (``block_causal``) cannot take this path: the document
        mask needs the full positions, which only the caller has. Build the
        full-length mask with ``get_attention_masks(full_positions)`` and pass
        it as ``attention_masks`` -- Q-sharded by ``shard_attention_mask_for_cp``
        for kv_allgather, full-length and unsharded for ulysses.
        """
        if getattr(self.model.config, "attn_mask_type", "causal") == "block_causal":
            raise ValueError(
                "Context parallel with packed sequences needs a prebuilt mask: "
                "build the full-length BlockMask with get_attention_masks from "
                "the FULL positions and pass it to forward as attention_masks "
                "-- Q-sharded with shard_attention_mask_for_cp under "
                "'kv_allgather', or full-length (unsharded) under 'ulysses', "
                "whose all-to-all reassembles the full sequence on every rank. "
                "The positions this forward receives are already CP-sharded and "
                "cannot describe the full document structure."
            )
        cp_size = self.cp_mesh.size()
        full_len = positions.shape[0] * cp_size
        full_positions = torch.arange(full_len, device=positions.device)
        mask = self.get_attention_masks(positions=full_positions)
        if self._cp_strategy == "ulysses":
            return mask
        return shard_attention_mask_for_cp(mask, self.cp_mesh, self._cp_load_balancer)

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        attention_masks=None,
        padding_mask: torch.Tensor | None = None,
        skip_lm_head: bool = False,
    ) -> torch.Tensor:
        """Run the decoder over one packed sequence and return logits.

        Args:
            input_ids: ``(T,)`` flat token ids. Under CP, this rank's sequence
                shard -- ``(T/cp,)``. Under pipeline parallelism, a non-first
                stage receives the previous stage's output instead: ``(T, H)``
                hidden states, detected by ``tok_embeddings`` having been split
                out (replaced by an ``nn.Identity``).
            positions: ``(T,)`` per-token positions, resetting at document
                boundaries. Drives RoPE. Defaults to ``arange``, which is correct
                only when the sequence is a single document. Under CP, the
                matching shard of the full positions.
            attention_masks: a prebuilt BlockMask. Only the flex backend consumes
                it (see ``_apply_attention``); with sdpa the decoder is left to
                its own causal default. Under CP, a full-length mask already
                Q-sharded by ``shard_attention_mask_for_cp``.
            padding_mask: ``(T,)`` boolean, true for padding, sharded exactly
                like ``input_ids``. Staged onto every swapped-in MoE block
                (``MoE.set_padding_mask``) so the blocks' load-balancing
                statistics skip padding tokens; the decoder's arithmetic is
                unaffected. The HF decoder layer's fixed
                ``self.mlp(hidden_states)`` call signature is why this is a
                staged side channel rather than a threaded argument.
            skip_lm_head: return the ``(T, H)`` hidden states instead of logits.
                The chunked-loss path uses this so the trainer can run lm_head +
                cross-entropy per sequence chunk (see
                ``components.loss.chunked_lm_head_cross_entropy``) rather than
                materialize the full ``T * V`` logits.
        """
        if isinstance(self.tok_embeddings, nn.Identity):
            # Non-first pipeline stage: the input IS the previous stage's
            # hidden states, so the embedding lookup is skipped by feeding the
            # decoder ``inputs_embeds`` directly. PP does not shard the
            # sequence, so the local ``arange`` positions default stays right.
            local_seq_len = input_ids.shape[0]
            decoder_input = {"inputs_embeds": input_ids.unsqueeze(0)}
        else:
            local_seq_len = input_ids.shape[0]
            decoder_input = {"input_ids": input_ids.unsqueeze(0)}
        if positions is None:
            positions = torch.arange(local_seq_len, device=input_ids.device)

        kwargs = self._apply_attention(positions, attention_masks)

        # Stage the padding mask on every MoE block -- including ``None``, so
        # a mask from a previous microbatch can never survive into this one.
        # Each block consumes its staged mask on its forward, once.
        for moe in _iter_moe_layers(self):
            moe.set_padding_mask(padding_mask)

        # A HF decoder expects a batch dim; the wrapper's contract is flat.
        hidden_states = self._decoder(
            **decoder_input,
            position_ids=positions.unsqueeze(0),
            use_cache=False,
            **kwargs,
        ).last_hidden_state.squeeze(0)

        if (
            self.lm_head is not None
            and not isinstance(self.lm_head, nn.Identity)
            and not skip_lm_head
        ):
            logits = self.lm_head(hidden_states)
        else:
            # Non-final pipeline stage, or a chunked-loss forward: the output
            # must own its storage rather than be a view of the decoder's
            # ``last_hidden_state`` (squeeze above), because split-backward
            # schedules (ZBVZeroBubble) call ``detach_()`` on stage outputs,
            # which views do not support. The chunked-loss caller does not
            # detach in place, but shares the branch so both skip reasons stay
            # one code path -- the clone is one T*H copy per forward.
            logits = hidden_states.clone()

        _dump_dir = os.environ.get("HF_BACKEND_LOGIT_DUMP")
        if _dump_dir is not None:
            self._maybe_dump_logits(_dump_dir, logits)

        return logits

    def _apply_attention(
        self, positions: torch.Tensor, attention_masks
    ) -> dict[str, Any]:
        """Cross the ROLE-IN / CONVENTION-OUT seam: decide what to hand the decoder.

        Roles are fixed: this wrapper ALWAYS routes through an attention
        implementation, and that implementation is ALWAYS fed a mask describing
        how tokens may attend -- flex consumes it as a ``BlockMask``. How a
        *backend* wants its mask is a different question (sdpa wants a boolean
        tensor, and derives causality from the mask's presence), so it is settled
        here and nowhere else.

        Consequently the mask is built unconditionally, even on the backend that
        ends up discarding it: not building it would make the two paths differ in
        more than the backend, and would hide the packing gap this fallback
        leaves open.
        """
        if attention_masks is None:
            if self.cp_mesh is None:
                attention_masks = self.get_attention_masks(positions=positions)
            else:
                attention_masks = self._get_cp_attention_masks(positions)

        # is_causal is the flex-only spelling: it selects which mod the BlockMask
        # runs, so it is withheld from every other backend (see below).
        if self.model.config._attn_implementation == _ATTN_IMPLEMENTATION:
            return {"attention_mask": attention_masks, "is_causal": False}

        # Every other backend wants a tensor mask (or none). There is no generic
        # conversion from a BlockMask, and one is not wanted: the only hpmesh
        # case that needs a tensor mask is packing, which the flex fallback
        # cannot express anyway. Fail loudly rather than run with the mask
        # silently dropped -- that would turn the packed path into full attention.
        #
        # Packing is detected from ``attn_mask_type``, not from scanning
        # ``positions`` for restarts: a restart scan misses a boundary after a
        # length-1 document (positions ``[0, 0, ...]`` never descend), which
        # would let sdpa run plain causal attention across the boundary with no
        # complaint. The flag is derived from the corpus at config build time
        # (``build_model_config_for``), so it cannot disagree with the batch.
        if getattr(self.model.config, "attn_mask_type", "causal") == "block_causal":
            raise ValueError(
                f"Attention backend {self.model.config._attn_implementation!r} "
                "cannot express the mask for a packed sequence: this run's "
                "corpus is packed (attn_mask_type='block_causal'), and a "
                "tensor-mask backend has no way to keep attention from "
                "crossing a document boundary. Sequence packing requires CUDA "
                "and the flex attention kernel."
            )

        # A single causal document. Hand the decoder NOTHING and let HF follow its
        # own path: it builds a 4D mask only when it must (sliding-window layers,
        # padding) and otherwise returns None, and its sdpa wrapper derives
        # ``is_causal`` from whether a mask is present. Passing ``is_causal=True``
        # ourselves would double up with the mask HF does build -- sdpa rejects
        # ``attn_mask`` together with ``is_causal=True`` -- which is what an
        # earlier version of this method got wrong.
        return {"attention_mask": None}

    def _maybe_dump_logits(self, dump_dir: str, logits: torch.Tensor) -> None:
        """Append this rank's logits (one entry per forward) for numerical tests."""
        rank = dist_utils.get_rank()
        cp_coord = self.cp_mesh.get_local_rank() if self.cp_mesh is not None else 0
        recs = getattr(self, "_logit_dump_recs", None)
        if recs is None:
            recs = self._logit_dump_recs = []
        recs.append((cp_coord, logits.detach().float().cpu()))
        torch.save(recs, os.path.join(dump_dir, f"logits_rank{rank}.pt"))

    def __setattr__(self, name, value) -> None:
        """Route property-backed names through their setters.

        ``nn.Module.__setattr__`` registers modules directly, which would bypass
        the property setters defined above (``self.layers = ...`` would create a
        new child instead of replacing the decoder's).
        """
        prop = getattr(type(self), name, None)
        if isinstance(prop, property) and prop.fset is not None:
            prop.fset(self, value)
            return
        super().__setattr__(name, value)
