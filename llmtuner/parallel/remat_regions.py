"""The region vocabulary ``RegionAC`` uses on HF decoder blocks.

Upstream torchtitan's ``RegionAC`` reads its region names off its own model code:
every ``torch_remat.region(...)`` call site there names a semantic unit
(``attention.qkv``, ``attention.inner_attention``, ``feed_forward.w13``,
``feed_forward.w2``), and ``Module.configure_remat_regions`` hands the block's
save patterns down to those call sites. llmtuner runs HF models and does not own
their decoder code, so the same idea is realized structurally instead: **every
``nn.Linear`` inside a decoder block is a region**, named by its FQN relative to
the block.

That is the upstream policy spelled in HF's vocabulary rather than a retreat
from it:

* upstream's ``save_regions`` examples name projections, and in HF those are
  exactly the ``nn.Linear``s -- ``self_attn.{q,k,v,o}_proj``,
  ``mlp.{gate,up,down}_proj``, and an MoE router's ``gate``;
* ``Linear.forward`` is ``forward(input) -> Tensor``, the flat
  tensor-in/tensor-out shape ``torch_remat.region`` requires of the callable it
  annotates (an arbitrary HF submodule may take and return nested structures,
  which that API walks only leniently);
* the regions are siblings, never nested, so they cannot reach torch_remat's
  "a recompute region inside a save region" error or its
  one-name-per-checkpoint-phase rule.

Not in the vocabulary, deliberately: packed expert weights (``GroupedExperts``
holds one stacked parameter, not a ``Linear``), attention's inner softmax (HF
gives it no module of its own), and the norms and activations -- cheap to
recompute, which is the whole point of the mode. A swapped-in llmtuner MoE
block's router gate *is* a ``nn.Linear``, so it stays a region.

Kept engine-free -- no ``torch_remat``, no ``apply_*`` import -- so the
vocabulary and the pattern rule can be imported, logged and tested without the
optional dependency, the way ``parallel/stages.py`` keeps the assembly order
engine-free.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence

import torch.nn as nn

__all__ = [
    "region_names",
    "region_policy",
    "should_recompute",
    "unmatched_save_patterns",
]


def region_names(block: nn.Module) -> tuple[str, ...]:
    """Every region of ``block``, in ``named_modules`` order.

    Names are relative to the block, so the same ``save_regions`` patterns apply
    to every decoder layer -- upstream's rule too ("save-region names are
    relative to a transformer block, so the same policy applies to every
    transformer block").
    """
    return tuple(
        name
        for name, module in block.named_modules()
        if name and isinstance(module, nn.Linear)
    )


def should_recompute(region: str, save_patterns: Sequence[str]) -> bool:
    """Whether ``region`` is recomputed during backward instead of retained.

    Upstream's ``Module.remat_should_recompute``: a region is saved when any
    pattern matches its qualified name, recomputed otherwise. ``fnmatch`` rather
    than ``fnmatchcase`` for the same reason upstream uses it -- on the POSIX
    ranks llmtuner targets the two are identical, and matching upstream keeps a
    cross-platform run from silently changing policy.
    """
    return not any(fnmatch.fnmatch(region, pattern) for pattern in save_patterns)


def unmatched_save_patterns(
    regions: Sequence[str], save_patterns: Sequence[str]
) -> tuple[str, ...]:
    """The patterns that matched no region of ``regions``.

    Upstream keeps these silent (its ``RegionAC`` carries a TODO to validate
    them across pipeline stages). llmtuner logs them at apply time instead: a
    typo in a pattern is otherwise indistinguishable from "this region is not in
    the model", and the log line also carries the names that *are* available.
    """
    return tuple(
        pattern
        for pattern in save_patterns
        if not any(fnmatch.fnmatch(region, pattern) for region in regions)
    )


def region_policy(
    regions: Sequence[str], save_patterns: Sequence[str]
) -> dict[str, bool]:
    """Map each region to its recompute decision (the applied policy).

    Both the apply-time log line and its test read this, so what gets logged is
    the policy actually handed to torch_remat and not a re-derivation of it.
    """
    return {
        region: should_recompute(region, save_patterns) for region in regions
    }
