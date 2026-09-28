"""Head-count divisibility, enforced by whichever axis is being applied.

torchtitan checks ``num_attention_heads % (tensor_parallel_degree *
context_parallel_degree) == 0`` once, when it parses the run config
(``config/validation.py``, ``validate_context_parallel``), because it builds the
model from its own config and so has the head counts before the model exists.
llmtuner builds the model from the checkpoint first and reads the counts off its
HF config, so the same invariant is checked at the wire-up seam instead:
``apply_tp`` asks for ``% tp`` and the ulysses CP path asks for ``% (tp * cp)``.
Run together they are upstream's single check -- each fires for the axis that
made it necessary, so a pure-TP run no longer has to switch CP on to be told
that ``tp=16`` does not divide 8 KV heads.

Why the projection-level guard cannot carry this: ``shard_weight`` only checks
that the *feature* dim divides, and 8 KV heads at ``head_dim=128`` is 1024
features -- divisible by 16 even though the 8 heads are not. Nothing between
there and HF's head reshape rejects that, so the failure surfaces deep inside
attention, if at all: a shape error on a run that already spent its startup
budget.

Only checked where the model exposes the counts: a stub, or a chunk that carries
no attention config, is skipped rather than rejected. The guard's job is to
refuse counts it can see and divide badly, not to demand a config.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["head_counts", "require_heads_divisible_by"]


def head_counts(model: nn.Module) -> list[tuple[str, int]]:
    """``model``'s attention head counts as ``(config field, count)`` pairs.

    Reads the HF config the wrapper holds under ``model.model``; returns an
    empty list when there is none. GQA configs may omit ``num_key_value_heads``
    entirely -- then every head is a KV head, so the count falls back to
    ``num_attention_heads``, exactly as upstream's
    ``n_kv_heads = getattr(attention, "n_kv_heads", None) or n_heads`` does.
    """
    config = getattr(getattr(model, "model", None), "config", None)
    n_heads = getattr(config, "num_attention_heads", None)
    if not n_heads:
        return []
    n_kv_heads = getattr(config, "num_key_value_heads", None) or n_heads
    return [
        ("num_attention_heads", n_heads),
        ("num_key_value_heads", n_kv_heads),
    ]


def require_heads_divisible_by(
    model: nn.Module,
    *,
    size: int,
    axis: str,
    why: str,
) -> None:
    """Raise unless every attention head count divides ``size``.

    ``axis`` names the parallelism whose size this is (``"tp"``, ``"tp*cp"``)
    and ``why`` gives the one-line reason that axis divides heads, so each call
    site carries its own argument rather than sharing a vague one. A size of 1 is
    the disabled axis and is never checked.
    """
    if size <= 1:
        return
    for field, count in head_counts(model):
        if count % size:
            raise ValueError(
                f"{field} ({count}) must be divisible by {axis} ({size}): {why}"
            )
