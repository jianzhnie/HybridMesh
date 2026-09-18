"""Batch-invariant mode: a global switch the flex mask builder reads.

Vendored from torchtitan ``distributed/utils.py`` (the flag and its accessor
only). In batch-invariant mode every matmul must produce the same result
regardless of batch composition, which forces the attention mask builder down a
different path -- see ``..models.common.masks.create_attention_mask``.

Kept as a plain module-level flag rather than a config field because the
readers are deep inside mask construction, which has no handle on the training
config.

TODO: wire an ``enable_batch_invariant_mode()`` that actually installs the
math-mode kernels; today the flag is settable but nothing configures it.
"""

from __future__ import annotations

__all__ = ["is_in_batch_invariant_mode", "set_batch_invariant_mode"]

_batch_invariant_enabled: bool = False


def is_in_batch_invariant_mode() -> bool:
    """Return whether batch-invariant mode is active."""
    return _batch_invariant_enabled


def set_batch_invariant_mode(enabled: bool) -> None:
    """Enable or disable batch-invariant mode."""
    global _batch_invariant_enabled
    _batch_invariant_enabled = enabled
