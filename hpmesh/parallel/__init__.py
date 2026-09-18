"""Parallelism: one apply_* per dimension, composed by one entry point.

Each dimension is intentionally minimal -- the learning goal is to understand the
ONE core mechanism of each, not to reproduce a production framework's surface.
Every apply_* is a no-op when its degree is 1, so the same trainer code runs from
step 0 (single device) through step 4 (full hybrid parallelism).
``parallelize_hf_transformers`` applies them in the one order that works
(tp/cp/ep -> compile -> fsdp).

``fsdp.py`` is torchtitan's ``distributed/fsdp.py`` (imports rewritten, its
``Decoder`` annotations dropped, layer iteration generalized to HF's container
layout); ``fsdp_wrap.py`` is the thin entry point that decides whether to shard
at all and drives it.
"""

from __future__ import annotations

from .cp_ep import apply_cp_ep
from .fsdp_wrap import apply_fsdp
from .parallelize_hf import parallelize_hf_transformers
from .pp import apply_pp
from .tp import apply_tp

__all__ = [
    "apply_fsdp",
    "apply_tp",
    "apply_pp",
    "apply_cp_ep",
    "parallelize_hf_transformers",
]
