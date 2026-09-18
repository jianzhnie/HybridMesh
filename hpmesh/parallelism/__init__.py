"""Parallelism: one apply_* per dimension, added incrementally along the path.

Each dimension is intentionally minimal -- the learning goal is to understand the
ONE core mechanism of each, not to reproduce a production framework's surface.
Every apply_* is a no-op when its degree is 1, so the same trainer code runs from
step 0 (single device) through step 4 (full hybrid parallelism).
"""

from .cp_ep import apply_cp_ep
from .fsdp import apply_fsdp
from .pp import apply_pp
from .tp import apply_tp

__all__ = ["apply_fsdp", "apply_tp", "apply_pp", "apply_cp_ep"]
