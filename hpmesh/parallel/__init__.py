"""Parallelism: one apply_* per dimension, added incrementally along the path.

Each dimension is intentionally minimal -- the learning goal is to understand the
ONE core mechanism of each, not to reproduce a production framework's surface.
Every apply_* is a no-op when its degree is 1, so the same trainer code runs from
step 0 (single device) through step 4 (full hybrid parallelism).

Imports are guarded so one dimension that targets a newer torch build cannot take
the whole package (and every other dimension) down with it. The failing import is
reported at call time rather than swallowed.
"""

from __future__ import annotations

import torch

from .cp_ep import apply_cp_ep
from .pp import apply_pp
from .tp import apply_tp

try:
    from .fsdp import apply_fsdp
except ImportError as _e:  # pragma: no cover - build dependent
    _fsdp_import_error = _e

    def apply_fsdp(model, mesh, cfg):  # type: ignore[misc]
        # Same no-op contract as the real implementation: FSDP is only needed
        # when there is more than one data-parallel rank.
        if mesh is None or cfg.dp == 1:
            return model
        raise ImportError(
            "hpmesh.parallel.fsdp failed to import -- it targets a torch build "
            f"with torch.distributed.fsdp.DataParallelMeshDims; this environment "
            f"has torch {torch.__version__}. Original error: {_fsdp_import_error}"
        )


__all__ = ["apply_fsdp", "apply_tp", "apply_pp", "apply_cp_ep"]
