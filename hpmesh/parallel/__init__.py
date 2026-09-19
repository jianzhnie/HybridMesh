"""Parallelism: one ``apply_*`` per dimension, composed by one entry point.

Each dimension gets its own module, except the two that come as a pair and are
therefore subpackages: ``tensor_parallel/{tp,linear}`` (the declaration and the
fused GEMMs it realizes into) and ``fsdp2/{fsdp,fsdp_wrap}`` (torchtitan's
vendored sharding logic behind a thin driver).

Every ``apply_*`` is a no-op when its degree is 1. That is what lets the same
trainer code run from step 0 (single device) through step 4 (full hybrid
parallelism) without an ``if degree > 1`` at any call site -- and it is why
``parallelize_hf_transformers`` can apply all of them unconditionally.

Not here: pipeline parallelism. ``pipeline_parallel/pipeline.py`` computes the
stage split, but nothing builds a schedule over it, so ``pp > 1`` raises rather
than quietly training every rank on the whole model.
"""

from __future__ import annotations

from .cp_ep import apply_cp_ep
from .fsdp2.fsdp_wrap import apply_fsdp
from .parallelize_hf import parallelize_hf_transformers
from .tensor_parallel.tp import apply_tp

__all__ = [
    "apply_cp_ep",
    "apply_fsdp",
    "apply_tp",
    "parallelize_hf_transformers",
]
