"""Parallelism: one ``apply_*`` per dimension, composed by one entry point.

Each dimension gets its own module or subpackage: ``tensor_parallel/{tp,linear}``
(the declaration and the fused GEMMs it realizes into), ``fully_shard/{fsdp,
apply}`` (torchtitan's vendored sharding logic behind a thin driver),
``context_parallel/`` (CP redistribution primitives, flex kernel, input
sharding, ``apply_cp``) and ``expert_parallel/`` (the HF MoE swap and
``apply_ep``).

Every ``apply_*`` is a no-op when its degree is 1. That is what lets the same
trainer code run from step 0 (single device) through step 4 (full hybrid
parallelism) without an ``if degree > 1`` at any call site -- and it is why
``parallelize_hf_transformers`` can apply all of them unconditionally.

Pipeline parallelism lives in ``pipeline_parallel/``: ``pipeline.py`` computes
the stage split and ``apply.py`` builds the schedule over this rank's stages.
``parallelize_hf_transformers`` dispatches to it when ``pp > 1`` and returns a
``PipelineParallelSetup`` instead of a model.
"""

from __future__ import annotations

from .context_parallel import apply_cp
from .expert_parallel import apply_ep
from .fully_shard.apply import apply_fsdp
from .parallelize_hf import parallelize_hf_transformers
from .tensor_parallel import apply_tp

__all__ = [
    "apply_cp",
    "apply_ep",
    "apply_fsdp",
    "apply_tp",
    "parallelize_hf_transformers",
]
