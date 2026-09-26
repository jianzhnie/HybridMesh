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

# Lazy (PEP 562): ``matrix`` is read by ``hpmesh/config``, which must stay
# importable without the engine layer (context_parallel pulls in the model
# stack and its spmd surface). Eager re-exports would make any submodule
# import -- matrix included -- pay for the whole engine.
_EXPORT_SOURCES = {
    "apply_cp": "context_parallel",
    "apply_ep": "expert_parallel",
    "apply_fsdp": "fully_shard.apply",
    "apply_tp": "tensor_parallel",
    "parallelize_hf_transformers": "parallelize",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    """Resolve an ``apply_*`` entry point on first touch (PEP 562)."""
    source = _EXPORT_SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{source}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
