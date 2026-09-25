"""The capability registry: one place that answers "does this build have X?".

Torch-version and environment probes (``hasattr`` on private config modules,
guarded imports) used to be scattered across a dozen call sites, each
re-answering the same question its own way. They live here now, one named
entry per capability, each documented with what it is, which torch version or
package introduces it, and who consumes it. Probes are cached: the answer
cannot change within a process.

Use ``has(name)`` at a guard point, keeping the site's own error message (the
message text is a tested contract); ``require(name, feature=...)`` is the
convenience form for new code, raising ``EnvironmentUnsupportedError`` with
the entry's unlock hint. An unknown name raises immediately -- a typo must
not silently read as "capability absent".

Not everything import-shaped belongs here; see docs/hybridmesh_design.md
§3.2. Optional *packages* (renderers, torchao, torchvision) keep raising
plain ``ImportError`` at their own sites. Device discovery
(``accelerator/device.py``) is availability probing with silent-absent
semantics, a different question. Hard imports with no fallback (DTensor,
flex_attention) have nothing to probe.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import cache

import torch

from hpmesh.errors import EnvironmentUnsupportedError

__all__ = ["CAPABILITIES", "has", "require"]


def _hasattr_torch(module: str, attr: str) -> Callable[[], bool]:
    """Probe factory: ``hasattr(<torch submodule attr chain>, attr)``."""

    def probe() -> bool:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            return False
        return getattr(mod, attr, None) is not None

    return probe


def _importable(module: str, attr: str | None = None) -> Callable[[], bool]:
    """Probe factory: the module imports (and optionally exposes ``attr``)."""

    def probe() -> bool:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            return False
        return attr is None or getattr(mod, attr, None) is not None

    return probe


def _grouped_mm_runs() -> bool:
    """Whether ``torch._grouped_mm`` can run here.

    Probed by doing it, rather than by checking the device or the torch
    version: the op is reachable on CPU as well as CUDA, and it imposes shape
    constraints of its own (strides must be 16-byte multiples, so the
    innermost dim has to be at least 8 bf16 elements). A version or device
    test would be wrong on both counts and would go stale silently.

    The probe is necessarily approximate -- a shape that satisfies the op
    need not be one a real layer uses. It is deliberately shaped like the
    real call (``(T, K) @ (E, K, N)``, bf16, int32 offsets) so that it fails
    for the same reasons a real call would.
    """
    grouped_mm = getattr(torch, "_grouped_mm", None)
    if grouped_mm is None:
        return False
    try:
        grouped_mm(
            torch.zeros(8, 8, dtype=torch.bfloat16),
            torch.zeros(2, 8, 8, dtype=torch.bfloat16),
            offs=torch.tensor([4, 8], dtype=torch.int32),
        )
    except Exception:
        return False
    return True


class _Capability:
    """One registry entry: probe, provenance, unlock hint, consumers."""

    def __init__(
        self,
        probe: Callable[[], bool],
        *,
        what: str,
        since: str,
        hint: str,
        consumers: str,
    ) -> None:
        self.probe = probe
        self.what = what
        self.since = since
        self.hint = hint
        self.consumers = consumers


CAPABILITIES: dict[str, _Capability] = {
    # -- compile-time knobs (consumers: parallel/compile.py) -------------------
    "dynamo_capture_scalar_outputs": _Capability(
        _hasattr_torch("torch._dynamo.config", "capture_scalar_outputs"),
        what="torch._dynamo.config.capture_scalar_outputs",
        since="torch 2.7 (dynamo config flag)",
        hint="Upgrade torch, or run the token-choice MoE without compile.",
        consumers="parallel/compile.py (token-choice MoE dispatch compile)",
    ),
    "inductor_micro_pipeline_tp": _Capability(
        _hasattr_torch("torch._inductor.config", "_micro_pipeline_tp"),
        what="torch._inductor.config._micro_pipeline_tp",
        since="torch 2.8 (inductor micro-pipeline TP pass)",
        hint="Upgrade torch, or run without async TP.",
        consumers="parallel/compile.py (compile_config.enable_async_tensor_parallel)",
    ),
    "fx_regional_inductor": _Capability(
        _importable("torch.fx.passes.regional_inductor", "regional_inductor"),
        what="torch.fx.passes.regional_inductor (+ torch._dynamo.backends.common)",
        since="torch 2.10 (fx regional-inductor pass)",
        hint="Upgrade torch, use backend='inductor', or run without compile.",
        consumers="parallel/compile.py (aot_eager backend on flex models)",
    ),
    # -- symmetric memory (consumers: parallel/compile.py, tensor_parallel) ----
    "symm_mem": _Capability(
        _importable("torch.distributed._symmetric_memory", "enable_symm_mem_for_group"),
        what="torch.distributed._symmetric_memory.enable_symm_mem_for_group",
        since="torch 2.8 (symmetric-memory collectives, CUDA-only)",
        hint="Upgrade torch, or run without async TP / symm-mem collectives.",
        consumers="parallel/compile.py (async TP), tensor_parallel/tp.py "
        "(fused symm-mem TP collectives), tensor_parallel/linear.py",
    ),
    # -- activation checkpointing (consumer: parallel/activation_checkpoint.py)
    "functorch_activation_memory_budget": _Capability(
        _hasattr_torch("torch._functorch.config", "activation_memory_budget"),
        what="torch._functorch.config.activation_memory_budget",
        since="torch 2.6 (functorch partitioner budget knob)",
        hint="Upgrade torch, or use activation_checkpoint_mode='full'/'selective'.",
        consumers="parallel/activation_checkpoint.py (mode='memory_budget')",
    ),
    # -- model kernels (consumer: models/common/grouped_experts.py) ------------
    "torch_grouped_mm": _Capability(
        _grouped_mm_runs,
        what="torch._grouped_mm",
        since="torch 2.7 (grouped GEMM, bf16; CPU-reachable but shape-constrained)",
        hint="Upgrade torch, or leave GroupedExperts on the looped-GEMM fallback.",
        consumers="models/common/grouped_experts.py (GroupedExperts forward)",
    ),
}


@cache
def _probe(name: str) -> bool:
    entry = CAPABILITIES.get(name)
    if entry is None:
        raise KeyError(
            f"unknown capability {name!r}; registered: {sorted(CAPABILITIES)}"
        )
    return bool(entry.probe())


def has(capability: str) -> bool:
    """Whether ``capability`` is present in this build (cached)."""
    return _probe(capability)


def require(capability: str, *, feature: str) -> None:
    """Raise ``EnvironmentUnsupportedError`` unless ``has(capability)``.

    Guard points with an existing (tested) message should keep their own
    raise and use ``has``; this is the convenience form for new code.
    """
    if has(capability):
        return
    entry = CAPABILITIES[capability]  # has() already validated the name
    raise EnvironmentUnsupportedError(
        f"{feature} needs {entry.what}, which this torch "
        f"({torch.__version__}) does not carry. {entry.hint} "
        f"(capability {capability!r}, introduced {entry.since}.)"
    )
