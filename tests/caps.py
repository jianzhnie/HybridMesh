"""Environment capability probes for the test suite.

Each entry answers "can this test module even be imported here": an import
attempt (sys.modules first, so the stub runner's injected fakes count as
present) or an attribute probe. Distinct from ``hpmesh.accelerator.capabilities``
on purpose: that is the *runtime* registry for torch knobs a built feature
checks; this table is about import-level hard dependencies of test modules
(DTensor, grain, pipelining, ...), which fail at collection, not at a guard.

A test module declares what it needs with a top-of-file comment::

    # requires-env: dtensor, spmd_types

and the conftest collection hook skips the whole module, with a uniform
``[env] missing: ...`` reason, when a probe fails. Modules that import fine
need no marker.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable


def _has_spec(module: str) -> bool:
    """find_spec, with a missing parent package meaning False (not raising)."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _importable(module: str) -> bool:
    """Whether ``module`` imports (or was already injected, e.g. by a stub)."""
    if module in sys.modules:
        return True
    if not _has_spec(module):
        return False
    try:
        importlib.import_module(module)
    except Exception:  # noqa: BLE001 - any import-time failure means absent
        return False
    return True


def _importable_attr(module: str, attr: str) -> bool:
    """Whether ``module`` imports and exposes ``attr``."""
    if module not in sys.modules:
        if not _has_spec(module):
            return False
        try:
            importlib.import_module(module)
        except Exception:  # noqa: BLE001 - as above
            return False
    return hasattr(sys.modules[module], attr)


def _transformers_has(symbol: str) -> bool:
    return _importable_attr("transformers", symbol)


def _torch_param_names() -> bool:
    """Whether optimizer ``state_dict()`` param groups carry ``param_names``."""
    if not _importable_attr("torch.optim", "Adam"):
        return False
    import torch

    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    return "param_names" in opt.state_dict()["param_groups"][0]


# name -> probe. The name is what the marker and the skip reason print.
CAPS: dict[str, Callable[[], bool]] = {
    # torch.distributed.tensor.DTensor (the DTensor/DTensorSpec import surface)
    "dtensor": lambda: _importable_attr("torch.distributed.tensor", "DTensor"),
    # the spmd_types shims the model/parallel layers import
    "spmd_types": lambda: _importable("spmd_types"),
    # Grain data graph (datasets/)
    "grain": lambda: _importable("grain.python"),
    # flex attention (BlockMask & friends)
    "flex_attention": lambda: _importable_attr(
        "torch.nn.attention.flex_attention", "create_block_mask"
    ),
    # torch.distributed.pipelining (PipelineStage, schedules)
    "pipelining": lambda: _importable("torch.distributed.pipelining"),
    # torch.distributed.checkpoint's HF storage surface the checkpointer reads
    "dcp": lambda: _importable_attr(
        "torch.distributed.checkpoint", "HuggingFaceStorageWriter"
    ),
    # torch.utils.checkpoint.CheckpointPolicy (selective AC)
    "checkpoint_policy": lambda: _importable_attr(
        "torch.utils.checkpoint", "CheckpointPolicy"
    ),
    # Weights & Biases logger
    "wandb": lambda: _importable("wandb"),
    # torch optimizer state_dicts carry param_names (newer torch)
    "torch_param_names": _torch_param_names,
    # torch._functorch.partitioners (selective-AC default op list)
    "functorch_partitioners": lambda: _importable_attr(
        "torch._functorch.partitioners", "get_default_op_list"
    ),
    # transformers' qwen3 (and family) layouts used by the EP/TP tests
    "transformers_qwen3": lambda: _transformers_has("Qwen3MoeForCausalLM"),
    # transformers' GPT-OSS layout
    "transformers_gpt_oss": lambda: _transformers_has("GptOssForCausalLM"),
}


def missing(names: list[str]) -> list[str]:
    """The subset of ``names`` whose probe fails; unknown names raise."""
    unknown = [n for n in names if n not in CAPS]
    if unknown:
        raise KeyError(f"unknown test capability: {unknown}")
    return [n for n in names if not CAPS[n]()]


def require_env(*names: str) -> None:
    """Module-level gate: skip the whole module when a capability is absent.

    Call at the top of a test module, before the imports that need the named
    capabilities. The skip reason is uniform -- ``[env] missing: <names>`` --
    so ``pytest -rs`` prints the environment coverage report.
    """
    import pytest

    absent = missing(list(names))
    if absent:
        pytest.skip(f"[env] missing: {', '.join(absent)}", allow_module_level=True)
