"""Device discovery and small backend-neutral runtime helpers.

NPU is preferred because it is HybridMesh's primary accelerator, followed by
CUDA and other torch accelerators that provide distributed collectives. MPS is
intentionally excluded: it has no distributed backend and cannot host a
``DeviceMesh`` even when torch reports it as available.

The per-vendor availability predicates (``is_npu_available`` and friends), the
peak-memory queries and ``is_npu_support_full_precision`` derive from
OpenMMLab's ``mmengine.device`` conventions, merged here so nothing needs the
mmengine dependency. Two pieces of that file were deliberately not carried
over: its import-time ``DEVICE`` constant and ``get_device()`` duplicate this
module's ``device_type`` / ``get_device_type()``, and its
``torch.npu.set_compile_mode`` call mutates global torch state at import time.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Any

import torch

try:
    importlib.import_module("torch_npu")
except ImportError:
    pass

# Register the vendor extensions so ``torch.mlu`` / ``torch.musa`` exist for
# the availability probes below. Each import is a no-op when the vendor's
# torch build is not installed.
for _ext in ("torch_mlu", "torch_musa"):
    try:
        importlib.import_module(_ext)
    except ImportError:
        pass
del _ext

try:
    from torch_npu.npu import utils as _npu_utils
except ImportError:
    _npu_utils = None

ACCELERATOR_TYPES = frozenset(("npu", "cuda", "xpu", "mlu", "musa"))
DEVICE_PRIORITY = ("npu", "cuda", "musa", "mlu", "xpu")
_BACKENDS = {
    "npu": "hccl",
    "cuda": "nccl",
    "xpu": "ccl",
    "mlu": "cncl",
    "musa": "mccl",
    "cpu": "gloo",
}


def is_device_type_available(kind: str) -> bool:
    """Return whether ``kind`` has an accessible torch device."""
    if kind == "cpu":
        return True
    module = getattr(torch, kind, None)
    if module is None:
        return False
    try:
        if not module.is_available():
            return False
        count = getattr(module, "device_count", None)
        return count is None or count() > 0
    except Exception:
        return False


def get_device_type() -> str:
    """Select the training device, honoring ``HPMESH_DEVICE`` when set."""
    override = os.environ.get("HPMESH_DEVICE", "").strip().lower()
    if override:
        if override not in ACCELERATOR_TYPES | {"cpu"}:
            raise ValueError(f"Unsupported HPMESH_DEVICE={override!r}")
        if not is_device_type_available(override):
            raise RuntimeError(f"Requested device type {override!r} is not available")
        return override
    return next(
        (kind for kind in DEVICE_PRIORITY if is_device_type_available(kind)), "cpu"
    )


def get_device_info() -> tuple[str, Any]:
    """Return ``(device type, torch device module)`` for training."""
    kind = get_device_type()
    return kind, torch if kind == "cpu" else getattr(torch, kind)


def get_env_dist_info() -> tuple[int, int, int]:
    """Return ``(rank, world size, local rank)`` from torchrun's environment.

    Named ``get_env_dist_info`` to stay distinct from
    ``dist_utils.get_dist_info(group)``, which queries the live process
    group and returns only ``(rank, world size)``.
    """
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


def get_current_device(*, use_cpu: bool = False) -> torch.device:
    """Return this process's device using ``LOCAL_RANK``."""
    if use_cpu or device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, get_env_dist_info()[2])


def get_distributed_backend() -> str:
    """Return the backend, with ``HPMESH_DIST_BACKEND`` as a diagnostic override."""
    override = os.environ.get("HPMESH_DIST_BACKEND", "").strip().lower()
    return override or _BACKENDS[device_type]


def set_device(device: torch.device) -> None:
    """Set the accelerator device, failing rather than misplacing ranks."""
    if device.type == "cpu":
        return
    module = getattr(torch, device.type, None)
    setter = getattr(module, "set_device", None)
    if setter is None:
        raise RuntimeError(f"Device type {device.type!r} has no set_device API")
    try:
        setter(device)
    except Exception as exc:
        raise RuntimeError(f"Failed to set device {device}: {exc}") from exc


def is_device_available(device: torch.device) -> bool:
    """Return whether a concrete device index is accessible."""
    if not is_device_type_available(device.type):
        return False
    if device.type == "cpu" or device.index is None:
        return True
    return device.index < getattr(torch, device.type).device_count()


def should_use_pin_memory(device: torch.device | None = None) -> bool:
    """Return whether asynchronous pinned-memory copies are useful."""
    device = get_current_device() if device is None else device
    return device.type in {"cuda", "npu"}


# -- Per-vendor availability predicates (mmengine.device-style surface). --
# Thin wrappers over ``is_device_type_available`` where hpmesh knows the type;
# ``mps`` and ``dipu`` are diagnostic-only and stay out of ACCELERATOR_TYPES.


def is_cuda_available() -> bool:
    """Return whether CUDA devices exist."""
    return is_device_type_available("cuda")


def is_npu_available() -> bool:
    """Return whether Ascend PyTorch and NPU devices exist."""
    return is_device_type_available("npu")


def is_mlu_available() -> bool:
    """Return whether Cambricon PyTorch and MLU devices exist."""
    return is_device_type_available("mlu")


def is_musa_available() -> bool:
    """Return whether MUSA PyTorch and devices exist."""
    return is_device_type_available("musa")


def is_mps_available() -> bool:
    """Return whether MPS devices exist (Apple Silicon; no distributed use)."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def is_dipu_available() -> bool:
    """Return whether the DIPU extension is importable."""
    return importlib.util.find_spec("torch_dipu") is not None


def is_npu_support_full_precision() -> bool:
    """Return whether the NPU SoC supports full-precision training."""
    if not is_npu_available() or _npu_utils is None:
        return False
    version_of_support_full_precision = 220
    return _npu_utils.get_soc_version() >= version_of_support_full_precision


def get_max_cuda_memory(device: torch.device | None = None) -> int:
    """Peak CUDA memory occupied by tensors, in MB, and reset the peak.

    With ``device=None`` the current device is reported. Note the side
    effect: the peak counter is reset, so consecutive calls measure the
    interval between calls, not the program maximum.
    """
    mem = torch.cuda.max_memory_allocated(device=device)
    torch.cuda.reset_peak_memory_stats()
    return int(mem) // (1024 * 1024)


def get_max_musa_memory(device: torch.device | None = None) -> int:
    """Peak MUSA memory occupied by tensors, in MB.

    Unlike the CUDA variant there is no reset: ``torch.musa`` does not
    support ``reset_peak_memory_stats`` yet.
    """
    mem = torch.musa.max_memory_allocated(device=device)
    return int(mem) // (1024 * 1024)


device_type, device_module = get_device_info()
