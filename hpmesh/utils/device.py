"""Device discovery and small backend-neutral runtime helpers.

NPU is preferred because it is HybridMesh's primary accelerator, followed by
CUDA and other torch accelerators that provide distributed collectives. MPS is
intentionally excluded: it has no distributed backend and cannot host a
``DeviceMesh`` even when torch reports it as available.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import torch

try:
    importlib.import_module("torch_npu")
except ImportError:
    pass

ACCELERATOR_TYPES = frozenset(("npu", "cuda", "xpu", "mlu", "musa"))
DEVICE_PRIORITY = ("npu", "cuda", "musa", "mlu", "xpu")
_BACKENDS = {
    "npu": "hccl",
    "cuda": "nccl",
    "xpu": "ccl",
    "mlu": "cncl",
    "musa": "musa",
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


def get_dist_info() -> tuple[int, int, int]:
    """Return ``(rank, world size, local rank)`` from torchrun's environment."""
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


def get_current_device(*, use_cpu: bool = False) -> torch.device:
    """Return this process's device using ``LOCAL_RANK``."""
    if use_cpu or device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, get_dist_info()[2])


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


def synchronize(device: torch.device | None = None) -> None:
    """Synchronize pending accelerator work; CPU is already synchronous."""
    device = get_current_device() if device is None else device
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def empty_cache() -> None:
    """Release unused cache blocks when the backend provides that operation."""
    if device_type != "cpu":
        function = getattr(device_module, "empty_cache", None)
        if function is not None:
            function()


def is_device_available(device: torch.device) -> bool:
    """Return whether a concrete device index is accessible."""
    if not is_device_type_available(device.type):
        return False
    if device.type == "cpu" or device.index is None:
        return True
    return device.index < getattr(torch, device.type).device_count()


def validate_device(device: torch.device) -> None:
    """Raise when ``device`` cannot be used by this process."""
    if not is_device_available(device):
        raise RuntimeError(f"Device {device} is not available")


def should_use_pin_memory(device: torch.device | None = None) -> bool:
    """Return whether asynchronous pinned-memory copies are useful."""
    device = get_current_device() if device is None else device
    return device.type in {"cuda", "npu"}


device_type, device_module = get_device_info()
