"""Device selection contracts that do not require accelerator hardware."""

from __future__ import annotations

import torch

from hpmesh.accelerator import device


def test_device_priority_prefers_npu(monkeypatch) -> None:
    available = {"npu", "cuda"}
    monkeypatch.setattr(
        device, "is_device_type_available", lambda kind: kind in available
    )

    assert device.get_device_type() == "npu"


def test_current_device_uses_local_rank(monkeypatch) -> None:
    monkeypatch.setattr(device, "device_type", "cuda")
    monkeypatch.setenv("LOCAL_RANK", "3")

    assert device.get_current_device() == torch.device("cuda:3")
    assert device.get_current_device(use_cpu=True) == torch.device("cpu")


def test_distributed_backend_has_explicit_override(monkeypatch) -> None:
    monkeypatch.setattr(device, "device_type", "npu")
    monkeypatch.delenv("HPMESH_DIST_BACKEND", raising=False)
    assert device.get_distributed_backend() == "hccl"

    monkeypatch.setenv("HPMESH_DIST_BACKEND", "gloo")
    assert device.get_distributed_backend() == "gloo"


def test_cpu_is_available_but_not_pinned() -> None:
    cpu = torch.device("cpu")
    assert device.is_device_available(cpu)
    assert not device.should_use_pin_memory(cpu)
