"""Device information, shared by the parallelism modules.

``device_type`` is the device used for *training*: "cuda" when a CUDA GPU is
available, otherwise "cpu". ``device_module`` is the matching ``torch`` submodule.

Deliberately NOT ``torch._utils._get_available_device_type()``: on Apple Silicon
that resolves to "mps", but MPS exposes no distributed collectives, so torchrun
still runs the process group (gloo) on CPU. Building a mesh on "mps" then fails.
torchtitan can use the raw helper because it targets CUDA machines.
"""

from __future__ import annotations

import torch


def get_device_info() -> tuple[str, object]:
    """Return ``(device_type, device_module)`` for distributed training."""
    if torch.cuda.is_available():
        return "cuda", torch.cuda
    return "cpu", torch


device_type, device_module = get_device_info()
