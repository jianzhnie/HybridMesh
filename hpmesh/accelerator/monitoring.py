"""Device probes, coloured output, and hardware peak throughput.

These are the pieces ``components/metrics.py`` needs in order to describe the
machine it is running on, kept out of that module because they are about the
hardware rather than about metrics.

Ported from torchtitan's ``tools/utils.py``, narrowed to the probes hpmesh
actually uses. Two departures:

* **One device type, decided once.** torchtitan resolves ``device_type`` with
  ``torch._utils._get_available_device_type()`` and then reaches for ``torch.xpu``,
  ``torch.neuron`` and friends at the point of use. hpmesh resolves its
  accelerator device in ``accelerator/device.py`` (NPU/CUDA/MLU/MUSA; see the
  note there on why MPS is excluded) and everything else stays out. A probe
  for a device this package cannot train on is a branch nothing can exercise.

* **No ``subprocess`` for ``lspci``.** torchtitan shells out to ``lspci`` to
  recover a fuller H100 variant string (NVL / PCIe / SXM), because
  ``get_device_name`` reports only "NVIDIA H100". hpmesh does not: shelling out
  from a training process is a side channel that can hang or fail in containers,
  and the probe buys nothing on a machine where the GPU is not an H100. The
  variant-less name selects the SXM figure, which is what the comment in
  ``get_peak_flops`` already calls the default.

* **The memory-history probes live here too.** They are the same kind of thing
  (a question about the machine) and they carry the same trap: they must be
  called on the device module, not on ``torch``. See :func:`record_memory_history`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields

import torch

from .device import device_module, device_type

__all__ = [
    "Color",
    "NoColor",
    "colors_enabled",
    "get_peak_flops",
    "get_device_name",
    "get_device_capacity_bytes",
    "record_memory_history",
    "read_memory_snapshot",
]


@dataclass(frozen=True)
class Color:
    """ANSI escapes, by name. Frozen so a field cannot be reassigned."""

    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"
    orange = "\033[38;2;180;60;0m"
    turquoise = "\033[38;2;54;234;195m"


@dataclass(frozen=True)
class NoColor:
    """The same fields, all empty -- so callers format one way, always."""

    black = ""
    red = ""
    green = ""
    yellow = ""
    blue = ""
    magenta = ""
    cyan = ""
    white = ""
    reset = ""
    orange = ""
    turquoise = ""


# If the two ever drift, a caller that formats with ``color.orange`` (say) works
# under Color and raises under NoColor -- on whichever machine chose the other.
assert {f.name for f in fields(Color)} == {f.name for f in fields(NoColor)}, (
    "NoColor must expose exactly the fields Color does."
)


def _stdout_supports_color() -> bool:
    """Whether writing ANSI escapes to stdout will be interpreted, not shown.

    Not the same question as ``isatty``: a terminal under ``TERM=dumb`` accepts
    the tty check and then prints the escapes literally, and ``FORCE_COLOR`` is
    how a user inside a multiplexer says they want them anyway.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def colors_enabled(*, disable_color_printing: bool) -> bool:
    """Whether colour escapes should be emitted.

    ``disable_color_printing`` is the config-level switch; the environment is
    the terminal-level one, and either can veto. Keeping them separate is what
    lets a redirected log stay clean without the job config having to know it is
    being redirected.

    torchtitan has only the config switch, so an unpiped run and a run whose
    stdout is a file get the same bytes. For a training log that is read back
    out of a file rather than watched, escapes are noise the reader has to
    strip, which is why the second check is here.
    """
    return not disable_color_printing and _stdout_supports_color()


def get_device_name() -> str:
    """The training device's name, or ``"cpu"`` when there is no device to ask.

    ``device_module`` is ``torch`` itself in the CPU case (see ``accelerator/device``),
    and ``torch`` exposes no ``get_device_name`` -- so the type is checked rather
    than assumed.
    """
    if device_type == "cpu":
        return "cpu"
    return device_module.get_device_name(0)


def get_device_capacity_bytes() -> int:
    """Total memory of the training device in bytes, or 0 for CPU."""
    if device_type == "cpu":
        return 0
    return device_module.get_device_properties(0).total_memory


def get_peak_flops(device_name: str) -> float:
    """Peak BF16 tensor throughput in FLOPS for a known accelerator.

    Used to turn a measured tokens/s into a model-FLOPs-utilization percentage.
    The figures are datasheet numbers, so they describe the chip rather than the
    run: a smaller-model run reports a low MFU and that is the correct answer,
    not a measurement error.

    Returns 0.0 for a device with no entry, rather than a plausible-looking
    substitute. MFU is a ratio against this number, so a guess would be reported
    as a fact -- and the caller suppresses MFU entirely when it is 0.
    """
    name = device_name.casefold()

    # Sorted so that a more specific name is tested before a substring of it:
    # "GB300" contains "B300", and "MI250X" must not be caught by "MI250".
    if "a100" in name:
        # https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    if "a6000" in name:
        # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/
        # NOTE: 309.7 TFLOPS is with sparsity; the dense value is half.
        return 154.85e12
    if "h100" in name:
        # https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: specifications are one-half lower without sparsity.
        if "nvl" in name:
            return 835e12
        if "pcie" in name:
            return 756e12
        return 989e12  # H100 SXM, and any variant the name does not distinguish
    if "h200" in name:
        # https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    if "h20" in name:
        # Region-specific variant with no first-hand figure on NVIDIA's global
        # site. 148 TFLOPS BF16 is the reported tensor peak.
        return 148e12
    if "gb200" in name or "gb300" in name:
        # Grace Blackwell Superchips: 2,500 TFLOPS BF16 dense per GPU, half of
        # the 5,000 with sparsity.
        return 2.5e15
    if "b300" in name or "b200" in name:
        # https://resources.nvidia.com/en-us-blackwell-architecture
        return 2.25e15
    if "mi350x" in name:
        # https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html
        return 2300e12
    if "mi355x" in name:
        return 2500e12
    if "mi300x" in name or "mi325x" in name:
        return 1300e12
    if "mi250x" in name:
        # Per GCD.
        return 191.5e12
    if "l40s" in name:
        # https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413
        return 362e12
    if "data center gpu max 1550" in name:
        # Ponte Vecchio, measured rather than quoted: the datasheet figure
        # depends on the compute-unit mode, which the device itself reports.
        max_compute_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_compute_units * 1300 * 10**6
    if name.startswith("tpu"):
        # Dense BF16 matrix-engine peak per device.
        # https://cloud.google.com/tpu/docs/system-architecture-tpu-vm
        if "v7" in name:
            # Published per-chip; v7 exposes each TensorCore as a device.
            return 2307e12 / 2
        if "v6e" in name:
            return 918e12
        if "v5p" in name:
            return 459e12
        if "v5e" in name:
            return 197e12
        if "v4" in name:
            return 275e12

    return 0.0


def _memory_module():
    """The module that owns memory history for the training device, or None.

    Not ``torch.cuda.memory`` unconditionally, the way torchtitan does it: its
    fallback for a non-CUDA device is ``torch.memory``, which is not a real
    module. Calling these on a CPU-only machine would raise AttributeError from
    a probe that is supposed to be optional.
    """
    if device_type == "cpu":
        # The CPU allocator keeps no history to record or snapshot.
        return None
    memory = getattr(device_module, "memory", None)
    if memory is not None and hasattr(memory, "_record_memory_history"):
        return memory
    return None


def record_memory_history(*, max_entries: int) -> bool:
    """Begin collecting the allocator's history; report whether it started.

    ``stacks="python"`` records Python frames only. That is a deliberate
    narrowing of what the default ``stacks="all"`` captures: symbolizing C++
    frames is slow enough that dumping a snapshot mid-training can take
    minutes, which turns a diagnostic into a stall. The Python frames are
    what names the training code in the resulting report anyway.

    Returns False when the device has no allocator history, so the caller can
    say the snapshot is unavailable rather than write an empty file.
    """
    memory = _memory_module()
    if memory is None:
        return False
    memory._record_memory_history(stacks="python", max_entries=max_entries)
    return True


def read_memory_snapshot():
    """The accumulated allocator history, for writing to a snapshot file."""
    memory = _memory_module()
    if memory is None:
        return None
    return memory._snapshot()
