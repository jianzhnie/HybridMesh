"""Full-precision fingerprint of a training run's loss trajectory.

``MetricsProcessor.log`` prints the loss with ``f"{x:8.5f}"`` -- five
significant digits, which cannot tell a bitwise-identical loss from one that
differs in the 6th. Anything that claims to be numerically neutral (an
activation-checkpointing change, a refactor, a rename) has to be proven at
full precision, or the claim is unfalsifiable.

``capture()`` wraps the processor and returns two lists of records:

* ``stable`` -- loss, max_loss, grad_norm, and the extra metrics. Pure
  functions of ``(seed, step)``; two runs of the same build reproduce them
  bitwise, so they are what gets sha256-compared.
* ``wallclock`` -- tps/tflops/mfu. Ratios against elapsed wall time, so they
  move run to run and must NOT enter the hash. Kept separately so a reader can
  see the derived metrics (e.g. whether mfu came back None) without those
  values polluting the comparison.

The torch import is deliberate: these values are produced by torch kernels and
a different torch build legitimately changes them. A mismatch across a torch
upgrade is expected; a mismatch across a source change with the same torch is
the thing this exists to catch.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hpmesh.components.metrics import MetricsProcessor


class Fingerprint:
    """A captured run, with the stable and wall-clock halves kept apart."""

    def __init__(self) -> None:
        self.stable: list[dict[str, Any]] = []
        self.wallclock: list[dict[str, Any]] = []

    def sha256(self) -> str:
        """Hash of the stable half only -- the part that must not change."""
        blob = json.dumps(self.stable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def losses(self) -> list[str]:
        return [rec["loss"] for rec in self.stable]

    def grad_norms(self) -> list[str]:
        return [rec["grad_norm"] for rec in self.stable]

    def __repr__(self) -> str:
        return f"<Fingerprint steps={len(self.stable)} sha256={self.sha256()[:12]}...>"


def capture_metrics(processor_cls=MetricsProcessor) -> Fingerprint:
    """Wrap ``MetricsProcessor.log`` and record at full precision.

    Returns the ``Fingerprint`` being filled. The wrap is installed here and
    must be undone by :func:`restore` -- a leaked wrap would keep appending
    across tests and silently merge two runs into one fingerprint.
    """
    fp = Fingerprint()
    original = processor_cls.log

    def log(
        self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None
    ):
        fp.stable.append(
            {
                "step": step,
                # repr() of a float is 17 significant digits and round-trips
                # exactly, so the dump is a faithful fingerprint.
                "loss": repr(global_avg_loss),
                "max_loss": repr(global_max_loss),
                "grad_norm": repr(grad_norm),
                "num_flops_per_token": repr(self.num_flops_per_token),
                "gpu_peak_flops": repr(self.gpu_peak_flops),
                "extra": {k: repr(v) for k, v in sorted((extra_metrics or {}).items())},
            }
        )
        # ``log`` anchors step_last_log on its first call; do the same here so
        # the derivation below is legal at this point in the order, then let
        # the real log do it too (its own guard makes that idempotent).
        if self.step_last_log is None:
            self.step_last_log = step - 1
        try:
            derived = self._derive(step)
        except Exception as exc:  # pragma: no cover - diagnostic only
            fp.wallclock.append(
                {"step": step, "derive_error": f"{type(exc).__name__}: {exc}"}
            )
        else:
            fp.wallclock.append(
                {
                    "step": step,
                    "tflops": repr(derived.tflops),
                    "mfu": repr(derived.mfu),
                    "mfu_is_none": derived.mfu is None,
                }
            )
        return original(
            self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics
        )

    fp._original = original
    processor_cls.log = log
    return fp


def restore(processor_cls=MetricsProcessor, original=None) -> None:
    """Undo :func:`capture_metrics`."""
    processor_cls.log = original if original is not None else MetricsProcessor.log
