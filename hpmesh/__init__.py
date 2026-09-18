"""hpmesh -- Hybrid Parallel training over a Torch DeviceMesh.

A minimal, learning-oriented framework for understanding the core modules of
distributed training (FSDP / TP / PP / CP / EP) by building them from scratch.

Two abstractions only:
  * a grouped ``HybridMeshConfig``      (hpmesh.trainer.config)
  * a ``HFTransformerModel``            (hpmesh.models.hf_wrapper)

Parallelism dimensions are added one at a time; each ``apply_*`` in
``hpmesh.parallel`` is a no-op when its degree is 1, so the same training loop
runs from a single device up to full hybrid parallelism.
"""

from .trainer import HybridMeshConfig

__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy import to avoid a circular import: Trainer depends on the config.
    # Importing Trainer only on first access breaks the cycle.
    if name == "Trainer":
        from .trainer.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["HybridMeshConfig", "Trainer", "__version__"]
