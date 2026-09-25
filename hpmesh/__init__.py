"""hpmesh -- Hybrid Parallel training over a Torch DeviceMesh.

A minimal, learning-oriented framework for understanding the core modules of
distributed training (FSDP / TP / PP / CP / EP) by building them from scratch.

Two abstractions only:
  * a grouped ``HybridMeshConfig``      (hpmesh.config)
  * a ``HFTransformerModel``            (hpmesh.models.hf_wrapper)

Parallelism dimensions are added one at a time; each ``apply_*`` in
``hpmesh.parallel`` is a no-op when its degree is 1, so the same training loop
runs from a single device up to full hybrid parallelism.

Public surface (see docs/hybridmesh_design.md §3.4 for the full policy):
the stable face is the two names below plus ``hpmesh.config`` and the CLI.
"""

from hpmesh.config import HybridMeshConfig

__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy import: trainer.trainer pulls in the full torch.distributed stack
    # (DTensor, pipelining, ...), which is heavier than ``import hpmesh``
    # should pay for and unavailable on older torch builds. Importing Trainer
    # only on first access keeps the package import light.
    if name == "Trainer":
        from .trainer.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["HybridMeshConfig", "Trainer", "__version__"]
