"""llmtuner -- Hybrid Parallel training over a Torch DeviceMesh.

A minimal, learning-oriented framework for understanding the core modules of
distributed training (FSDP / TP / PP / CP / EP) by building them from scratch.

Two abstractions only:
  * a grouped ``LLMTunerConfig``      (llmtuner.config)
  * a ``HFTransformerModel``            (llmtuner.models.hf.model)

Parallelism dimensions are added one at a time; each ``apply_*`` in
``llmtuner.parallel`` is a no-op when its degree is 1, so the same training loop
runs from a single device up to full hybrid parallelism.

Public surface (see docs/torchllmtuner_design.md §3.4 for the full policy):
the stable face is the two names below plus ``llmtuner.config`` and the CLI.
"""

from llmtuner.config import LLMTunerConfig
from llmtuner.utils.lazy_exports import resolve_export

__version__ = "0.1.0"

#: Name -> submodule, for the one name this package keeps lazy. ``Trainer`` is
#: the expensive half: ``trainer.trainer`` pulls in the whole
#: ``torch.distributed`` stack (DTensor, pipelining), which ``import llmtuner``
#: must not pay for and older torch builds do not have.
_EXPORT_SOURCES = {
    "Trainer": "trainer.trainer",
}


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = ["LLMTunerConfig", "Trainer", "__version__"]
