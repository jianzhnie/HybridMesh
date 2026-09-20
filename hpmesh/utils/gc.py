"""A GC helper for the training loop and the checkpointer.

Vendored from torchtitan's ``tools/utils.py`` (the ``GarbageCollection`` class
only). It exists because CPython's cyclic collector is costly to run over a
process holding a multi-gigabyte object graph, and the default schedule fires at
unpredictable times -- often in the middle of a forward. torchtitan disables it
outright and collects at points the training loop chooses instead.

``collect`` is the piece the checkpointer needs: a save allocates a lot of
transient host memory (staging buffers, flattened state dicts), and dropping the
previous step's garbage before allocating more measurably reduces peak RSS.
"""

from __future__ import annotations

import gc
import time

from .logger_utils import get_logger

# ``get_logger``, not a bare ``logging.getLogger``: without the handler this
# module's installer attaches, the collection notices are emitted and then
# dropped, which is indistinguishable from the collector never running.
logger = get_logger(__name__)


class GarbageCollection:
    """Periodic, manually scheduled cyclic collection.

    Args:
        gc_freq: collect every this many steps. 0 disables periodic collection
            while leaving the collector off.
        debug: collect every step and warn about tensor reference cycles.
    """

    def __init__(self, gc_freq: int = 1000, debug: bool = False) -> None:
        if gc_freq <= 0:
            raise ValueError(f"gc_freq must be a positive integer, got {gc_freq}")
        self.gc_freq = gc_freq
        self.debug = debug
        gc.disable()
        self.collect("Initial GC collection")
        if debug:
            import torch.distributed as dist
            from torch.utils.viz._cycles import warn_tensor_cycles

            if not dist.is_initialized() or dist.get_rank() == 0:
                warn_tensor_cycles()

    def run(self, step_count: int) -> bool:
        """Collect if this step should. Returns whether a collection ran."""
        if self.debug:
            self.collect(
                "Force GC to perform collection to obtain debug information",
                generation=2,
            )
            return True
        if step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Performing periodic GC collection")
            return True
        return False

    @staticmethod
    def collect(reason: str, generation: int = 1) -> None:
        begin = time.monotonic()
        gc.collect(generation)
        logger.info("[GC] %s took %.2f seconds", reason, time.monotonic() - begin)
