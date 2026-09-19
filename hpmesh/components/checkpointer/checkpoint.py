"""Checkpointing: save training state so a run can be resumed.

What is kept from torchtitan: the trainer is *stateful* -- it exposes exactly
what a resume needs (a step counter, a token counter, and the model/optimizer),
and ``train()`` loads before the loop and saves inside it.

What is NOT kept, and why: torchtitan's checkpointer is a component with a
config, asynchronized staging, and ``torch.distributed.checkpoint`` underneath.
DCP is what makes a *sharded* checkpoint cheap -- each rank writes only its own
shard and the layout is reconstructed on load. hpmesh's parameters are plain
tensors, so there is nothing to shard: each rank writes its own file and reads
it back. The moment FSDP + TP are composed on one mesh (see ``fsdp_wrap``), the
DTensor path becomes relevant and DCP is the right answer -- this file is where
that swap happens, and it is deliberately the only file that would change.

The file format is intentionally the boring one (``torch.save`` of plain dicts,
one per rank) so the checkpoint is inspectable with a REPL.
"""

from __future__ import annotations

import os
from typing import Any

import torch

__all__ = ["Checkpointer"]


class Checkpointer:
    """Per-rank checkpoint save/load for one training run.

    Every rank writes its own file. That is not an optimization, it is the
    correct model for the tensor layouts hpmesh can produce today: parameters are
    plain local tensors (possibly FSDP-sharded, in which case each rank's file
    holds that rank's shard), so there is no cross-rank structure to preserve and
    no collective to coordinate. Loading restores the same rank's shard.

    Args:
        folder: directory for the checkpoints. Created on first save.
        rank: this process's global rank, used to name its file.
        device: where loaded tensors are placed.
    """

    def __init__(self, folder: str, *, rank: int, device: torch.device) -> None:
        self.folder = folder
        self.rank = rank
        self.device = device

    @property
    def path(self) -> str:
        return os.path.join(self.folder, f"rank{self.rank:02d}.pt")

    def save(
        self,
        step: int,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        counters: dict[str, Any],
    ) -> str:
        """Write this rank's training state. Returns the path written.

        Written to a temporary file and renamed into place: a crash mid-write
        then leaves the previous checkpoint intact rather than a truncated file
        that fails to load and takes the whole run with it.
        """
        os.makedirs(self.folder, exist_ok=True)
        payload = {
            "step": step,
            "counters": counters,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        tmp = f"{self.path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.path)
        return self.path

    def load(
        self, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> dict[str, Any] | None:
        """Restore this rank's training state, or ``None`` when there is none.

        ``None`` is the normal first-run answer, not an error, so the caller can
        treat "fresh run" and "resumed run" as the same code path.
        """
        if not os.path.exists(self.path):
            return None

        payload = torch.load(self.path, map_location=self.device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        return {"step": payload["step"], **payload["counters"]}
