"""Wire EP onto a model: swap every HF MoE block for the hpmesh MoE."""

from __future__ import annotations

import logging

import torch.distributed as dist
import torch.nn as nn

from hpmesh.trainer.config import ParallelConfig

from .ep import swap_hf_moe_blocks

logger = logging.getLogger(__name__)

__all__ = ["apply_ep"]


def apply_ep(
    model: nn.Module,
    cfg: ParallelConfig,
    *,
    ep_group: dist.ProcessGroup | None = None,
) -> nn.Module:
    """Swap every HF MoE block for the EP-capable hpmesh MoE.

    No-op when ``cfg.ep == 1`` (the model is handed back untouched), so the
    caller needs no degree check of its own.

    The swap moves weights, so it must happen before FSDP wraps the model.
    With a multi-rank group each rank keeps ``num_experts / ep`` of them and
    tokens cross ranks by all-to-all; the two are orthogonal to the CP
    attention kernel, so CP and EP compose freely.
    """
    if cfg.ep == 1:
        return model

    if ep_group is None or ep_group.size() != cfg.ep:
        raise ValueError(
            f"ep={cfg.ep} requires an EP process group of that size, got "
            f"{None if ep_group is None else ep_group.size()}. The group comes "
            "from the sparse mesh's 'ep' axis (parallelize_hf_transformers "
            "resolves it from parallel_dims)."
        )
    swapped = swap_hf_moe_blocks(model, ep_group=ep_group)
    logger.info(
        "Applied EP (all-to-all dispatch): swapped %d MoE blocks, degree %d",
        swapped,
        cfg.ep,
    )
    return model
