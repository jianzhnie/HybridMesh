"""Context parallelism: the CP flex kernel, input sharding, and wiring.

``cp_kernel.py`` holds the attention redistribution itself -- both strategies
(KV all-gather and Ulysses all-to-all) live in ``CPFlexKernel``, which is what
actually runs inside HF attention. ``input_shard.py`` holds the batch and mask
sharding; ``apply.py`` wires the kernel onto a model.
"""

from .apply import apply_cp
from .cp_kernel import CPFlexKernel
from .input_shard import (
    shard_attention_mask_for_cp,
    shard_batch_for_cp,
    shard_batch_for_tp,
)

__all__ = [
    "CPFlexKernel",
    "apply_cp",
    "shard_attention_mask_for_cp",
    "shard_batch_for_cp",
    "shard_batch_for_tp",
]
