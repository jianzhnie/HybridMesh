"""Context parallelism: input/mask sharding and the CP flex kernel.

The attention redistribution primitives themselves (KV all-gather, Ulysses
all-to-all) live one level up in ``parallel/cp_ep.py``; this subpackage holds
the wiring that makes a CP step runnable end to end.
"""

from .cp_kernel import CPFlexKernel
from .input_shard import shard_attention_mask_for_cp, shard_batch_for_cp

__all__ = [
    "CPFlexKernel",
    "shard_attention_mask_for_cp",
    "shard_batch_for_cp",
]
