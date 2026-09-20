"""Context parallelism: primitives, the CP flex kernel, and input sharding.

``primitives.py`` holds the attention redistribution primitives (KV all-gather,
Ulysses all-to-all); ``cp_kernel.py`` the flex kernel that drives them inside
HF attention; ``input_shard.py`` the batch/mask sharding; ``apply.py`` wires
the kernel onto a model. Together they make a CP step runnable end to end.
"""

from .apply import apply_cp
from .cp_kernel import CPFlexKernel
from .input_shard import shard_attention_mask_for_cp, shard_batch_for_cp
from .primitives import (
    HEAD_DIM,
    TOKEN_DIM,
    KVAllGatherContextParallel,
    UlyssesContextParallel,
    cp_group,
    cp_redistribute,
)

__all__ = [
    "HEAD_DIM",
    "TOKEN_DIM",
    "CPFlexKernel",
    "KVAllGatherContextParallel",
    "UlyssesContextParallel",
    "apply_cp",
    "cp_group",
    "cp_redistribute",
    "shard_attention_mask_for_cp",
    "shard_batch_for_cp",
]
