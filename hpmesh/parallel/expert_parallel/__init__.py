"""Expert parallelism: swap HF MoE blocks for the EP-capable hpmesh MoE.

EP core idea (MoE): shard experts across ranks; an all-to-all routes each token
to its expert's rank and back. ``swap.py`` is the weight-moving swap itself;
``apply.py`` wires it onto a model given the EP process group.

Both are re-exported eagerly. ``swap.py`` imports ``models/common/moe`` (the
stack it swaps in), which used to import ``parallel/spmd_types`` back -- the
cycle that once forced a lazy re-export here. The SPMD mesh context now lives
in ``hpmesh/accelerator/spmd_context.py`` below both layers, so nothing under
``models/common`` imports ``hpmesh.parallel`` and the cycle is gone.
"""

from .apply import apply_ep
from .swap import swap_hf_moe_blocks

__all__ = [
    "apply_ep",
    "swap_hf_moe_blocks",
]
