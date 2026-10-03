"""Expert parallelism: swap HF MoE blocks for the EP-capable llmtuner MoE.

EP core idea (MoE): shard experts across ranks; an all-to-all routes each token
to its expert's rank and back. ``swap.py`` is the weight-moving swap itself;
``apply.py`` wires it onto a model given the EP process group.

The two entry points are re-exported lazily (PEP 562): ``swap.py`` pulls in the
whole MoE stack (``models/common/moe`` and its custom ops), and lighter
submodules like ``ckpt.py`` -- imported by the checkpointer and optimizer
containers -- must stay loadable on hosts whose torch predates those ops.
"""

__all__ = [
    "apply_ep",
    "swap_hf_moe_blocks",
]


def __getattr__(name: str):
    if name == "apply_ep":
        from .apply import apply_ep

        return apply_ep
    if name == "swap_hf_moe_blocks":
        from .swap import swap_hf_moe_blocks

        return swap_hf_moe_blocks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
