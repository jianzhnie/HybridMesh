"""Expert parallelism: swap HF MoE blocks for the EP-capable hpmesh MoE.

EP core idea (MoE): shard experts across ranks; an all-to-all routes each token
to its expert's rank and back. ``ep.py`` is the weight-moving swap itself;
``apply.py`` wires it onto a model given the EP process group.

``swap_hf_moe_blocks`` is re-exported lazily rather than imported at module
scope. Importing ``ep`` here would pull ``models/common/moe`` into
``hpmesh.parallel``'s own import, and ``models/common`` imports
``parallel/spmd_types`` -- so any import order starting inside ``hpmesh.models``
would see a half-initialized ``models.common.linear``. ``apply_ep`` does not
touch the MoE stack until it is called with ``ep > 1``, and neither does
``parallelize_hf_transformers`` unless EP is configured, so that is where the
cost belongs. ``apply_ep`` itself is cheap (it imports only the config), so it
stays eager.
"""

from typing import TYPE_CHECKING, Any

from .apply import apply_ep

if TYPE_CHECKING:
    from .ep import swap_hf_moe_blocks

__all__ = [
    "apply_ep",
    "swap_hf_moe_blocks",
]


def __getattr__(name: str) -> Any:
    """Resolve ``swap_hf_moe_blocks`` on first access (PEP 562)."""
    if name == "swap_hf_moe_blocks":
        from .ep import swap_hf_moe_blocks

        return swap_hf_moe_blocks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
