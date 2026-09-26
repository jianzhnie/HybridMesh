"""The assembly stage list: the order contract, as data.

``parallelize_hf_transformers``'s correctness is its call order, and this
table is the single source of it: the unsplit path runs ``STAGE_ORDER``, the
PP per-part path runs the ``on_pp`` subsequence, so the two paths cannot
drift into different relative orders. Kept engine-free (a dataclass and
tuples, no ``apply_*`` imports) so the contract is importable anywhere --
``hpmesh/config``, tests, docs tooling -- without paying for the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PP_STAGE_ORDER",
    "STAGE_ORDER",
    "STAGES",
    "Stage",
    "stage_enabled",
]

# The order IS the contract (see parallelize.py's docstring).
@dataclass(frozen=True)
class Stage:
    """One assembly stage: name, whether PP runs it, why it sits here."""

    name: str
    on_pp: bool
    why_here: str


STAGES: tuple[Stage, ...] = (
    Stage(
        "tp",
        True,
        "Sharding wrappers first: TP cuts the projections before anything "
        "wraps the layers.",
    ),
    Stage(
        "ep",
        False,
        "After TP: the swap consumes TP's dense sharding and owns the routed "
        "experts. Not on the PP path -- pp x ep is refused (matrix.pp_cp_ep).",
    ),
    Stage(
        "cp",
        False,
        "With the other sharding wrappers. Not on the PP path -- pp x cp is "
        "refused (matrix.pp_cp_ep).",
    ),
    Stage(
        "ac",
        False,
        "After the sharding wrappers (it must enclose the TP/CP-modified "
        "layer), before compile and FSDP -- torchtitan's order in "
        "``parallelize_llama``. Not on the PP path "
        "(matrix.pp_activation_checkpoint).",
    ),
    Stage(
        "compile",
        True,
        "After AC, before FSDP; runs only when the caller passes "
        "compile=True.",
    ),
    Stage(
        "fsdp",
        True,
        "Last, so FSDP's hooks sit outermost.",
    ),
)

STAGE_ORDER: tuple[str, ...] = tuple(stage.name for stage in STAGES)
PP_STAGE_ORDER: tuple[str, ...] = tuple(stage.name for stage in STAGES if stage.on_pp)


def stage_enabled(name: str, *, compile: bool) -> bool:
    """Whether a stage runs this call: only ``compile`` is conditional."""
    return name != "compile" or compile
