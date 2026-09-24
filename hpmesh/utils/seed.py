"""Mesh-aware seed derivation.

Pipeline stages must not share an RNG stream: stages hold *different* layers,
so seeding every stage identically would correlate their weight initialization
and dropout patterns. Ranks inside one SPMD group, by contrast, must draw the
same numbers so sharded parameters stay consistent. The derivation therefore
offsets the base seed only along mesh dimensions declared distinct (``pp``),
keeping all other coordinates' ranks on the base seed.
"""

from collections.abc import Iterable


def derive_distinct_seed(seed: int, distinct_coords: Iterable[tuple[int, int]]) -> int:
    """Offset ``seed`` by this rank's coordinates along distinct mesh dims.

    Each ``(local_rank, dim_size)`` pair contributes ``local_rank`` times the
    product of all previous dimensions' sizes -- row-major indexing over the
    distinct sub-mesh, so every coordinate tuple maps to a unique offset. The
    result is reduced mod 2**64 to stay inside ``torch.manual_seed``'s range.

    An empty ``distinct_coords`` (or all-zero local ranks, e.g. a size-1 dim)
    returns ``seed`` unchanged, so single-stage runs are bit-identical to
    seeding without derivation.
    """
    offset = 0
    cumulative_size = 1
    for local_rank, dim_size in distinct_coords:
        offset += local_rank * cumulative_size
        cumulative_size *= dim_size
    return (seed + offset) % 2**64
