"""Packing nodes: turn samples into fixed-length rows.

Three modules, split by what each part depends on:

* ``build`` -- the entry points (``build_concat_then_split_packing``,
  ``build_first_fit_packing``) and the row-length policy they share.
* ``iterators`` -- the Grain iterators that splice documents across row
  boundaries, plus the resumable state that makes a restart land mid-corpus.
* ``conversions`` -- the pure per-row conversions between a ``TextSequence``
  and the dict the packers emit.

The public names are re-exported here, so callers keep importing ``..packing``.
"""

from .build import build_concat_then_split_packing, build_first_fit_packing

__all__ = ["build_concat_then_split_packing", "build_first_fit_packing"]
