"""Expert-parallel wiring: the single-rank guards.

The swap and the multi-rank all-to-all live in ``tests/test_ep_swap.py`` and
``tests/ep_wiring_equivalence.py``. What can be checked without a mesh is the
refusal path: EP is wired, so without the sparse mesh's EP group it must refuse
loudly rather than silently swap to a replicated (local) dispatcher -- and with
EP off it must leave the model untouched.
"""

from __future__ import annotations

from tests.caps import require_env

require_env('spmd_types')


import pytest
import torch

from hpmesh.parallel.expert_parallel import apply_ep
from hpmesh.trainer import ParallelConfig


def test_apply_ep_is_a_no_op_when_ep_is_off() -> None:
    """EP=1 must hand the model back untouched rather than raise."""
    model = torch.nn.Linear(4, 4)
    cfg = ParallelConfig(expert_parallel_size=1)

    assert apply_ep(model, cfg) is model


def test_apply_ep_requires_an_ep_group_when_ep_is_on() -> None:
    """Without the sparse mesh's EP group it must refuse loudly rather than
    silently swap to a replicated (local) dispatcher."""
    cfg = ParallelConfig(expert_parallel_size=2)

    with pytest.raises(ValueError, match="EP process group"):
        apply_ep(torch.nn.Linear(4, 4), cfg)
