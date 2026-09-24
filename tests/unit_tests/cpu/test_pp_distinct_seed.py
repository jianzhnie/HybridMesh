"""PP per-stage seed derivation (hpmesh.utils.seed).

trainer.py is not importable in a minimal CPU environment, so the derivation
lives in a dependency-free helper; these tests pin the seed semantics the
trainer relies on: distinct RNG streams per PP stage, reproducible per stage,
and bit-identical behavior when pp == 1.
"""

import torch

from hpmesh.utils.seed import derive_distinct_seed


def _rng_snapshot(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.get_rng_state()


class TestDeriveDistinctSeed:
    def test_distinct_pp_ranks_get_distinct_rng_states(self):
        base_seed = 1234
        pp_size = 4
        snapshots = {
            _rng_snapshot(derive_distinct_seed(base_seed, [(pp_rank, pp_size)]))
            for pp_rank in range(pp_size)
        }
        # RNG states are byte tensors; distinct contents => distinct streams.
        unique = {bytes(s) for s in snapshots}
        assert len(unique) == pp_size

    def test_same_pp_rank_is_reproducible(self):
        base_seed = 42
        derived = derive_distinct_seed(base_seed, [(2, 4)])
        assert derived == derive_distinct_seed(base_seed, [(2, 4)])
        torch.manual_seed(derived)
        first = torch.rand(8)
        torch.manual_seed(derive_distinct_seed(base_seed, [(2, 4)]))
        assert torch.equal(first, torch.rand(8))

    def test_pp1_leaves_seed_unchanged(self):
        base_seed = 2026
        # A size-1 pp mesh contributes local_rank 0: identity derivation.
        assert derive_distinct_seed(base_seed, [(0, 1)]) == base_seed
        # And with no distinct dims at all there is nothing to offset.
        assert derive_distinct_seed(base_seed, []) == base_seed

    def test_offset_matches_row_major_indexing(self):
        # Multi-dim generality: (r0, s0), (r1, s1) -> r0 + r1 * s0.
        assert derive_distinct_seed(100, [(2, 3), (1, 5)]) == 100 + 2 + 1 * 3

    def test_result_stays_in_torch_seed_range(self):
        base_seed = 2**64 - 2
        derived = derive_distinct_seed(base_seed, [(3, 4)])
        assert derived == (base_seed + 3) % 2**64
        torch.manual_seed(derived)  # must be accepted by torch
