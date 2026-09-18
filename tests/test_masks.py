"""Flex-attention masks and packed-document metadata.

The vendored helpers were verified identical to torchtitan's originals at
migration time (all five mask modifiers, both ``create_varlen_metadata``
branches, the padding branch, and ``round_up`` compared with ``torch.equal``).
That check needs torchtitan on the path, so what these tests pin is the
semantics each helper must keep on its own.

The masks all share one convention -- ``positions`` resets to 0 at each packed
document's first token -- and a mask that read a different boundary would still
produce a plausible-looking BlockMask while letting tokens attend across
documents. So the tests below evaluate the modifiers on an explicit index grid
rather than trusting a shape.
"""

from __future__ import annotations

import pytest
import torch

from hpmesh.models.common.masks import (
    VarlenMetadata,
    create_varlen_metadata_for_document,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_fixed_block_mask_mod,
    get_sliding_window_mask_mod,
    round_up,
)

# Three packed documents, lengths 4/3/4.
POSITIONS = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3])


def _grid(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(q_idx, kv_idx)`` pair a mask modifier is called with."""
    idx = torch.arange(seq_len)
    return torch.meshgrid(idx, idx, indexing="ij")


def _eval(mod, seq_len: int = POSITIONS.numel()) -> torch.Tensor:
    q_idx, kv_idx = _grid(seq_len)
    return mod(0, 0, q_idx, kv_idx)


# -- individual masks --------------------------------------------------------


def test_causal_mask_allows_past_and_self_only() -> None:
    mask = _eval(get_causal_mask_mod())
    for q in range(POSITIONS.numel()):
        for kv in range(POSITIONS.numel()):
            assert bool(mask[q, kv]) == (q >= kv)


def test_document_mask_confines_attention_to_one_document() -> None:
    mask = _eval(get_document_mask_mod(POSITIONS))
    # Doc 0 is [0,4). Cross-document pairs must be blocked in both directions.
    assert mask[0, 3]
    assert not mask[0, 4]  # doc 0 -> doc 1
    assert not mask[4, 3]  # doc 1 -> doc 0
    assert mask[4, 6]
    assert not mask[7, 6]  # doc 2 -> doc 1


def test_efficient_packed_mask_plus_causal_equals_document_mask() -> None:
    """The fast path is only correct *composed* with a causal mask.

    Its docstring is explicit that it supplies the lower bound and the causal
    mask the upper one, and that it is therefore not a stand-alone document
    mask. Composed, it must reproduce causal-and-same-document exactly --
    otherwise the two packs would attend differently.
    """
    q_idx, kv_idx = _grid(POSITIONS.numel())
    lower = get_efficient_causal_mask_mod_for_packed_document(POSITIONS)(
        0, 0, q_idx, kv_idx
    )
    causal = get_causal_mask_mod()(0, 0, q_idx, kv_idx)
    composed = lower & causal
    # The generic document mask is not causal on its own, so it needs the
    # causal half too before the comparison is meaningful.
    expected = get_document_mask_mod(POSITIONS)(0, 0, q_idx, kv_idx) & causal
    assert torch.equal(composed, expected)


def test_efficient_packed_mask_alone_leans_on_the_causal_bound() -> None:
    """Without the causal half it is *not* a document mask -- the docs say so."""
    lower = _eval(get_efficient_causal_mask_mod_for_packed_document(POSITIONS))
    expected = _eval(get_document_mask_mod(POSITIONS))
    assert not torch.equal(lower, expected)


def test_fixed_block_mask_allows_only_same_block() -> None:
    mask = _eval(get_fixed_block_mask_mod(4), seq_len=12)
    for q in range(12):
        for kv in range(12):
            assert bool(mask[q, kv]) == (q // 4 == kv // 4)


def test_fixed_block_mask_name_records_the_block_size() -> None:
    """The name lands in the compiled kernel's cache key, so it must vary."""
    assert get_fixed_block_mask_mod(4).__name__ == "blocked_mask_mod_fixed_block_size_4"
    assert get_fixed_block_mask_mod(8).__name__ == "blocked_mask_mod_fixed_block_size_8"


def test_sliding_window_mask_is_causal_and_bounded() -> None:
    mask = _eval(get_sliding_window_mask_mod(3), seq_len=6)
    for q in range(6):
        for kv in range(6):
            expected = kv <= q and (q - kv) < 3
            assert bool(mask[q, kv]) == expected


def test_sliding_window_of_one_attends_to_self_only() -> None:
    mask = _eval(get_sliding_window_mask_mod(1), seq_len=5)
    assert torch.equal(mask, torch.eye(5, dtype=torch.bool))


def test_sliding_window_rejects_a_zero_window() -> None:
    """A window of 0 masks the diagonal too, leaving all-zero attention rows."""
    with pytest.raises(ValueError):
        get_sliding_window_mask_mod(0)


# -- varlen metadata ---------------------------------------------------------


def test_varlen_metadata_lists_every_document_start() -> None:
    meta = create_varlen_metadata_for_document(POSITIONS)
    assert isinstance(meta, VarlenMetadata)
    # Starts at 0, 4, 7 plus the terminal token count.
    assert torch.equal(meta.cu_seq_q, torch.tensor([0, 4, 7, 11], dtype=torch.int32))
    assert meta.max_q == 4
    assert meta.max_k == 4


def test_varlen_metadata_uses_position_resets_as_boundaries() -> None:
    """Contiguous tokens with no reset are one document."""
    meta = create_varlen_metadata_for_document(torch.arange(6))
    assert torch.equal(meta.cu_seq_q, torch.tensor([0, 6], dtype=torch.int32))
    assert meta.max_q == 6


def test_varlen_metadata_handles_a_single_token() -> None:
    meta = create_varlen_metadata_for_document(torch.zeros(1, dtype=torch.long))
    assert torch.equal(meta.cu_seq_q, torch.tensor([0, 1], dtype=torch.int32))
    assert meta.max_q == 1


def test_fixed_shape_metadata_reserves_a_trailing_slot() -> None:
    """``max_num_documents`` real slots, one sentinel, one terminal count.

    The sentinel is what keeps a document whose start was not recorded from
    leaking into its neighbour; the terminal entry is the token count the
    kernel reads as the end of the last segment.
    """
    meta = create_varlen_metadata_for_document(
        POSITIONS, max_num_documents=4, max_context_length=8
    )
    # 4 document slots + 1 sentinel slot, with the terminal count landing in
    # the last slot.
    assert meta.cu_seq_q.numel() == 5
    assert torch.equal(
        meta.cu_seq_q, torch.tensor([0, 4, 7, 11, 11], dtype=torch.int32)
    )
    assert meta.max_q == 8


def test_fixed_shape_slots_grow_with_the_document_bound() -> None:
    """More documents allowed means more reserved slots, not a compaction."""
    small = create_varlen_metadata_for_document(
        POSITIONS, max_num_documents=3, max_context_length=8
    )
    large = create_varlen_metadata_for_document(
        POSITIONS, max_num_documents=8, max_context_length=8
    )
    assert small.cu_seq_q.numel() == 4
    assert large.cu_seq_q.numel() == 9
    # Both describe the same three real documents.
    assert torch.equal(small.cu_seq_q[:3], large.cu_seq_q[:3])


def test_fixed_shape_metadata_requires_a_context_length() -> None:
    with pytest.raises(ValueError):
        create_varlen_metadata_for_document(POSITIONS, max_num_documents=3)


def test_padding_starts_do_not_consume_document_capacity() -> None:
    """Padding resets must be reserved separately, not spent on real docs."""
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3])
    padding = torch.zeros_like(positions, dtype=torch.bool)
    padding[4:7] = True  # the middle "document" is really padding

    # Only docs 0 and 2 are real, and the 3-token padding run fits one segment.
    meta = create_varlen_metadata_for_document(
        positions, padding_mask=padding, max_num_documents=2, max_context_length=8
    )
    assert meta.cu_seq_q.numel() == 2 + 1 + 1 + 1


def test_exceeding_the_reserved_document_capacity_is_caught() -> None:
    """A deferred assert fires rather than silently writing past the slots."""
    with pytest.raises(RuntimeError):
        create_varlen_metadata_for_document(
            POSITIONS, max_num_documents=1, max_context_length=8
        )


def test_exceeding_the_reserved_padding_capacity_is_caught() -> None:
    """Two padding segments are needed; reserving one must be caught."""
    positions = torch.arange(12, dtype=torch.long) % 4
    padding = torch.zeros(12, dtype=torch.bool)
    padding[:9] = True  # a 9-token run with max_context_length=8 spans two

    with pytest.raises(RuntimeError):
        create_varlen_metadata_for_document(
            positions, padding_mask=padding, max_num_documents=1, max_context_length=8
        )


# -- round_up ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "multiple", "expected"),
    [(0, 128, 0), (1, 128, 128), (128, 128, 128), (129, 128, 256), (100, 1, 100)],
)
def test_round_up(value: int, multiple: int, expected: int) -> None:
    assert round_up(value, multiple) == expected
