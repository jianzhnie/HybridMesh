"""Vision-to-text fusion: span-based and gather-based.

The vendored helpers were verified identical to torchtitan's originals at
migration time (all five functions plus both ValueError paths, compared with
``torch.equal``). What these tests pin is the part that would corrupt a model
silently: the *alignment* between the text placeholders and the vision features.

Both strategies copy item N's features into the Nth span of placeholders. If the
two streams disagree about how many tokens an item produced, the copy still
succeeds -- it just writes the wrong features into the wrong positions, and
attention trains happily on them. So every misalignment below is asserted to
raise rather than to produce a wrong tensor.
"""

from __future__ import annotations

import pytest
import torch

from hpmesh.models.common.multimodal import (
    build_vision_bank_indices,
    gather_vision_embeds,
    get_vision_positions,
    scatter_vision_embeds,
)

PLACEHOLDER = 99
DIM = 6


def _tokens() -> torch.Tensor:
    """Two vision spans: 3 placeholders at 2:5, then 2 at 7:9."""
    return torch.tensor(
        [1, 2, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, 3, 4, PLACEHOLDER, PLACEHOLDER, 5]
    )


def _embeds(seed: int = 0) -> torch.Tensor:
    return torch.randn(
        _tokens().numel(), DIM, generator=torch.Generator().manual_seed(seed)
    )


def _bank(rows: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(rows, DIM, generator=torch.Generator().manual_seed(seed))


# -- span locations ----------------------------------------------------------


def test_positions_report_each_run() -> None:
    positions = get_vision_positions(_tokens(), torch.tensor([3, 2]), PLACEHOLDER)
    assert positions == [(0, 2, 3), (1, 7, 2)]


def test_positions_require_one_run_per_item() -> None:
    """More items than runs would leave an item's features unwritten."""
    with pytest.raises(ValueError, match="contiguous run"):
        get_vision_positions(_tokens(), torch.tensor([3, 2, 1]), PLACEHOLDER)


def test_positions_require_a_run_to_match_its_item_length() -> None:
    with pytest.raises(ValueError, match="spans"):
        get_vision_positions(_tokens(), torch.tensor([3, 9]), PLACEHOLDER)


def test_positions_accept_a_single_run_touching_the_sequence_start() -> None:
    tokens = torch.tensor([PLACEHOLDER, PLACEHOLDER, 7])
    assert get_vision_positions(tokens, torch.tensor([2]), PLACEHOLDER) == [(0, 0, 2)]


def test_positions_accept_a_run_reaching_the_sequence_end() -> None:
    """The end-of-run mask is shifted from the mask itself, so the last index
    must still close a run rather than fall off."""
    tokens = torch.tensor([7, PLACEHOLDER, PLACEHOLDER])
    assert get_vision_positions(tokens, torch.tensor([2]), PLACEHOLDER) == [(0, 1, 2)]


def test_positions_with_no_placeholders_need_no_items() -> None:
    assert (
        get_vision_positions(torch.tensor([1, 2, 3]), torch.tensor([]), PLACEHOLDER)
        == []
    )


# -- bank indices ------------------------------------------------------------


def test_bank_indices_number_placeholders_in_order() -> None:
    indices = build_vision_bank_indices(_tokens(), placeholder_id=PLACEHOLDER)
    assert indices.tolist() == [-1, -1, 0, 1, 2, -1, -1, 3, 4, -1]


def test_bank_indices_are_negative_one_off_placeholder() -> None:
    """The sentinel is what ``gather_vision_embeds`` masks on."""
    indices = build_vision_bank_indices(
        torch.tensor([1, 2]), placeholder_id=PLACEHOLDER
    )
    assert (indices == -1).all()


def test_bank_indices_are_non_negative_exactly_on_placeholders() -> None:
    tokens = _tokens()
    indices = build_vision_bank_indices(tokens, placeholder_id=PLACEHOLDER)
    assert torch.equal(indices >= 0, tokens == PLACEHOLDER)


# -- gather ------------------------------------------------------------------


def test_gather_writes_each_bank_row_into_its_token() -> None:
    tokens = _tokens()
    indices = build_vision_bank_indices(tokens, placeholder_id=PLACEHOLDER)
    bank = _bank(5)
    inputs = _embeds()

    out = gather_vision_embeds(
        inputs, vision_bank_VD=bank, vision_bank_indices_T=indices
    )

    for token in range(tokens.numel()):
        row = indices[token].item()
        expected = bank[row] if row >= 0 else inputs[token]
        assert torch.equal(out[token], expected)


def test_gather_leaves_text_positions_untouched() -> None:
    tokens = _tokens()
    indices = build_vision_bank_indices(tokens, placeholder_id=PLACEHOLDER)
    inputs = _embeds()

    out = gather_vision_embeds(
        inputs, vision_bank_VD=_bank(5), vision_bank_indices_T=indices
    )

    text = tokens != PLACEHOLDER
    assert torch.equal(out[text], inputs[text])


def test_gather_with_an_empty_bank_returns_the_input_unchanged() -> None:
    """A pure-text batch must pass through, not error on an empty gather."""
    indices = build_vision_bank_indices(_tokens(), placeholder_id=PLACEHOLDER)
    inputs = _embeds()

    out = gather_vision_embeds(
        inputs, vision_bank_VD=_bank(0), vision_bank_indices_T=indices
    )

    assert out is inputs


def test_gather_casts_the_bank_to_the_input_dtype() -> None:
    tokens = _tokens()
    indices = build_vision_bank_indices(tokens, placeholder_id=PLACEHOLDER)
    inputs = _embeds().bfloat16()

    out = gather_vision_embeds(
        inputs, vision_bank_VD=_bank(5).double(), vision_bank_indices_T=indices
    )

    assert out.dtype is torch.bfloat16


def test_gather_does_not_mutate_the_input() -> None:
    indices = build_vision_bank_indices(_tokens(), placeholder_id=PLACEHOLDER)
    inputs = _embeds()
    original = inputs.clone()

    gather_vision_embeds(inputs, vision_bank_VD=_bank(5), vision_bank_indices_T=indices)

    assert torch.equal(inputs, original)


# -- scatter -----------------------------------------------------------------


def test_scatter_copies_span_by_span() -> None:
    positions = get_vision_positions(_tokens(), torch.tensor([3, 2]), PLACEHOLDER)
    bank = _bank(5)
    inputs = _embeds()

    out = scatter_vision_embeds(
        inputs.clone(), vision_embeds=bank, vision_positions=positions
    )

    assert torch.equal(out[2:5], bank[0:3])
    assert torch.equal(out[7:9], bank[3:5])


def test_scatter_leaves_non_span_tokens_alone() -> None:
    positions = get_vision_positions(_tokens(), torch.tensor([3, 2]), PLACEHOLDER)
    inputs = _embeds()
    original = inputs.clone()

    out = scatter_vision_embeds(
        inputs.clone(), vision_embeds=_bank(5), vision_positions=positions
    )

    mask = torch.ones(inputs.shape[0], dtype=torch.bool)
    mask[2:5] = mask[7:9] = False
    assert torch.equal(out[mask], original[mask])


def test_scatter_rejects_an_unconsumed_remaining_bank() -> None:
    """Extra vision features mean an item never found its placeholders."""
    positions = get_vision_positions(_tokens(), torch.tensor([3, 2]), PLACEHOLDER)

    with pytest.raises(ValueError, match="consume"):
        scatter_vision_embeds(
            _embeds(), vision_embeds=_bank(7), vision_positions=positions
        )


def test_scatter_rejects_a_bank_too_small_for_the_spans() -> None:
    positions = get_vision_positions(_tokens(), torch.tensor([3, 2]), PLACEHOLDER)

    with pytest.raises(ValueError, match="consume"):
        scatter_vision_embeds(
            _embeds(), vision_embeds=_bank(4), vision_positions=positions
        )


def test_scatter_casts_to_the_input_dtype() -> None:
    positions = get_vision_positions(_tokens(), torch.tensor([3, 2]), PLACEHOLDER)
    out = scatter_vision_embeds(
        _embeds().bfloat16(),
        vision_embeds=_bank(5).double(),
        vision_positions=positions,
    )
    assert out.dtype is torch.bfloat16


# -- the two strategies agree ------------------------------------------------


def test_gather_and_scatter_produce_the_same_fusion() -> None:
    """Both paths must land the same features in the same positions.

    The gather path carries absolute bank rows while the scatter path walks
    runs, so agreement is not automatic -- it holds because the bank rows are
    numbered in placeholder order, which is what the positions list assumes too.
    """
    tokens = _tokens()
    bank = _bank(5)
    per_item = torch.tensor([3, 2])

    scattered = scatter_vision_embeds(
        _embeds().clone(),
        vision_embeds=bank,
        vision_positions=get_vision_positions(tokens, per_item, PLACEHOLDER),
    )
    gathered = gather_vision_embeds(
        _embeds(),
        vision_bank_VD=bank,
        vision_bank_indices_T=build_vision_bank_indices(
            tokens, placeholder_id=PLACEHOLDER
        ),
    )

    assert torch.equal(scattered, gathered)
