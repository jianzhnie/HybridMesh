"""Model-agnostic vision-to-text fusion for VLMs.

Vendored from torchtitan ``models/common/multimodal.py``. The only removal is a
pair of ``spmd_types`` calls in ``gather_vision_embeds`` (``spmd.local()`` and an
``is_type_checking()`` assertion) -- type-checker annotations with no runtime
effect, describing a placement the framework already inferred.

Two fusion strategies are supported, and they are not interchangeable:

* **Span-based** (``get_vision_positions`` + ``scatter_vision_embeds``) walks
  the sequence for contiguous placeholder runs and copies packed vision features
  into each run. It is positional: the Nth run gets the Nth item's features.
* **Gather-based** (``build_vision_bank_indices`` + ``gather_vision_embeds``)
  carries an absolute packed-bank row for every placeholder token, so it does
  not care where the placeholders sit or whether the runs are contiguous.

Shape legend, scoped to this file: ``T`` = tokens, ``D`` = model dimension,
``V`` = packed vision-bank rows.
"""

from __future__ import annotations

import torch

__all__ = [
    "build_vision_bank_indices",
    "gather_vision_embeds",
    "get_vision_positions",
    "scatter_vision_embeds",
]


def get_vision_positions(
    tokens: torch.Tensor,
    num_vision_tokens_per_item: torch.Tensor,
    placeholder_id: int,
) -> list[tuple[int, int, int]]:
    """Locate each visual item's placeholder run in the token sequence.

    Args:
        tokens: ``(T,)`` token IDs.
        num_vision_tokens_per_item: ``(num_items,)`` valid token count per
            visual item, in the order the items appear in ``tokens``.
        placeholder_id: token id whose contiguous runs mark vision spans.

    Returns:
        ``(item_idx, vision_start, n_tokens)`` per item.

    Raises:
        ValueError: if the number of placeholder runs does not equal the number
            of visual items, or a run's length does not match the item's token
            count. Either mismatch means the text and vision streams are
            misaligned; scattering anyway would silently corrupt the embeddings,
            so it fails loudly with the offending counts instead.
    """
    vision_mask = tokens == placeholder_id
    prev_mask = torch.zeros_like(vision_mask)
    prev_mask[1:] = vision_mask[:-1]
    next_mask = torch.zeros_like(vision_mask)
    next_mask[:-1] = vision_mask[1:]
    # A run starts where the mask turns on and ends where it turns off.
    region_starts = torch.where(vision_mask & ~prev_mask)[0]
    region_ends = torch.where(vision_mask & ~next_mask)[0]

    num_items = int(num_vision_tokens_per_item.shape[0])
    num_runs = int(region_starts.shape[0])
    if num_runs != num_items:
        raise ValueError(
            f"Multimodal misalignment: found {num_runs} contiguous run(s) of "
            f"placeholder id {placeholder_id} in the token sequence but received "
            f"{num_items} visual item(s). Each visual item must correspond to "
            f"exactly one placeholder run."
        )

    # Convert each metadata tensor once. Per-item ``.item()`` calls would
    # synchronize CUDA once per scalar.
    region_starts_list = region_starts.tolist()
    run_lengths = (region_ends - region_starts + 1).tolist()
    num_vision_tokens_per_item_list = num_vision_tokens_per_item.tolist()
    positions: list[tuple[int, int, int]] = []
    for i in range(num_items):
        start = int(region_starts_list[i])
        n_tokens = int(num_vision_tokens_per_item_list[i])
        if run_lengths[i] != n_tokens:
            raise ValueError(
                f"Multimodal misalignment: placeholder run {i} spans "
                f"{run_lengths[i]} token(s) but visual item {i} produced "
                f"{n_tokens} embedding(s). The placeholder count in the prompt "
                f"must match the vision token count for that item."
            )
        positions.append((i, start, n_tokens))
    return positions


def build_vision_bank_indices(
    tokens_T: torch.Tensor,
    *,
    placeholder_id: int,
) -> torch.Tensor:
    """Map vision placeholder tokens to absolute packed-bank rows.

    Returns a ``(T,)`` tensor holding each placeholder token's row in the packed
    vision bank, or ``-1`` for a non-placeholder token. The row is a running
    count of placeholders seen so far, so it matches the order the vision tower
    emitted them in -- ``gather_vision_embeds`` relies on that.
    """
    vision_mask_T = tokens_T == placeholder_id
    vision_bank_indices_T = torch.cumsum(vision_mask_T.to(torch.long), dim=0) - 1
    return vision_bank_indices_T.masked_fill(~vision_mask_T, -1)


def gather_vision_embeds(
    inputs_TD: torch.Tensor,
    *,
    vision_bank_VD: torch.Tensor,
    vision_bank_indices_T: torch.Tensor,
) -> torch.Tensor:
    """Gather packed vision features into their placeholder token positions.

    The gather-based counterpart to ``scatter_vision_embeds``: every placeholder
    token already knows its bank row, so no run detection is needed.
    """
    if vision_bank_VD.shape[0] == 0:
        return inputs_TD
    vision_bank_VD = vision_bank_VD.to(inputs_TD.dtype)
    is_vision_T1 = (vision_bank_indices_T >= 0).unsqueeze(-1)
    # clamp keeps the -1 sentinels in range; ``where`` discards those rows.
    gathered_TD = vision_bank_VD[vision_bank_indices_T.clamp(min=0)]
    return torch.where(is_vision_T1, gathered_TD, inputs_TD)


def scatter_vision_embeds(
    inputs_embeds: torch.Tensor,
    *,
    vision_embeds: torch.Tensor,
    vision_positions: list[tuple[int, int, int]],
) -> torch.Tensor:
    """Copy packed vision features into the text sequence at placeholder runs.

    Args:
        inputs_embeds: ``(T, D)`` text embeddings, modified in place.
        vision_embeds: packed vision features ``(total_tokens, dim)``.
        vision_positions: from ``get_vision_positions``.

    Returns:
        ``inputs_embeds``, with the vision spans overwritten.

    Raises:
        ValueError: if the spans consume a different number of embeddings than
            the packed vision output holds, which would mean the two streams
            disagree about how much vision there is.
    """
    vision_offset = 0
    for _, vision_start, num_tokens in vision_positions:
        inputs_embeds[vision_start : vision_start + num_tokens] = vision_embeds[
            vision_offset : vision_offset + num_tokens
        ].to(inputs_embeds.dtype)
        vision_offset += num_tokens

    if vision_offset != vision_embeds.shape[0]:
        raise ValueError(
            f"Vision placeholder runs consume {vision_offset} embeddings but "
            f"the packed vision output contains {vision_embeds.shape[0]}."
        )
    return inputs_embeds
