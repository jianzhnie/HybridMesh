"""Text processing utilities for multimodal datasets.

Vendored from torchtitan ``hf_datasets/multimodal/utils/text.py``; the
``pad_seq_len`` / ``pad_batch_dim`` helpers were dropped as unused.
"""

from __future__ import annotations

__all__ = ["insert_vision_placeholders"]


def insert_vision_placeholders(
    input_parts: list[str | None],
    num_vision_tokens: list[int],
    *,
    vision_start_token: str,
    vision_token: str,
    vision_end_token: str,
    eos_token: str = "",
) -> str:
    """Insert vision placeholder token sequences into text.

    Args:
        input_parts: Mixed list of text strings and ``None`` entries.
            Each ``None`` marks where a vision region (image or video) should
            be inserted; text strings are kept as-is.  Produced by the dataset
            processor which sets ``texts[idx] = None`` for each image/video.
        num_vision_tokens: Number of vision tokens per ``None`` placeholder.
        vision_start_token: Token marking start of a vision region.
        vision_token: Repeated placeholder token (image or video).
        vision_end_token: Token marking end of a vision region.
        eos_token: Appended at the end if non-empty.

    Returns:
        Text with vision placeholders expanded.
    """
    output_parts: list[str] = []
    vision_idx = 0

    for part in input_parts:
        if part is None and vision_idx < len(num_vision_tokens):
            output_parts.extend(
                [
                    vision_start_token,
                    *([vision_token] * num_vision_tokens[vision_idx]),
                    vision_end_token,
                ]
            )
            vision_idx += 1
        else:
            output_parts.append(part)

    result = "".join(output_parts).strip()
    if eos_token and not result.endswith(eos_token):
        result += eos_token
    return result
