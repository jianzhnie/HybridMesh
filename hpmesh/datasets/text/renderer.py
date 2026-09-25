"""Optional adapter to the ``renderers`` library for multi-turn chat SFT.

Vendored from torchtitan ``components/renderer.py``, minus the Configurable
machinery: hpmesh takes the renderer's name as a plain CLI string and builds
the library's typed config here instead of carrying one through the config
tree. The ``renderers`` package is NOT a hard dependency -- it is imported
lazily inside :func:`build_chat_renderer`, so an environment without it runs
the default chat-template path untouched, and enabling the renderer there
fails with an install hint rather than at import time.

Semantic contract kept from upstream: the renderer owns the token sequence
and the per-token loss mask (every assistant turn supervised, prompts and
non-content tokens masked), ``ensure_final_stop=True`` guarantees a terminal
stop token, and the mask is shifted with the labels because label ``j``
predicts token ``j + 1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...components.tokenizer import HuggingFaceTokenizer

__all__ = ["RENDERERS_INSTALL_HINT", "RendererTokenizerWrapper", "build_chat_renderer"]


RENDERERS_INSTALL_HINT = (
    "chat_renderer requires the optional `renderers` package, which is not "
    "installed. Install it with `pip install renderers==0.1.11`, or leave "
    "chat_renderer unset to use the chat-template path."
)


class RendererTokenizerWrapper:
    """Adapt hpmesh's loaded tokenizer to ``renderers.OffsetTokenizer``.

    ``renderers`` needs Hugging Face-style special-token attributes, raw
    encoding without automatic BOS/EOS, token-to-id lookup, and character
    offsets. The offsets identify tokens that come from message content
    (``is_content``). This adapter exposes that interface from the
    ``tokenizers.Tokenizer`` hpmesh already loaded; it does not load a second
    tokenizer.
    """

    def __init__(self, tokenizer: HuggingFaceTokenizer):
        # The `tokenizers.Tokenizer` inside; it has the offsets and token -> id lookup.
        self._tokenizer_backend = tokenizer.tokenizer
        self.name_or_path = tokenizer.tokenizer_path
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.bos_token_id = tokenizer.bos_id
        self.eos_token_id = tokenizer.eos_id
        # `tokenizers` returns None for unknown tokens; it has no unk id.
        self.unk_token_id = None

    def encode(
        self, text: str, add_special_tokens: bool = False, **kwargs
    ) -> list[int]:
        return self._tokenizer_backend.encode(
            text, add_special_tokens=add_special_tokens
        ).ids

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        return self._tokenizer_backend.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )

    def convert_tokens_to_ids(
        self, tokens: str | list[str]
    ) -> int | None | list[int | None]:
        if isinstance(tokens, str):
            return self._tokenizer_backend.token_to_id(tokens)
        return [self._tokenizer_backend.token_to_id(token) for token in tokens]

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict:
        encoding = self._tokenizer_backend.encode(
            text, add_special_tokens=add_special_tokens
        )
        output: dict[str, Any] = {"input_ids": encoding.ids}
        if return_offsets_mapping:
            output["offset_mapping"] = encoding.offsets
        return output


def build_chat_renderer(*, tokenizer: HuggingFaceTokenizer, renderer_name: str):
    """Build a ``renderers`` renderer over hpmesh's tokenizer.

    ``renderer_name`` is the exact name of a config class the ``renderers``
    package exports at top level, e.g. ``Qwen3RendererConfig``. Only
    model-specific renderers are accepted: the library's ``auto`` resolution
    keys off ``tokenizer.name_or_path`` exact matches and its ``default``
    renderer needs HF ``apply_chat_template`` special-token variables this
    tokenizer does not provide, and both would silently produce different
    tokens, so they are refused like upstream refuses them.
    """
    import importlib

    try:
        renderers = importlib.import_module("renderers")
    except ImportError as exc:
        raise ImportError(RENDERERS_INSTALL_HINT) from exc

    config_cls = getattr(renderers, renderer_name, None)
    if config_cls is None:
        raise ValueError(
            f"renderers has no top-level config named {renderer_name!r}. "
            "Pass the exact name of one of its renderer config classes, "
            "e.g. 'Qwen3RendererConfig'."
        )
    config = config_cls()
    if config.name == "auto":
        raise ValueError(
            f"{renderer_name} resolves by exact match of tokenizer.name_or_path "
            f"({tokenizer.tokenizer_path!r}) against renderers' "
            "MODEL_RENDERER_MAP, else falls back to DefaultRenderer "
            "(unsupported here). Pick the model's renderer, e.g. "
            "Qwen3RendererConfig."
        )
    if config.name == "default":
        raise ValueError(
            f"{renderer_name} needs Hugging Face apply_chat_template; hpmesh's "
            "template rendering lacks its special-token variables (bos_token, "
            "...) and would silently produce different tokens. Pick the "
            "model's renderer, e.g. Qwen3RendererConfig."
        )
    return renderers.create_renderer(
        tokenizer=RendererTokenizerWrapper(tokenizer), config=config
    )
