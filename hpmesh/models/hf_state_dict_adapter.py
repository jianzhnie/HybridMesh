"""Hugging Face safetensors mapping for :class:`HFTransformerModel`.

The wrapper stores a HF ``ForCausalLM`` below ``self.model``. Consequently its
state-dict FQNs have exactly one extra ``model.`` prefix compared with the
checkpoint produced by Transformers. No tensor conversion is required.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from torch.distributed.checkpoint import HuggingFaceStorageReader

logger = logging.getLogger(__name__)


class HFTransformerStateDictAdapter:
    """Translate hpmesh wrapper FQNs to standard HF safetensors FQNs."""

    def __init__(self, model_config: Any, hf_assets_path: str | None) -> None:
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self._hf_keys: set[str] | None = None
        self.fqn_to_index_mapping: dict[str, int] | None = None

        if hf_assets_path is None:
            return
        index_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            # HuggingFaceStorageReader also supports a single model.safetensors.
            return
        with open(index_path, encoding="utf-8") as stream:
            weight_map = json.load(stream)["weight_map"]
        self._hf_keys = set(weight_map)
        self.fqn_to_index_mapping = {
            key: int(match.group())
            for key, filename in weight_map.items()
            if (match := re.search(r"\d+", filename)) is not None
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state = {
            key.removeprefix("model."): value for key, value in state_dict.items()
        }
        if self._hf_keys is not None:
            requested = set(hf_state)
            missing = self._hf_keys - requested
            unexpected = requested - self._hf_keys
            # A tied lm_head is legitimately omitted from a HF checkpoint.
            if getattr(self.model_config, "tie_word_embeddings", False):
                unexpected.discard("lm_head.weight")
                hf_state.pop("lm_head.weight", None)
            if missing or unexpected:
                raise ValueError(
                    "HF checkpoint keys do not exactly cover the hpmesh model: "
                    f"missing={sorted(missing)[:8]}, "
                    f"unexpected={sorted(unexpected)[:8]}"
                )
        return hf_state

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        if (
            "lm_head.weight" not in hf_state_dict
            and "model.embed_tokens.weight" in hf_state_dict
            and getattr(self.model_config, "tie_word_embeddings", False)
        ):
            hf_state_dict["lm_head.weight"] = hf_state_dict[
                "model.embed_tokens.weight"
            ]
        return {f"model.{key}": value for key, value in hf_state_dict.items()}

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            raise NotImplementedError("Quantized HF checkpoints are not supported")
        return HuggingFaceStorageReader(path)
