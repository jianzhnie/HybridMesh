"""Attention pieces: the fused QKV projection and the mask modifiers.

Two nodes, both splits of torchtitan ``models/common/attention.py`` (the rest of
that module is a family of attention implementations that would duplicate the HF
models llmtuner wraps):

* ``qkv.py`` -- ``QKVLinear`` (one fused QKV projection, split along the
  per-KV-group axis) and ``local_head_split``.
* ``masks.py`` -- the flex-attention mask modifiers and the varlen metadata
  builder for packed documents.

Unlike the other package indexes here, this one re-exports nothing on purpose.
``masks.py`` imports ``torch.nn.attention.flex_attention`` at module scope, which
not every torch build has; keeping the index inert is what lets
``models.common`` (which re-exports ``qkv``) stay importable on a CPU-only or
pre-flex build. Import the leaf you want.
"""
