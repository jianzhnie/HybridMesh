"""Attention pieces: the fused QKV projection and the mask modifiers.

Two nodes, both splits of torchtitan ``models/common/attention.py`` (the rest of
that module is a family of attention implementations that would duplicate the HF
models llmtuner wraps):

* ``qkv.py`` -- ``QKVLinear`` (one fused QKV projection, split along the
  per-KV-group axis) and ``local_head_split``.
* ``masks.py`` -- the flex-attention mask modifiers and the varlen metadata
  builder for packed documents.

Unlike the other package indexes here, this one re-exports nothing at all: both
nodes are imported by name from where they are used (the wrapper, the CP kernel,
the tests), and an inert index is one less thing to keep in step. Import the
leaf you want.
"""
