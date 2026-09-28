"""The model layer: the HF adaptation, and the model vocabulary it is built on.

Two halves, and the split is the one that matters when reading code:

* ``hf/`` -- the adaptation layer. ``wrapper.py`` holds the
  ``HFTransformerModel`` the trainer drives, ``factory.py`` builds the HF config
  and resolves the model class, ``state_dict_adapter.py`` maps the checkpoint
  FQNs. Only this half knows about Hugging Face.
* ``common/`` -- the model vocabulary, vendored from torchtitan
  ``models/common/``: attention pieces (``attention/``), MoE (``moe/``),
  feed-forward, linear, embedding, RoPE, activations, the aux-loss carrier and
  the multimodal glue. Nothing here knows which family a model belongs to.

Two conventions, both inherited from ``common/``: a package index is for
discoverability while a leaf module is what you import, and a component with
real behavior of its own gets its own file.
"""
