"""The Hugging Face adaptation layer: build it, wrap it, checkpoint it.

Three nodes:

* ``factory.py`` -- builds the HF ``PretrainedConfig`` (offline architecture,
  hub id or local checkpoint), resolves the ``ForCausalLM`` class, materializes
  a meta-device model, and counts FLOPs per token (the MFU denominator).
* ``wrapper.py`` -- ``HFTransformerModel``: the five-part contract
  (``tok_embeddings`` / ``layers`` / ``norm`` / ``lm_head`` / ``rotary_emb``),
  the packed-document attention wiring, and the forward the trainer calls.
* ``state_dict_adapter.py`` -- ``HFTransformerStateDictAdapter``: the wrapper's
  state-dict FQNs to standard HF safetensors keys.

Like the other indexes in this package, this one re-exports nothing: importing
one node must not drag in its siblings (``wrapper`` pulls in the whole
Hugging Face modeling stack, ``factory`` only needs the config classes), so
import the leaf you want.
"""
