"""The Hugging Face adaptation layer: build it, run it, count it, checkpoint it.

Four nodes:

* ``factory.py`` -- builds the HF ``PretrainedConfig`` (offline architecture,
  hub id or local checkpoint), resolves the ``ForCausalLM`` class, and
  materializes a meta-device model. The construction seam.
* ``model.py`` -- ``HFTransformerModel``: the five-part contract
  (``tok_embeddings`` / ``layers`` / ``norm`` / ``lm_head`` / ``rotary_emb``),
  the packed-document attention wiring, and the forward the trainer calls.
  This is the consumer of everything else here, and its name follows upstream's
  ``experiments/transformers_modeling_backend/model.py``.
* ``flops.py`` -- training FLOPs per token, the MFU denominator. Pure
  arithmetic over an HF config, kept apart from the building; the trainer's
  entry point (``num_flops_per_token``) stays in ``factory.py`` because it is
  the one that has to resolve the config first.
* ``state_dict_adapter.py`` -- ``HFTransformerStateDictAdapter``: the model's
  state-dict FQNs to standard HF safetensors keys.

Like the other indexes in this package, this one re-exports nothing: importing
one node must not drag in its siblings (``model`` pulls in the whole Hugging
Face modeling stack, ``factory``/``flops`` need only the config classes), so
import the leaf you want.
"""
