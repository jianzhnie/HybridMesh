"""Trainer-side components: metrics, tokenizer, optimizer, checkpointing.

One subpackage (``checkpointer/``) and three modules (``loss``, ``metrics``,
``tokenizer``); each has its own package-level docstring for the split it
represents. Nothing here imports the trainer layer -- the trainer imports these,
and the checkpointer index is lazy so that ``llmtuner.config`` can read the
checkpoint state keys without pulling the backends in.
"""
