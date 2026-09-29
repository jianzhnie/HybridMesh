"""Checkpoint format names: the state-dict keys, and the HF index file.

Both ``components/checkpointer`` (which writes states under these keys) and
``llmtuner/config`` (which validates policies over them, such as
``exclude_from_loading``) need the same strings. The config layer reads this
submodule directly -- it is dependency-free, so that import never pulls the
checkpointer backends (``base``/``dcp`` and their torch.distributed surface)
into ``llmtuner.config``; the package ``__init__`` re-exports those lazily for
the same reason. ``models/hf/state_dict_adapter.py`` reads the same way for the
index name: it writes that file, the DCP checkpointer probes for it, and one
spelling means the two cannot drift apart.
"""

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"
EMA = "ema"

# Written by the HF safetensors export (``HFTransformerStateDictAdapter``) and
# probed by the DCP checkpointer's "is this a resumable checkpoint" test.
SAFETENSORS_INDEX = "model.safetensors.index.json"

__all__ = [
    "DATALOADER",
    "EMA",
    "LR_SCHEDULER",
    "MODEL",
    "OPTIMIZER",
    "SAFETENSORS_INDEX",
    "TRAIN_STATE",
]
