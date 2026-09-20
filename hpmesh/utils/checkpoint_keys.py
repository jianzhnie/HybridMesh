"""The top-level state-dict keys a checkpoint is keyed by.

Neutral ground: both ``components/checkpointer`` (which writes states under
these keys) and ``trainer/config`` (which validates policies over them, such as
``exclude_from_loading``) need the same strings, and neither may import the
other -- the checkpointer already reads ``trainer.config`` under
``TYPE_CHECKING``, so a runtime import back would close a cycle. Keys live one
level below both, next to the other shared vocabulary in ``utils/``.
"""

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"

__all__ = [
    "DATALOADER",
    "LR_SCHEDULER",
    "MODEL",
    "OPTIMIZER",
    "TRAIN_STATE",
]
