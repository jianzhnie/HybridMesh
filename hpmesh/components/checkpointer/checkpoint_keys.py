"""The top-level state-dict keys a checkpoint is keyed by.

Both ``components/checkpointer`` (which writes states under these keys) and
``hpmesh/config`` (which validates policies over them, such as
``exclude_from_loading``) need the same strings. The config layer reads this
submodule directly -- it is dependency-free, so that import never pulls the
checkpointer backends (``base``/``dcp`` and their torch.distributed surface)
into ``hpmesh.config``; the package ``__init__`` re-exports those lazily for
the same reason.
"""

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"
EMA = "ema"

__all__ = [
    "DATALOADER",
    "EMA",
    "LR_SCHEDULER",
    "MODEL",
    "OPTIMIZER",
    "TRAIN_STATE",
]
