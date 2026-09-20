"""Checkpointing: the manager contract, and the backends that implement it.

Ported from torchtitan's ``components/checkpointer/``. Three layers:

* ``base`` -- the contract (``BaseCheckpointManager``) plus the policies every
  backend shares: retention, step discovery, async draining, load selection.
  Nothing here knows how bytes are written.
* ``dcp`` -- ``CheckpointManager``, backed by ``torch.distributed.checkpoint``.
  This is the one hpmesh runs. It shards under FSDP, saves asynchronously, and
  loads in place.
* ``torch_checkpointing`` -- ``TorchCheckpointingManager``, backed by the
  ``torch_checkpointing`` package. That package is not a hpmesh dependency and
  is not installed in the development environment, so this module imports (its
  backend imports are deferred) but constructing the manager raises
  ``ImportError`` with an install hint. Not ported for completeness alone: it is
  the third of torchtitan's backends, and installing the package is the only
  step needed to use it.

``utils`` holds ``canonical_fqn``, which strips the activation-checkpoint
wrapper segment from an FQN.

State keys. A checkpoint is keyed by the top-level names in ``base``: ``model``,
``optimizer``, ``lr_scheduler``, ``dataloader``, ``train_state``. hpmesh fills in
the first two plus whatever the caller passes in ``states``:

* ``model`` -- a ``ModelWrapper`` over the model chunks.
* ``optimizer`` -- an ``OptimizerWrapper`` around the optimizer. A bare
  ``torch.optim.Optimizer`` satisfies DCP's ``Stateful``, but it cannot restore
  into a *fresh* optimizer: Adam's moments do not exist until the first
  ``step()``, and DCP writes into the tensors a state dict reports rather than
  calling ``load_state_dict``. The wrapper materializes them first. (No FQN
  flattening is needed alongside it: hpmesh rejects ``pp > 1``, the only
  configuration that would collide two optimizers' positional param group
  indices -- see ``OptimizerWrapper``.)
* ``train_state`` -- the ``Trainer`` itself. torchtitan's ``Trainer`` is a
  ``Stateful`` exposing ``step`` and ``ntokens_seen``; hpmesh's trainer exposes
  the same two through ``state_dict``/``load_state_dict`` for the same reason:
  they are the counters a resumed run needs.

``lr_scheduler`` and ``dataloader`` exist here as key names and nothing more --
hpmesh has neither component, so no run populates them yet.
"""

from .base import (
    DATALOADER,
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    TRAIN_STATE,
    BaseCheckpointManager,
    BaseCheckpointManagerConfig,
    CheckpointStorage,
    ModelWrapper,
    OptimizerWrapper,
    init_optim_state,
)
from .dcp import AsyncMode, CheckpointManager
from .utils import canonical_fqn

__all__ = [
    "AsyncMode",
    "BaseCheckpointManager",
    "BaseCheckpointManagerConfig",
    "CheckpointManager",
    "CheckpointStorage",
    "DATALOADER",
    "LR_SCHEDULER",
    "MODEL",
    "ModelWrapper",
    "OPTIMIZER",
    "OptimizerWrapper",
    "TRAIN_STATE",
    "canonical_fqn",
    "init_optim_state",
]
