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
* ``optimizer`` -- the ``OptimizersContainer`` itself, passed through. It is
  already a ``Stateful`` whose state dict is flat and FQN-keyed, which is what
  DCP needs to reshard a pipeline checkpoint, and it materializes a fresh
  optimizer's Adam moments before DCP plans a load. Both jobs used to be an
  ``OptimizerWrapper``'s; the container absorbed them. The wrapper still exists
  in ``components/optimizer/`` for the single-optimizer case, but nothing in
  this package is handed one.
* ``train_state`` -- the ``Trainer`` itself. torchtitan's ``Trainer`` is a
  ``Stateful`` exposing ``step`` and ``ntokens_seen``; hpmesh's trainer exposes
  the same two through ``state_dict``/``load_state_dict`` for the same reason:
  they are the counters a resumed run needs.
* ``lr_scheduler`` -- the ``LRSchedulersContainer`` from
  ``components/optimizer/lr_scheduler.py``, registered only when the run has a
  non-constant schedule. A constant lr is a function of nothing, so saving it
  would be a key that can only disagree with the optimizer it already sits
  beside.
* ``dataloader`` -- the ``BaseDataLoader``, registered only when it is loadable.
  See the trainer's ``_build_dataloader`` for why the synthetic one is not.
"""

from ..optimizer import OptimizerWrapper, init_optim_state
from .base import (
    DATALOADER,
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    TRAIN_STATE,
    BaseCheckpointManager,
    CheckpointStorage,
    ModelWrapper,
)
from .dcp import AsyncMode, CheckpointManager
from .utils import canonical_fqn

__all__ = [
    "AsyncMode",
    "BaseCheckpointManager",
    "CheckpointManager",
    "CheckpointStorage",
    "DATALOADER",
    "LR_SCHEDULER",
    "MODEL",
    "ModelWrapper",
    "OPTIMIZER",
    "TRAIN_STATE",
    "canonical_fqn",
]
