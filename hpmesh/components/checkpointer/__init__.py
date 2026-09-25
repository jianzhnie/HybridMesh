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
``optimizer``, ``lr_scheduler``, ``dataloader``, ``train_state``, ``ema``.
hpmesh fills in the first two plus whatever the caller passes in ``states``:

* ``model`` -- a ``ModelWrapper`` over the model chunks.
* ``optimizer`` -- the ``OptimizersContainer`` itself, passed through. It is
  already a ``Stateful`` whose state dict is flat and FQN-keyed, which is what
  DCP needs to reshard a pipeline checkpoint, and it materializes a fresh
  optimizer's Adam moments before DCP plans a load.
* ``train_state`` -- the ``Trainer`` itself. torchtitan's ``Trainer`` is a
  ``Stateful`` exposing ``step`` and ``ntokens_seen``; hpmesh's trainer exposes
  the same two through ``state_dict``/``load_state_dict`` for the same reason:
  they are the counters a resumed run needs.
* ``lr_scheduler`` -- the ``LRSchedulersContainer`` from
  ``components/optimizer/lr_scheduler.py``, always registered. It holds one
  integer, ``last_epoch``, that nothing else in the checkpoint carries: the
  optimizer restores the ``base_lrs`` so the *current* lr comes back right, but
  the step count is the scheduler's own, and a resumed run's fresh scheduler
  starts it at 0. Without it the curve restarts on the step after a resume --
  invisible while the lr is constant, wrong for the rest of the run once warmup
  or decay is set.
* ``dataloader`` -- the ``BaseDataLoader``, registered only when it is loadable.
  See the trainer's ``_build_dataloader`` for why the synthetic one is not.
* ``ema`` -- the ``EMA`` pseudo-optimizer from
  ``components/optimizer/ema.py``, registered only when the run configures one
  (``training.ema_config``). Its state dict is the same flat, FQN-keyed layout
  as optimizer state. A load that restores the model but not this key -- an
  ``exclude_from_loading=["ema"]`` load, or any model-only load -- cold-starts
  the average from the just-loaded weights (see ``dcp.CheckpointManager``).
"""

# Lazy (PEP 562): ``checkpoint_keys`` / ``filesystem`` live in this package and
# are read by ``hpmesh/config``, which must stay importable without the
# torch.distributed surface ``base``/``dcp`` pull in (DTensor & friends are
# absent on older torch). Eager re-exports would make any submodule import pay
# for the backends.
_EXPORT_SOURCES = {
    "BaseCheckpointManager": "base",
    "CheckpointStorage": "base",
    "ModelWrapper": "base",
    "DATALOADER": "base",
    "EMA": "base",
    "LR_SCHEDULER": "base",
    "MODEL": "base",
    "OPTIMIZER": "base",
    "TRAIN_STATE": "base",
    "AsyncMode": "dcp",
    "CheckpointManager": "dcp",
    "canonical_fqn": "utils",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    """Resolve backend names on first touch (PEP 562 lazy re-export)."""
    source = _EXPORT_SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{source}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
