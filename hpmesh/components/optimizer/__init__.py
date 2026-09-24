"""Optimizer machinery: parameters, state serialization, and containers.

The same split as torchtitan's ``components/optimizer/`` package, at the same
paths under the same names, so the two trees can be read side by side:

* ``utils`` -- ``init_optim_state`` and the flat, FQN-keyed state-dict helpers.
  Free functions over any ``torch.optim.Optimizer``; model-agnostic.
* ``optimizer`` -- ``OptimizersContainer``, the single ``Optimizer`` the training
  loop drives, over one inner optimizer per (model part, optimizer name)
  ``ParamGroupConfig`` produced.
* ``lr_scheduler`` -- ``LRSchedulersContainer``, one ``LambdaLR`` per inner
  optimizer, and the WSD curve they share.
* ``ema`` -- ``EMA``, an online exponential moving average of the weights,
  shaped as a pseudo-``OptimizersContainer`` so its checkpoint state rides the
  same flat, FQN-keyed format.

``utils`` is vendored from torchtitan's ``components/optimizer/utils.py``;
``init_optim_state`` was moved verbatim out of ``components/checkpointer/base.py``,
where it lived only because hpmesh had nowhere to put it.
"""

from .ema import EMA
from .lr_scheduler import LRSchedulersContainer, build_lr_scheduler
from .optimizer import OptimizersContainer
from .utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)

__all__ = [
    "EMA",
    "LRSchedulersContainer",
    "OptimizersContainer",
    "build_lr_scheduler",
    "get_flat_optim_state_dict",
    "init_optim_state",
    "load_flat_optim_state_dict",
]
