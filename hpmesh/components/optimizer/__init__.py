# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer machinery: parameters, state serialization, and containers.

The same split as torchtitan's ``components/optimizer/`` package, at the same
paths under the same names, so the two trees can be read side by side:

* ``utils`` -- ``init_optim_state`` and the flat, FQN-keyed state-dict helpers.
  Free functions over any ``torch.optim.Optimizer``; model-agnostic.
* ``optimizer`` -- ``OptimizersContainer``, the single ``Optimizer`` the training
  loop drives, over one inner optimizer per (model part, optimizer name)
  ``ParamGroupConfig`` produced; plus ``OptimizerWrapper``, the narrower
  single-optimizer ``Stateful`` view.
* ``lr_scheduler`` -- ``LRSchedulersContainer``, one ``LambdaLR`` per inner
  optimizer, and the WSD curve they share.

``utils`` is vendored from torchtitan's ``components/optimizer/utils.py``;
``init_optim_state`` was moved verbatim out of ``components/checkpointer/base.py``,
where it lived only because hpmesh had nowhere to put it.
"""

from .lr_scheduler import LRSchedulersContainer, build_lr_scheduler
from .optimizer import OptimizersContainer, OptimizerWrapper
from .utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)

__all__ = [
    "LRSchedulersContainer",
    "OptimizersContainer",
    "OptimizerWrapper",
    "build_lr_scheduler",
    "get_flat_optim_state_dict",
    "init_optim_state",
    "load_flat_optim_state_dict",
]
