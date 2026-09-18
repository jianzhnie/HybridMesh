"""Parameter initialization helpers.

Vendored from torchtitan ``models/common/param_init.py``. Upstream these exist
because a model config names its initializer as a callable
(``param_init={"weight": partial(trunc_normal_, std=0.02)}``). hpmesh builds HF
models, which run their own ``_init_weights``, so the config-plumbing half of
that convention does not carry over -- what is kept are the two initializers
themselves, which are ordinary callables usable with ``nn.init``-style APIs or
``module.apply()``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

__all__ = ["depth_scaled_std", "skip_param_init"]


def skip_param_init(param: nn.Parameter) -> None:
    """No-op initializer: explicitly skip initialization for a parameter.

    Useful when a parameter is tied to another (e.g. weight tying), where
    initializing it independently would silently break the tie.
    """
    pass


def depth_scaled_std(base_std: float, layer_id: int) -> float:
    """Depth-dependent std, ``base_std / sqrt(2 * (layer_id + 1))``.

    Later layers get a smaller init so the residual stream's variance does not
    grow with depth. Returns the std rather than writing it, so the caller
    decides which distribution to hand it to.
    """
    return base_std / (2 * (layer_id + 1)) ** 0.5


# Named for symmetry with ``param_init``'s role upstream: a mapping from a
# parameter name to the callable that initializes it.
ParamInitFn = Callable[[nn.Parameter], None]
