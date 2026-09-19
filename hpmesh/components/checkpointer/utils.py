# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FQN helpers shared by checkpointing and the components it serializes."""

__all__ = [
    "canonical_fqn",
]

# The segment the activation-checkpoint wrapper inserts into named_parameters().
# It can appear at any level of an FQN and is not part of the canonical model
# contract. torch.compile is applied in place and adds no segment.
_WRAPPER_PREFIXES: tuple[str, ...] = ("_checkpoint_wrapped_module",)


def canonical_fqn(name: str, prefixes: tuple[str, ...] = _WRAPPER_PREFIXES) -> str:
    """Strip wrapper segments from a dotted FQN.

    A segment may appear at any level, e.g.
    ``layers.0._checkpoint_wrapped_module.attention.wq.weight`` ->
    ``layers.0.attention.wq.weight``.

    This is what lets an optimizer state keyed on parameter FQNs stay stable
    across a run that turns activation checkpointing on or off, which is the
    difference between a checkpoint that resumes and one that silently loads
    nothing for the wrapped layers.
    """
    return ".".join(p for p in name.split(".") if p not in prefixes)
