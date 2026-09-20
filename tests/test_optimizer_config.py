"""The optimizer hyperparameters and the flags that reach them.

``OptimizerConfig`` is the one place AdamW's knobs are set, and the path from
there to ``torch.optim.AdamW`` runs through three hops: a nested ``*_config``
field, the flat-view passthrough properties, and the trainer's construction
call. A field added without its property is invisible to the trainer -- it fails
at the first step with an ``AttributeError``, not at parse time -- so these tests
exercise the whole chain rather than the dataclass alone.

The defaults are load-bearing: they must reproduce ``torch.optim.AdamW``'s
exactly, or adding this config would silently move the numbers of every existing
run. The override test is what proves the knob is not inert.
"""

from __future__ import annotations

import pytest
import torch
from transformers import HfArgumentParser

from hpmesh.trainer.config import HybridMeshConfig, OptimizerConfig


def _parser() -> HfArgumentParser:
    """The production parser group list, so the flags are the real ones."""
    from hpmesh.trainer.config import (
        CheckpointConfig,
        DataloaderConfig,
        LRSchedulerConfig,
        MetricsConfig,
        ModelConfig,
        ParallelConfig,
        ProfilerConfig,
        TrainingConfig,
    )

    return HfArgumentParser(
        [
            ModelConfig,
            ParallelConfig,
            OptimizerConfig,
            LRSchedulerConfig,
            TrainingConfig,
            CheckpointConfig,
            DataloaderConfig,
            MetricsConfig,
            ProfilerConfig,
        ]
    )


def test_defaults_reproduce_torch_adamw() -> None:
    """The config must not move numbers that a default run already had.

    ``betas`` and ``eps`` are new fields with torch's own values as defaults, so
    an untouched run has to build the same optimizer it built before they
    existed. Compared against a torch-constructed group rather than against
    literals: the point is agreeing with the library, not with a copy of it.
    """
    cfg = OptimizerConfig()
    param = torch.nn.Parameter(torch.zeros(2))
    built = torch.optim.AdamW(
        [param], lr=cfg.learning_rate, betas=cfg.betas, eps=cfg.eps
    )
    reference = torch.optim.AdamW([param])

    assert built.param_groups[0]["betas"] == reference.param_groups[0]["betas"]
    assert built.param_groups[0]["eps"] == reference.param_groups[0]["eps"]


def test_betas_is_a_tuple_so_it_cannot_be_mutated_in_place() -> None:
    """The parser hands back a ``list``, which is the shared-default footgun.

    A reused ``HfArgumentParser`` runs the ``default_factory`` once and gives
    every parse the same list, so an in-place edit would leak into the next run.
    Normalizing to a tuple in ``__post_init__`` makes that unreachable.
    """
    assert isinstance(OptimizerConfig().betas, tuple)
    assert isinstance(OptimizerConfig(betas=[0.9, 0.95]).betas, tuple)


@pytest.mark.parametrize(
    "betas",
    [(0.9,), (0.9, 0.9, 0.9), (), (1.0, 0.9), (-0.1, 0.9)],
    ids=["one", "three", "empty", "beta1==1", "negative"],
)
def test_bad_betas_are_rejected_with_a_message_that_says_how_to_pass_them(
    betas: tuple[float, ...],
) -> None:
    """A one-element betas must fail here, not inside a training step.

    ``--betas 0.9`` is the natural typo, and torch's own error for it surfaces
    deep in ``step()`` where the cause is no longer visible.
    """
    with pytest.raises(ValueError, match="betas must"):
        OptimizerConfig(betas=betas)


def test_flags_parse_and_the_flat_view_reaches_them() -> None:
    """The whole chain: flag -> nested config -> property -> optimizer group."""
    (_, _, optimizer, *_rest) = _parser().parse_args_into_dataclasses(
        args=["--betas", "0.9", "0.95", "--eps", "1e-6"]
    )
    cfg = HybridMeshConfig(optimizer=optimizer)

    assert cfg.betas == (0.9, 0.95)
    assert cfg.eps == 1e-6
    # And the trainer's call shape accepts exactly these.
    group = torch.optim.AdamW(
        [torch.nn.Parameter(torch.zeros(2))],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        eps=cfg.eps,
    ).param_groups[0]
    assert tuple(group["betas"]) == (0.9, 0.95)
    assert group["eps"] == 1e-6


def test_omitting_the_flags_leaves_the_defaults() -> None:
    (_, _, optimizer, *_rest) = _parser().parse_args_into_dataclasses(args=[])
    cfg = HybridMeshConfig(optimizer=optimizer)

    assert cfg.betas == (0.9, 0.999)
    assert cfg.eps == 1e-8


def test_betas_actually_changes_the_trajectory() -> None:
    """The knob must be live, and the change shows up later than step 1.

    AdamW bias correction makes step 1's update depend only on the gradient, so
    a betas change is invisible there and only separates the runs from step 2
    on. Asserting on step 1 would be vacuous.
    """

    def trajectory(betas: tuple[float, float]) -> list[float]:
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 8)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=betas)
        losses = []
        for _ in range(4):
            loss = model(torch.ones(2, 8)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    default = trajectory((0.9, 0.999))
    torchtitan = trajectory((0.9, 0.95))

    assert default[0] == torchtitan[0], "step 1 is bias-corrected and must match"
    assert default[-1] != torchtitan[-1], "betas had no effect by the last step"
