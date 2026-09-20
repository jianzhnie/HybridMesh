"""Config validation: the rejections, since the acceptances are the default runs.

Every ``__post_init__`` in ``trainer/config.py`` guards a combination that would
otherwise fail late -- inside a distributed launch, a checkpoint load, or a mesh
build -- or, worse, silently train something other than what was asked for. The
valid defaults are exercised by every other test in this suite (they construct a
``HybridMeshConfig``); these are the branches that only run when a user is wrong.

Each ``with pytest.raises`` is paired with a positive case where the guard has a
boundary worth pinning (``-1`` is allowed for dp_shard, ``0`` is not), so the
test cannot pass by the constructor rejecting everything.
"""

from __future__ import annotations

import pytest

from hpmesh.components.checkpointer import LR_SCHEDULER, MODEL, OPTIMIZER
from hpmesh.trainer.config import (
    CheckpointConfig,
    LRSchedulerConfig,
    ParallelConfig,
    TrainingConfig,
)

# -- ParallelConfig ----------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
        "expert_parallel_size",
    ],
)
def test_a_size_below_one_is_rejected(field: str) -> None:
    """A zero size is a division by zero three layers down; catch it at parse."""
    with pytest.raises(ValueError, match=f"{field} must be >= 1"):
        ParallelConfig(**{field: 0})


def test_dp_shard_accepts_minus_one_as_derive_but_not_zero() -> None:
    """``-1`` means "derive it", ``0`` is a typo for neither."""
    assert ParallelConfig(data_parallel_shard_size=-1).data_parallel_shard_size == -1
    with pytest.raises(ValueError, match="must be >= 1 or -1"):
        ParallelConfig(data_parallel_shard_size=0)


def test_an_empty_load_balancer_is_rejected_rather_than_coerced() -> None:
    """``""`` is not ``None``: one disables, the other is a mistake."""
    with pytest.raises(ValueError, match="cannot be an empty string"):
        ParallelConfig(context_parallel_load_balancer="")
    assert (
        ParallelConfig(
            context_parallel_load_balancer=None
        ).context_parallel_load_balancer
        is None
    )


def test_a_load_balancer_that_is_not_a_known_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        ParallelConfig(context_parallel_load_balancer="roundrobin")


def test_an_unknown_context_parallel_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        ParallelConfig(context_parallel_strategy="ring")


def test_ulysses_cannot_share_a_load_balancer() -> None:
    """The all-to-all reassembles by rank order, which only a contiguous split gives.

    Worth noting how easy this is to trip: the load balancer defaults to
    ``'headtail'``, so selecting ``ulysses`` on its own is already the illegal
    pairing -- the user has to actively turn the balancer off.
    """
    with pytest.raises(ValueError, match="requires.*load_balancer=None"):
        ParallelConfig(context_parallel_strategy="ulysses")

    cfg = ParallelConfig(
        context_parallel_strategy="ulysses",
        context_parallel_load_balancer=None,
    )
    assert cfg.context_parallel_load_balancer is None
    # The default strategy keeps the default balancer; the pairing is only
    # constrained the other way round.
    assert ParallelConfig().context_parallel_load_balancer == "headtail"


def test_symmetric_memory_is_rejected_off_a_supported_device() -> None:
    """On a machine without the capability there is no silent fallback to catch."""
    # The development machine is CPU-only, so this is the unsupported path.
    with pytest.raises(ValueError, match="compute capability 9.0"):
        ParallelConfig(enable_fsdp_symm_mem=True)


def test_an_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        ParallelConfig(backend="mpi")


def test_an_unknown_pipeline_schedule_is_rejected() -> None:
    """``get_schedule_class`` is the authority; the error names the bad value."""
    with pytest.raises(
        ValueError, match="Invalid parallelism.pipeline_parallel_schedule"
    ):
        ParallelConfig(pipeline_parallel_schedule="NotASchedule")


# -- LRSchedulerConfig -------------------------------------------------------


def test_a_negative_warmup_is_rejected() -> None:
    with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
        LRSchedulerConfig(warmup_steps=-1)
    assert LRSchedulerConfig(warmup_steps=0).warmup_steps == 0


def test_total_steps_if_given_must_be_positive() -> None:
    """``None`` means "take it from training"; an explicit value must be real."""
    assert LRSchedulerConfig(total_steps=None).total_steps is None
    with pytest.raises(ValueError, match="total_steps must be >= 1"):
        LRSchedulerConfig(total_steps=0)


def test_decay_ratio_is_a_fraction() -> None:
    with pytest.raises(ValueError, match="decay_ratio must be in"):
        LRSchedulerConfig(decay_ratio=1.5)
    with pytest.raises(ValueError, match="decay_ratio must be in"):
        LRSchedulerConfig(decay_ratio=-0.1)
    assert LRSchedulerConfig(decay_ratio=1.0).decay_ratio == 1.0


def test_min_lr_factor_is_half_open_at_one() -> None:
    """``1.0`` would mean "decay to the base lr", i.e. not decay at all."""
    assert LRSchedulerConfig(min_lr_factor=0.0).min_lr_factor == 0.0
    with pytest.raises(ValueError, match=r"min_lr_factor must be in \[0, 1\)"):
        LRSchedulerConfig(min_lr_factor=1.0)


# -- CheckpointConfig --------------------------------------------------------


def test_a_whitespace_folder_is_not_a_folder() -> None:
    with pytest.raises(ValueError, match="'folder' field cannot be empty"):
        CheckpointConfig(folder="   ")


def test_load_step_is_either_derive_or_non_negative() -> None:
    assert CheckpointConfig(load_step=-1).load_step == -1
    with pytest.raises(ValueError, match="load_step must be -1 or non-negative"):
        CheckpointConfig(load_step=-2)


def test_negative_retention_is_rejected() -> None:
    with pytest.raises(ValueError, match="keep_latest_k cannot be negative"):
        CheckpointConfig(keep_latest_k=-1)


def test_the_model_can_never_be_excluded_from_a_load() -> None:
    """Loading everything *except* the weights is never what a user means."""
    with pytest.raises(ValueError, match="shouldn't be in exclude_from_loading"):
        CheckpointConfig(exclude_from_loading=[MODEL])


def test_excluding_the_optimizer_must_exclude_the_schedule_too() -> None:
    """``LRSchedulersContainer`` reads ``base_lrs`` off the optimizers it restores.

    A schedule without its optimizers would restore against a cold optimizer and
    silently restart the lr curve. The pairing is enforced rather than inferred.
    """
    with pytest.raises(ValueError, match=f"{LR_SCHEDULER} must be excluded"):
        CheckpointConfig(exclude_from_loading=[OPTIMIZER])
    # The paired form is accepted.
    cfg = CheckpointConfig(exclude_from_loading=[OPTIMIZER, LR_SCHEDULER])
    assert set(cfg.exclude_from_loading) == {OPTIMIZER, LR_SCHEDULER}


def test_a_relative_initial_load_path_is_rejected() -> None:
    """Resuming from a relative path resolves against the launch dir, not the cwd."""
    with pytest.raises(ValueError, match="must be an absolute path or a remote URI"):
        CheckpointConfig(initial_load_path="./weights")
    assert (
        CheckpointConfig(initial_load_path="/abs/weights").initial_load_path
        == "/abs/weights"
    )


def test_hf_load_modes_imply_each_other() -> None:
    """Each ``*_in_hf`` flag needs a partner that is not on by default.

    The ``*_model_only`` partners default to ``True``, so the Implies fire only
    when a caller turns the partner *off* -- which is the mistake worth pinning.
    """
    with pytest.raises(ValueError, match="requires initial_load_model_only"):
        CheckpointConfig(initial_load_in_hf=True, initial_load_model_only=False)
    with pytest.raises(ValueError, match="requires initial_load_in_hf"):
        CheckpointConfig(initial_load_in_hf_quantized=True)
    with pytest.raises(ValueError, match="requires last_save_model_only"):
        CheckpointConfig(last_save_in_hf=True, last_save_model_only=False)
    # The pairing the defaults describe is accepted without spelling it out.
    assert CheckpointConfig(initial_load_in_hf=True).initial_load_model_only is True


def test_an_unknown_async_mode_is_rejected_and_the_valid_one_is_lowered() -> None:
    with pytest.raises(ValueError, match="Invalid async_mode"):
        CheckpointConfig(async_mode="Threaded")
    # The field is normalized in place, so a mixed-case spelling still works.
    assert CheckpointConfig(async_mode="ASYNC").async_mode == "async"


# -- TrainingConfig ----------------------------------------------------------


@pytest.mark.parametrize(
    "field, bad",
    [
        ("global_batch_size", 0),
        ("max_seq_len", 0),
        ("steps", 0),
        ("gradient_accumulation_steps", 0),
    ],
)
def test_a_non_positive_loop_parameter_is_rejected(field: str, bad: int) -> None:
    """Each of these divides or iterates; zero is a hang or a ZeroDivision."""
    with pytest.raises(ValueError, match=f"{field} must be >= 1"):
        TrainingConfig(**{field: bad})


# -- the flat view the trainer reads -------------------------------------------
#
# ``HybridMeshConfig`` exposes the trainer's scalars as hand-written properties,
# so a new field on a group stays invisible to the trainer until its passthrough
# exists. The failure mode is an AttributeError on the first training step, not
# at parse time -- so the passthroughs are worth pinning explicitly.


def test_accumulation_and_gc_freq_reach_the_flat_view() -> None:
    """The trainer reads both off ``cfg``, not off ``cfg.training``."""
    from hpmesh.trainer import HybridMeshConfig

    cfg = HybridMeshConfig(
        training=TrainingConfig(gradient_accumulation_steps=3, gc_freq=7)
    )
    assert cfg.gradient_accumulation_steps == 3
    assert cfg.gc_freq == 7
    # The defaults the trainer runs with when nothing is passed.
    assert HybridMeshConfig().gradient_accumulation_steps == 1
    assert HybridMeshConfig().gc_freq == 50


def test_cp_must_divide_seq_len() -> None:
    """A ragged sequence split would give ranks unequal token counts."""
    from hpmesh.trainer import HybridMeshConfig

    with pytest.raises(ValueError):
        HybridMeshConfig(
            parallel=ParallelConfig(context_parallel_size=3),
            training=TrainingConfig(max_seq_len=64),
        )
