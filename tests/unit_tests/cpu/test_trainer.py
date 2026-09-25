"""Training-loop machinery: reductions, the data iterator, and checkpointing.

The reductions and the iterator run without a process group, which is the point
-- the parts of the loop that are easy to get wrong are the ones that do not need
a cluster to exercise. The collectives are checked in their single-rank form
(where the reduction is the identity) and their clip semantics, which is where
the real bug risk lives: clipping is easy to write such that it silently does
nothing.

Checkpointing is exercised through the real ``CheckpointManager``, which runs
single-process as long as no process group is initialized -- so the tests cover
the DCP path the trainer actually uses rather than a substitute. The optimizer
cases are deliberately built on *fresh* objects: restoring into an optimizer that
has already taken a step hides the bug this suite exists to catch, because a
cold Adam has no ``exp_avg`` tensors for DCP to write into.
"""

from __future__ import annotations

import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hpmesh.accelerator.collectives import (
    clip_grad_norm_,
)
from hpmesh.components.checkpointer import (
    DATALOADER,
    TRAIN_STATE,
    CheckpointManager,
)
from hpmesh.components.loss import (
    IGNORE_INDEX,
    cross_entropy_loss,
    next_token_targets,
    vocab_shard_bounds,
)
from hpmesh.components.optimizer import OptimizersContainer
from hpmesh.components.optimizer.lr_scheduler import build_lr_scheduler
from hpmesh.config import (
    CheckpointConfig,
    HybridMeshConfig,
    LRSchedulerConfig,
    OptimizerConfig,
    ParallelConfig,
    ParamGroupConfig,
    TrainingConfig,
)
from hpmesh.datasets.random_data import (
    Batch,
    DataLoaderExhausted,
    RandomTokenDataLoader,
    RandomTokenSource,
    batch_iterator,
)
from hpmesh.models.hf_factory import build_model_config
from hpmesh.models.hf_wrapper import HFTransformerModel
from hpmesh.trainer.trainer import Trainer


def test_pp_forward_backward_releases_consumed_loss_graphs() -> None:
    """The PP schedule's reporting losses must not retain completed graphs."""
    activation_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    loss_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    loss_containers: list[list[torch.Tensor]] = []
    gradients: list[torch.Tensor] = []

    def schedule_step(**kwargs) -> None:
        losses = kwargs["losses"]
        loss_containers.append(losses)
        for value in (1.0, 2.0):
            activation = torch.tensor(value, requires_grad=True)
            # PP losses are normalized before the schedule runs backward. The
            # trainer multiplies their detached sum by the global denominator
            # to recover the raw reporting sum.
            loss = activation.square().view(()) / kwargs["loss_kwargs"][
                "global_valid_tokens"
            ]
            loss.backward()
            assert activation.grad is not None
            gradients.append(activation.grad.detach().clone())
            activation_refs.append(weakref.ref(activation))
            loss_refs.append(weakref.ref(loss))
            losses.append(loss)

    trainer = cast(
        Trainer,
        SimpleNamespace(
            pp_has_first_stage=True,
            pp_has_last_stage=True,
            pp_schedule=SimpleNamespace(step=schedule_step),
            parallel_dims=None,
            _param_context=nullcontext,
            _pp_microbatches=lambda batch: [batch, batch],
            _preprocess=lambda microbatch: (
                torch.ones(1),
                torch.ones(1, dtype=torch.long),
                {},
            ),
        ),
    )

    reporting_loss = Trainer._pp_forward_backward_body(
        trainer,
        Batch(input_ids=torch.ones(1, 1), labels=torch.ones(1, 1)),
        global_valid_tokens=torch.tensor(2),
    )

    torch.testing.assert_close(reporting_loss, torch.tensor(5.0))
    torch.testing.assert_close(torch.stack(gradients), torch.tensor([1.0, 2.0]))
    assert not reporting_loss.requires_grad
    assert reporting_loss.grad_fn is None
    assert loss_containers == [[]]
    assert all(reference() is None for reference in loss_refs)
    assert all(reference() is None for reference in activation_refs)


# -- losses -------------------------------------------------------------------


def test_next_token_targets_shifts_within_each_row() -> None:
    """Row ``r`` predicts its own next token -- never the row after it."""
    labels = torch.arange(6).reshape(2, 3)  # rows [0,1,2] and [3,4,5]

    targets = next_token_targets(labels.reshape(-1), seq_len=3)

    # Row 0 predicts 1,2 and row 1 predicts 4,5; the row-final positions are
    # not predictions at all.
    assert targets.tolist() == [1, 2, IGNORE_INDEX, 4, 5, IGNORE_INDEX]


def test_next_token_targets_never_crosses_a_row_boundary() -> None:
    """The bug this guards: a flat shift would pair row 0's last token with
    row 1's first, predicting a token the model was never given context for."""
    labels = torch.tensor([[10, 11, 12], [20, 21, 22]])

    targets = next_token_targets(labels.reshape(-1), seq_len=3)

    assert 20 not in targets.tolist()  # 19 does not exist, and 20 is row 1's start


def test_cross_entropy_ignores_the_shifted_padding() -> None:
    """Ignored positions contribute neither loss nor denominator."""
    torch.manual_seed(0)
    logits = torch.randn(4, 7)
    targets = torch.tensor([1, 2, IGNORE_INDEX, 6])

    total = cross_entropy_loss(logits, targets)

    # Same number as computing the CE over only the three real targets.
    kept_logits = logits[[0, 1, 3]]
    expected = cross_entropy_loss(kept_logits, torch.tensor([1, 2, 6]))
    torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-6)


def test_cross_entropy_selects_the_local_path_by_shape() -> None:
    """With a TP group but full-vocab logits, the plain path must run.

    Selecting the vocab-parallel path by a flag instead would compute a loss
    over a vocabulary that is not actually sharded -- wrong everywhere, and
    invisible because the shapes still line up.

    The group is a bare sentinel on purpose: the vocab-parallel path is the only
    thing that would touch it, so reaching it turns this test into an error
    rather than a wrong number.
    """
    logits = torch.randn(3, 5)
    targets = torch.tensor([0, 1, 2])

    full = cross_entropy_loss(logits, targets, tp_group=object(), global_vocab_size=5)

    torch.testing.assert_close(
        full, cross_entropy_loss(logits, targets), rtol=1e-5, atol=1e-6
    )


def test_vocab_shard_bounds_are_contiguous_and_cover_the_vocabulary() -> None:
    for vocab_size in (7, 8, 9, 100):
        for tp in (1, 2, 3, 4):
            bounds = [vocab_shard_bounds(vocab_size, tp, r) for r in range(tp)]
            assert bounds[0][0] == 0
            assert bounds[-1][1] == vocab_size
            for (_, end), (start, _) in zip(bounds, bounds[1:], strict=False):
                assert end == start, f"gap at V={vocab_size} tp={tp}"


def test_vocab_shard_bounds_never_exceed_the_vocabulary() -> None:
    """tp > V leaves the tail ranks empty rather than negative."""
    bounds = [vocab_shard_bounds(3, 5, r) for r in range(5)]

    assert all(start <= end for start, end in bounds)
    assert sum(end - start for start, end in bounds) == 3


# -- reductions ---------------------------------------------------------------


def test_the_timeout_applies_to_every_one_dimensional_group_and_the_default(
    monkeypatch,
) -> None:
    """``set_pg_timeouts`` must reach *all* the groups, plus the world group.

    The failure this guards against is a timeout lowered on some groups and not
    others: a hang on an untouched group would still wait out the long startup
    value, which is exactly what the function exists to prevent. The world group
    (``None``) is the easy one to forget, because it is not part of any mesh.
    """
    from datetime import timedelta

    from hpmesh.parallel import collectives

    lowered: list[tuple[timedelta, object]] = []

    class _Mesh:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_group(self):
            return f"group:{self.name}"

    class _Dims:
        def get_all_one_dimensional_meshes(self):
            return {"tp": _Mesh("tp"), "dp_shard": _Mesh("dp_shard")}

    monkeypatch.setattr(collectives.dist, "barrier", lambda **_: None)
    monkeypatch.setattr(
        collectives.dist,
        "set_timeout",
        lambda t, g=None: lowered.append((t, g)),
        raising=False,
    )

    collectives.set_pg_timeouts(timedelta(seconds=7), _Dims())

    assert lowered == [
        (timedelta(seconds=7), "group:tp"),
        (timedelta(seconds=7), "group:dp_shard"),
        (timedelta(seconds=7), None),  # the default (world) group, not a mesh
    ]


def test_timeout_uses_torch_210_compatibility_api(monkeypatch) -> None:
    """Torch 2.10 keeps runtime timeout adjustment in distributed_c10d."""
    from datetime import timedelta

    from hpmesh.parallel import collectives

    lowered = []

    class _Dims:
        def get_all_one_dimensional_meshes(self):
            return {}

    monkeypatch.delattr(collectives.dist, "set_timeout", raising=False)
    monkeypatch.setattr(collectives.dist, "barrier", lambda **_: None)
    monkeypatch.setattr(
        collectives.dist.distributed_c10d,
        "_set_pg_timeout",
        lambda timeout, group=None: lowered.append((timeout, group)),
    )

    collectives.set_pg_timeouts(timedelta(seconds=11), _Dims())

    assert lowered == [(timedelta(seconds=11), None)]


def test_timeout_adjustment_is_skipped_for_hccl(monkeypatch) -> None:
    """HCCL does not implement c10d's runtime timeout adjustment."""
    from datetime import timedelta

    from hpmesh.parallel import collectives

    class _Dims:
        def get_all_one_dimensional_meshes(self):
            raise AssertionError("groups must not be visited for HCCL")

    monkeypatch.setattr(collectives.dist, "barrier", lambda **_: None)
    monkeypatch.setattr(collectives.device_module, "synchronize", lambda *_: None)
    monkeypatch.setattr(
        collectives.dist.distributed_c10d,
        "_set_pg_timeout",
        lambda *_: pytest.fail("HCCL timeout setter must not be called"),
    )

    collectives.set_pg_timeouts(
        timedelta(seconds=11), _Dims(), device=torch.device("npu:0")
    )


def test_a_non_positive_train_timeout_is_rejected() -> None:
    """A zero timeout is a typo for "no limit", and breaks every collective."""
    with pytest.raises(ValueError, match="train_timeout_seconds must be greater"):
        ParallelConfig(train_timeout_seconds=0)
    assert ParallelConfig().train_timeout_seconds == 100


# -- gradient clipping --------------------------------------------------------


def _graded(*zeros: bool) -> list[nn.Parameter]:
    params = []
    for is_zero in zeros:
        p = nn.Parameter(torch.zeros(4))
        p.grad = torch.zeros(4) if is_zero else torch.full((4,), 3.0)
        params.append(p)
    return params


def test_clip_scales_gradients_to_the_threshold() -> None:
    params = _graded(False)  # norm = 3 * 2 = 6
    norm = clip_grad_norm_(params, max_norm=1.0)

    assert float(norm) > 1.0
    # After clipping the concatenated gradient vector has norm exactly max_norm.
    clipped = torch.cat([p.grad.reshape(-1) for p in params])
    torch.testing.assert_close(clipped.norm(), torch.tensor(1.0), rtol=1e-5, atol=1e-6)


def test_non_positive_max_norm_reports_the_norm_without_clipping() -> None:
    """``max_norm<=0`` disables clipping but must still return the real norm.

    This is the mode that proves the refactor is numerically inert: the norm is
    computed, reported, and nothing is scaled.
    """
    params = _graded(False)
    before = [p.grad.clone() for p in params]

    norm = clip_grad_norm_(params, max_norm=-1.0)

    assert float(norm) > 1.0
    assert all(torch.equal(p.grad, b) for p, b in zip(params, before, strict=True))


def test_clip_ignores_parameters_with_no_gradient() -> None:
    """A parameter that received no gradient must not contribute to the norm."""
    params = _graded(False, True)
    norm = clip_grad_norm_(params, max_norm=-1.0)

    torch.testing.assert_close(norm, torch.tensor(6.0), rtol=1e-5, atol=1e-6)


def test_clip_does_not_exhaust_a_generator() -> None:
    """Passing a generator must not silently train on a clipped-only subset."""
    params = _graded(False, False)
    norm = clip_grad_norm_((p for p in params), max_norm=1.0)

    assert float(norm) > 1.0
    assert all(p.grad is not None for p in params)


# -- the data iterator --------------------------------------------------------


def _cfg_with_batch(**overrides) -> HybridMeshConfig:
    fields = {"global_batch_size": 8, "max_seq_len": 16, "seed": 42, **overrides}
    return HybridMeshConfig(training=TrainingConfig(**fields))


def _bare_trainer(cfg: HybridMeshConfig) -> Trainer:
    """A Trainer with ``__init__`` bypassed, for testing pure data helpers."""
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    return trainer


def test_batch_size_per_rank_divides_evenly() -> None:
    trainer = _bare_trainer(_cfg_with_batch(global_batch_size=8))

    # 8 over 2 ranks is 4 each; 8 over 8 is 1 each.
    assert trainer._batch_size_per_rank(2) == 4
    assert trainer._batch_size_per_rank(8) == 1


def test_an_indivisible_global_batch_is_rejected() -> None:
    """A floor would silently train a smaller global batch than the config names.

    Every number derived from it -- the lr, the token count, the value logged as
    ``batch_size`` -- would then describe a batch that is not the one being read.
    The random loader happens to reject this too, but only after it is built and
    only on that one path; the check has to sit where both paths compute it.
    """
    trainer = _bare_trainer(_cfg_with_batch(global_batch_size=10))

    with pytest.raises(ValueError, match="divisible by"):
        trainer._batch_size_per_rank(4)


def _source(n: int, *, batch_size: int = 2, seq_len: int = 4) -> RandomTokenSource:
    return RandomTokenSource(
        seed=0, vocab_size=16, batch_size=batch_size, seq_len=seq_len
    )


def test_iterator_is_deterministic_from_seed_and_step() -> None:
    """Batch ``n`` depends only on ``(seed, n)`` -- never on how many came before.

    Two independent iterators must therefore agree element for element. That is
    the property a resumed run and a fresh DP comparison both rely on.
    """
    it1, it2 = batch_iterator(_source(0)), batch_iterator(_source(0))

    for _ in range(3):
        assert torch.equal(next(it1).input_ids, next(it2).input_ids)


def test_different_steps_produce_different_batches() -> None:
    it = batch_iterator(_source(0))
    first, second = next(it).input_ids, next(it).input_ids

    assert not torch.equal(first, second)


def test_iterator_restarts_a_finite_source() -> None:
    """Exhausting the source restarts it rather than stopping the loop."""
    finite = [Batch(torch.zeros(2, 4), torch.zeros(2, 4))]  # exactly one batch

    it = batch_iterator(finite)
    for _ in range(5):
        batch = next(it)
        assert batch.input_ids.shape == (2, 4)


def test_iterator_rejects_an_empty_source() -> None:
    """An empty source must raise, not spin forever in the restart loop."""
    it = batch_iterator([])
    try:
        next(it)
    except DataLoaderExhausted:
        return
    raise AssertionError("an empty source should raise DataLoaderExhausted")


# -- checkpointing ------------------------------------------------------------


def _model_and_optimizer() -> tuple[nn.Module, OptimizersContainer]:
    """A model and the container the trainer would build around it.

    The checkpointer is handed the container, not a bare ``AdamW``: its state
    dict is flat and FQN-keyed, which is what makes optimizer state
    unambiguous under pipeline parallelism, and it materializes a fresh
    optimizer's state before DCP plans a load.
    """
    model = nn.Linear(4, 4)
    optimizer = OptimizersContainer(
        OptimizerConfig(
            learning_rate=0.1,
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.1},
                )
            ],
        ),
        model_parts=[model],
    )
    return model, optimizer


class _TrainState:
    """Stand-in for the two counters the Trainer contributes to a checkpoint."""

    def __init__(self) -> None:
        self.step = 0
        self.ntokens_seen = 0

    def state_dict(self) -> dict[str, int]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]


def _lr_scheduler(optimizer, *, warmup_steps: int = 0, training_steps: int = 8):
    """The schedule the trainer builds, over the same inner optimizers."""
    return build_lr_scheduler(
        LRSchedulerConfig(warmup_steps=warmup_steps),
        optimizers=list(optimizer),
        training_steps=training_steps,
    )


def _manager(
    folder: str, model: nn.Module, optimizer, state: _TrainState, **overrides
) -> CheckpointManager:
    # keep_latest_k=0 keeps the default runs unbounded, so a test that asserts on
    # what is on disk is describing the save path rather than the purge thread.
    # Tests about retention pass keep_latest_k explicitly.
    config = {"keep_latest_k": 0, **overrides}
    return CheckpointManager(
        CheckpointConfig(enable=True, folder="checkpoint", **config),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_lr_scheduler(optimizer),
        states={TRAIN_STATE: state},
        folder=folder,
    )


def _step(model: nn.Module, optimizer: OptimizersContainer, times: int = 2) -> None:
    for _ in range(times):
        optimizer.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()


def _optimizer_state(optimizer: OptimizersContainer) -> dict:
    """One step's optimizer state, keyed by FQN, from the container's flat dict.

    The trainable parameters are all ``weight``/``bias``, each carrying
    ``exp_avg`` and ``exp_avg_sq``.
    """
    return {
        key: value.clone()
        for key, value in optimizer.state_dict().items()
        if key.startswith("state.") and key.endswith(("exp_avg", "exp_avg_sq"))
    }


def test_checkpoint_round_trips_model_optimizer_and_counters(tmp_path) -> None:
    """The three things a resume needs, restored into brand-new objects."""
    model, optimizer = _model_and_optimizer()
    state = _TrainState()
    state.step, state.ntokens_seen = 7, 99
    _step(model, optimizer)

    saved_weights = model.weight.detach().clone()
    saved_optim = _optimizer_state(optimizer)

    manager = _manager(str(tmp_path), model, optimizer, state, interval=1)
    assert manager.save(7) is True
    manager.close()

    torch.manual_seed(0)
    fresh_model, fresh_optimizer = _model_and_optimizer()
    with torch.no_grad():
        fresh_model.weight.zero_()
    fresh_state = _TrainState()

    resumed = _manager(str(tmp_path), fresh_model, fresh_optimizer, fresh_state)
    assert resumed.load(-1) is True
    resumed.close()

    assert (fresh_state.step, fresh_state.ntokens_seen) == (7, 99)
    assert torch.equal(fresh_model.weight.detach(), saved_weights)

    # The load-bearing case. A fresh Adam has no exp_avg to write into, so a
    # manager that did not materialize the state first would report success
    # while leaving the optimizer cold.
    restored_optim = _optimizer_state(fresh_optimizer)
    assert restored_optim.keys() == saved_optim.keys()
    for key, value in saved_optim.items():
        assert torch.equal(restored_optim[key], value)


def test_checkpoint_round_trips_the_lr_schedule(tmp_path) -> None:
    """A resumed run continues the lr curve instead of restarting it.

    The optimizer restores ``base_lrs``, so the *current* lr comes back right
    whether or not the schedule was checkpointed -- which is what let the bug
    hide. ``last_epoch`` is the scheduler's own counter, and a fresh scheduler
    starts it at 0. Restoring it is what puts the next step's lr on the curve.
    """
    model, optimizer = _model_and_optimizer()
    state = _TrainState()
    schedule = _lr_scheduler(optimizer, warmup_steps=8)
    _step(model, optimizer)
    schedule.step()
    schedule.step()
    assert schedule.state_dict() == {"last_epoch": 2}

    manager = CheckpointManager(
        CheckpointConfig(enable=True, folder="checkpoint", keep_latest_k=0, interval=1),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=schedule,
        states={TRAIN_STATE: state},
        folder=str(tmp_path),
    )
    assert manager.save(2) is True
    manager.close()

    # A brand-new trainer's objects, as a resume builds them. Both counters are
    # fresh, so a resume that restored neither could not be told from this one.
    fresh_model, fresh_optimizer = _model_and_optimizer()
    fresh_state = _TrainState()
    fresh_schedule = _lr_scheduler(fresh_optimizer, warmup_steps=8)

    resumed = CheckpointManager(
        CheckpointConfig(enable=True, folder="checkpoint", keep_latest_k=0),
        model_parts=[fresh_model],
        optimizer=fresh_optimizer,
        lr_scheduler=fresh_schedule,
        states={TRAIN_STATE: fresh_state},
        folder=str(tmp_path),
    )
    assert resumed.load(-1) is True
    resumed.close()

    assert fresh_schedule.state_dict() == {"last_epoch": 2}
    # The property that matters: the next step's lr is the one an uninterrupted
    # run would have used. The optimizer restores ``base_lrs`` (0.1), and at
    # last_epoch 3 the 8-step linear warmup gives a factor of 4/8 -- 0.05. With
    # the schedule's state lost the factor would be 1/8, i.e. 0.0125: the curve
    # restarted from the beginning.
    fresh_schedule.step()
    assert fresh_schedule.get_metrics() == {"lr/AdamW": pytest.approx(0.05)}


def test_the_schedule_is_not_restored_from_a_missing_checkpoint(tmp_path) -> None:
    """A fresh run's schedule must start at zero, not at whatever it last was."""
    _, optimizer = _model_and_optimizer()
    schedule = _lr_scheduler(optimizer, warmup_steps=8)
    schedule.step()
    assert schedule.state_dict() == {"last_epoch": 1}

    # No checkpoint on disk: nothing to restore, and nothing must be invented.
    fresh_model, fresh_optimizer = _model_and_optimizer()
    fresh_schedule = _lr_scheduler(fresh_optimizer, warmup_steps=8)
    manager = CheckpointManager(
        CheckpointConfig(enable=True, folder="checkpoint", keep_latest_k=0),
        model_parts=[fresh_model],
        optimizer=fresh_optimizer,
        lr_scheduler=fresh_schedule,
        states={TRAIN_STATE: _TrainState()},
        folder=str(tmp_path),
    )
    assert manager.load(-1) is False
    manager.close()

    assert fresh_schedule.state_dict() == {"last_epoch": 0}


def test_excluding_both_optimizer_and_schedule_leaves_them_untouched(tmp_path) -> None:
    """The pairing ``config.py`` demands must actually be loadable.

    The config rejects excluding the optimizer without the schedule, on the
    grounds that a restored ``last_epoch`` would be applied against cold
    ``base_lrs``. That rule is only enforceable if both keys exist in the
    manager's states: before the schedule was registered, asking to exclude it
    raised "lr_scheduler not found in state_dict", so the pairing the config
    mandates could not be expressed at all.
    """
    model, optimizer = _model_and_optimizer()
    schedule = _lr_scheduler(optimizer, warmup_steps=8)
    _step(model, optimizer)
    schedule.step()

    manager = CheckpointManager(
        CheckpointConfig(
            enable=True,
            folder="checkpoint",
            keep_latest_k=0,
            interval=1,
            exclude_from_loading=["optimizer", "lr_scheduler"],
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=schedule,
        states={TRAIN_STATE: _TrainState()},
        folder=str(tmp_path),
    )
    assert manager.save(1) is True
    manager.close()

    fresh_model, fresh_optimizer = _model_and_optimizer()
    fresh_schedule = _lr_scheduler(fresh_optimizer, warmup_steps=8)
    resumed = CheckpointManager(
        CheckpointConfig(
            enable=True,
            folder="checkpoint",
            keep_latest_k=0,
            exclude_from_loading=["optimizer", "lr_scheduler"],
        ),
        model_parts=[fresh_model],
        optimizer=fresh_optimizer,
        lr_scheduler=fresh_schedule,
        states={TRAIN_STATE: _TrainState()},
        folder=str(tmp_path),
    )
    # Reaching here at all is the point: the exclusion is a legal request.
    assert resumed.load(-1) is True
    resumed.close()

    # Excluded means excluded -- the schedule is not quietly advanced to match.
    assert fresh_schedule.state_dict() == {"last_epoch": 0}


def test_checkpoint_save_includes_optimizer_state_without_a_prior_step(
    tmp_path,
) -> None:
    """A save at step 1 must carry optimizer state, not just the weights.

    The first checkpoint of a run is taken by an optimizer that has just stepped
    for the first time -- but any save before that would have written a model-only
    file, which restores cleanly and silently resumes from a cold optimizer.
    """
    folder = tmp_path / "checkpoint"
    model, optimizer = _model_and_optimizer()
    state = _TrainState()

    manager = _manager(str(tmp_path), model, optimizer, state, interval=1)
    manager.save(1)
    manager.close()

    assert (folder / "step-1" / ".metadata").is_file()

    fresh_model, fresh_optimizer = _model_and_optimizer()
    fresh_state = _TrainState()
    resumed = _manager(str(tmp_path), fresh_model, fresh_optimizer, fresh_state)
    resumed.load(-1)
    resumed.close()

    assert _optimizer_state(fresh_optimizer) != {}


def test_load_returns_false_when_there_is_no_checkpoint(tmp_path) -> None:
    """A first run is not an error: ``False`` lets resume be the same code path."""
    model, optimizer = _model_and_optimizer()
    manager = _manager(str(tmp_path), model, optimizer, _TrainState())
    assert manager.load(-1) is False
    manager.close()


def test_checkpoint_is_sharded_over_steps_not_over_ranks(tmp_path) -> None:
    """Every rank writes into one shared step directory; DCP records the layout.

    The old per-rank files (``rank00.pt``) could not express a sharded tensor:
    they worked only while one rank owned the whole parameter.
    """
    model, optimizer = _model_and_optimizer()
    manager = _manager(str(tmp_path), model, optimizer, _TrainState(), interval=1)

    manager.save(1)
    manager.save(3)
    manager.close()

    steps = sorted(p.name for p in (tmp_path / "checkpoint").iterdir())
    assert steps == ["step-1", "step-3"]
    # ``.metadata`` is what marks a step directory resumable; it is written per
    # step, not per rank, so its presence proves nothing collided.
    assert (tmp_path / "checkpoint" / "step-3" / ".metadata").is_file()


def test_retention_keeps_the_latest_k_and_deletes_the_rest(tmp_path) -> None:
    model, optimizer = _model_and_optimizer()
    manager = _manager(
        str(tmp_path), model, optimizer, _TrainState(), interval=1, keep_latest_k=2
    )

    for step in (1, 2, 3, 4):
        manager.save(step)
    manager.close()

    remaining = sorted(p.name for p in (tmp_path / "checkpoint").iterdir())
    # keep_latest_k counts the checkpoint the next save is about to take, so 2
    # retained slots leave the two most recent on disk.
    assert remaining == ["step-3", "step-4"]


def test_step_discovery_ignores_unparseable_directory_names(tmp_path) -> None:
    """A stray name must not be parsed into a step the loader would then pick."""
    model, optimizer = _model_and_optimizer()
    manager = _manager(str(tmp_path), model, optimizer, _TrainState(), interval=1)
    manager.save(4)
    manager.close()

    checkpoint_folder = tmp_path / "checkpoint"
    (checkpoint_folder / "step-007").mkdir()
    (checkpoint_folder / "notes").mkdir()

    fresh_model, fresh_optimizer = _model_and_optimizer()
    resumed = _manager(str(tmp_path), fresh_model, fresh_optimizer, _TrainState())
    assert resumed.load(-1) is True
    resumed.close()


# -- the dataloader seam ------------------------------------------------------
#
# The trainer drives whichever ``BaseDataLoader`` the config names, and the two
# implementations disagree about what a batch is: the synthetic one yields
# ``(B, T)`` rows of one document each, the Grain one a flat packed stream. The
# trainer reconciles the *count* and asks the model to reconcile the *shape*:
# the denominator is a whole-batch property that has to be reduced across DP
# before the first backward, so it cannot come from a per-micro-batch model
# call. The tests below pin both halves of that split.


def _random_batch(batch_size: int = 4, seq_len: int = 6) -> Batch:
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 32, (batch_size, seq_len), generator=generator)
    return Batch(input_ids=ids, labels=ids.clone())


def _wrapper() -> HFTransformerModel:
    """The smallest real wrapper: the shape/shift path reads no model weights."""
    config = build_model_config(
        "qwen3",
        seq_len=8,
        arch_overrides={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )
    return HFTransformerModel(config).eval()


def test_count_valid_tokens_passes_the_synthetic_shape_through_unchanged() -> None:
    batch = _random_batch()
    num_valid = Trainer._count_valid_tokens(batch)
    # The synthetic source has no collator, so the trainer counts the
    # predictable labels itself. It must: the loss divides by the step's token
    # count *before* it backwards, so a ``None`` here would leave the
    # denominator unknown at the one point it is still needed.
    assert num_valid == int(
        (next_token_targets(batch.labels.reshape(-1), seq_len=6) != IGNORE_INDEX).sum()
    )


def test_count_valid_tokens_prefers_the_collators_count() -> None:
    """The count is the loss denominator, so a missing one is recomputed.

    Nothing in the current collators omits it; the fallback exists so that a
    loader written against the ``BaseDataLoader`` contract alone still gets a
    correct denominator rather than a zero divide on the first backward.
    """
    labelled = {
        "input": torch.arange(5),
        "labels": torch.tensor([1, 2, IGNORE_INDEX, 4, IGNORE_INDEX]),
    }
    assert Trainer._count_valid_tokens(labelled) == 3

    counted = {**labelled, "num_valid_tokens": 5}
    assert Trainer._count_valid_tokens(counted) == 5


def test_count_valid_tokens_reports_the_row_final_mask_the_loss_skips() -> None:
    """The two halves of "the count is not the loss's business".

    A synthetic batch loses one prediction per row to the shift, so the count
    the trainer reports and the labels ``_loss_sum`` scores disagree by exactly
    that many positions -- and the count is the smaller, correct one.
    """
    batch = _random_batch(batch_size=3, seq_len=4)

    num_valid = Trainer._count_valid_tokens(batch)

    assert num_valid == batch.labels.numel() - batch.labels.shape[0]


def test_preprocess_inputs_shifts_the_synthetic_labels_within_a_row() -> None:
    """Row ``r`` must never predict row ``r + 1``'s first token.

    The synthetic source hands over labels equal to its inputs; the shift is
    the model's now. Doing it globally would pair each row's last position with
    the next document's first token -- a target the model had no context for.
    """
    batch = _random_batch(batch_size=3, seq_len=4)
    inputs, labels, extra_kwargs = _wrapper().preprocess_inputs(
        batch, parallel_dims=None
    )

    assert torch.equal(inputs, batch.labels.reshape(-1))
    assert torch.equal(labels, next_token_targets(batch.labels.reshape(-1), seq_len=4))
    # Every row-final position is excluded, one per row.
    assert int((labels == IGNORE_INDEX).sum()) == 3
    # And the surviving pairs are the intra-row ones.
    assert torch.equal(labels.reshape(-1, 4)[:, :3], batch.labels[:, 1:])
    # The synthetic source carries no positions, so the forward's own arange
    # default applies -- right for one document, and not for a packed one.
    assert extra_kwargs == {}


def test_preprocess_inputs_reads_the_grain_batch_without_consuming_it() -> None:
    """The collator already shifted and masked the labels; the shift is a read.

    ``num_valid_tokens`` never reaches the model -- the trainer pops it first,
    because the denominator is the whole-batch count and ``positions`` is the
    one entry the forward takes.
    """
    grain_batch = {
        "input": torch.arange(8),
        "labels": torch.tensor([1, 2, IGNORE_INDEX, 4, IGNORE_INDEX, 6, 7, 8]),
        "positions": torch.arange(8),
    }

    inputs, labels, extra_kwargs = _wrapper().preprocess_inputs(
        grain_batch, parallel_dims=None
    )

    assert torch.equal(inputs, torch.arange(8))
    assert torch.equal(labels, grain_batch["labels"])
    assert torch.equal(extra_kwargs["positions"], torch.arange(8))
    # Nothing was consumed: the loader owns the batch, and a second pass over
    # it must see the same dict.
    assert set(grain_batch) == {"input", "labels", "positions"}


def test_preprocess_inputs_drops_a_padding_mask_the_forward_would_reject() -> None:
    """Anything left in the dict is splatted into ``forward`` as a kwarg.

    The wrapper takes exactly two of them, so an unclaimed ``padding_mask``
    would reach the decoder as ``padding_mask=...`` and raise. Dropping it here
    is the whole reason the return value is a dict rather than the batch.
    """
    grain_batch = {
        "input": torch.arange(4),
        "labels": torch.arange(4),
        "positions": torch.arange(4),
        "padding_mask": torch.zeros(4, dtype=torch.bool),
    }

    _, _, extra_kwargs = _wrapper().preprocess_inputs(grain_batch, parallel_dims=None)

    assert set(extra_kwargs) == {"positions"}


def test_loss_sum_scores_every_label_ignored_ones_included() -> None:
    """The shift, not the loss, is what removes positions from the denominator.

    ``_loss_sum`` returns the raw summed CE; the count that normalizes it is
    taken upstream, from the *unsharded* batch, because context parallelism
    slices ``labels`` after the fact and a recount here would undercount by
    ``cp``. So the loss itself only skips the ignored rows.
    """
    logits = torch.randn(6, 8)
    labels = torch.tensor([1, 2, IGNORE_INDEX, 4, IGNORE_INDEX, IGNORE_INDEX])

    loss_sum = Trainer._loss_sum(logits, labels)

    expected = F.cross_entropy(logits.float(), labels, reduction="sum")
    torch.testing.assert_close(loss_sum, expected, rtol=1e-5, atol=1e-8)


def test_loss_sum_makes_one_prediction_per_predictable_label() -> None:
    """``logits[t]`` scores ``labels[t]``: the two are already aligned."""
    logits = torch.randn(4, 8)
    labels = torch.tensor([1, 2, 3, 4])
    loss_sum = Trainer._loss_sum(logits, labels)

    expected = F.cross_entropy(logits.float(), labels, reduction="sum")
    torch.testing.assert_close(loss_sum, expected, rtol=1e-5, atol=1e-8)


def test_checkpoint_carries_a_dataloader_read_position(tmp_path) -> None:
    """Resuming a real corpus must resume the *data*, not just the weights.

    Without this the run would restore trained weights and then re-read the
    corpus from the beginning, silently training a second pass over the start
    of the data while the step counter said otherwise.
    """
    model, optimizer = _model_and_optimizer()
    loader = RandomTokenDataLoader(
        seed=3, vocab_size=16, batch_size=4, seq_len=6, dp_rank=0, dp_world_size=1
    )

    manager = CheckpointManager(
        CheckpointConfig(enable=True, folder="checkpoint", keep_latest_k=0, interval=1),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_lr_scheduler(optimizer),
        states={TRAIN_STATE: _TrainState(), DATALOADER: loader},
        folder=str(tmp_path),
    )
    for _ in range(3):
        next(iter(loader))
    assert manager.save(3)
    manager.close()

    # A fresh loader restored from the checkpoint must continue where the old
    # one stopped rather than restart.
    resumed_loader = RandomTokenDataLoader(
        seed=3, vocab_size=16, batch_size=4, seq_len=6, dp_rank=0, dp_world_size=1
    )
    fresh_model, fresh_optimizer = _model_and_optimizer()
    resumed = CheckpointManager(
        CheckpointConfig(enable=True, folder="checkpoint", keep_latest_k=0, interval=1),
        model_parts=[fresh_model],
        optimizer=fresh_optimizer,
        lr_scheduler=_lr_scheduler(fresh_optimizer),
        states={TRAIN_STATE: _TrainState(), DATALOADER: resumed_loader},
        folder=str(tmp_path),
    )
    assert resumed.load(-1) is True
    resumed.close()

    # Where the original would have gone next.
    reference = RandomTokenDataLoader(
        seed=3, vocab_size=16, batch_size=4, seq_len=6, dp_rank=0, dp_world_size=1
    )
    for _ in range(3):
        next(iter(reference))
    expected = next(iter(reference))

    got = next(iter(resumed_loader))
    assert torch.equal(got.input_ids, expected.input_ids)
    assert torch.equal(got.labels, expected.labels)


# -- the synthetic data iterator ------------------------------------------------


def test_synthetic_loader_is_checkpointed_like_any_other() -> None:
    """``_build_dataloader`` hands back the synthetic loader, not ``None``.

    Dropping it (the old behavior) kept the loader out of the checkpoint's
    ``states``, so a resumed run restored trained weights and then re-read the
    corpus from batch 0. The loader's ``load_state_dict`` replay is what a
    resume needs, and it is dead code unless the loader is registered.
    """
    trainer = _bare_trainer(_cfg_with_batch(global_batch_size=8, max_seq_len=16))

    loader = trainer._build_dataloader()

    assert isinstance(loader, RandomTokenDataLoader)


def test_microbatch_defers_the_device_transfer_to_consumption() -> None:
    """Reading an accumulation window must not move it to the device.

    ``_microbatch`` runs at read time for every group of the window; moving
    tensors there would keep the whole window resident in device memory. The
    transfer belongs to ``_to_device``, which ``_preprocess`` calls once per
    group, just ahead of that group's forward.
    """
    trainer = _bare_trainer(_cfg_with_batch())
    trainer.device = torch.device("cpu")
    trainer.ntokens_seen = 0
    batch = _random_batch()

    calls = 0
    original_to = torch.Tensor.to

    def counting_to(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_to(self, *args, **kwargs)

    try:
        torch.Tensor.to = counting_to
        microbatch = trainer._microbatch(batch)
        assert calls == 0, "_microbatch moved tensors at read time"
        trainer._to_device(microbatch["batch"])
        assert calls == 2  # input_ids and labels
    finally:
        torch.Tensor.to = original_to

    assert microbatch["num_valid_tokens"] == batch.labels.numel() - 4


def test_microbatch_counts_only_the_ranks_share_under_sequence_sharding() -> None:
    """The cumulative token count must survive the loss-mesh sum unchanged.

    Every rank of a CP/TP group reads the *same* batch from the loader, and the
    sequence is cut across the group in ``preprocess_inputs``. ``train_step``
    sums ``ntokens_seen`` over the dp*cp*tp loss mesh, so each rank must
    contribute only its ``1 / (cp * tp)`` share -- counting the whole batch
    would report a corpus cp * tp times its true size.
    """
    trainer = _bare_trainer(_cfg_with_batch())
    trainer.ntokens_seen = 0
    trainer.parallel_dims = SimpleNamespace(cp=2, tp=2)

    batch = _random_batch()
    trainer._microbatch(batch)

    assert trainer.ntokens_seen == batch.labels.numel() // 4
    # Non-vacuity: the share differs from both the full count and zero.
    assert 0 < trainer.ntokens_seen < batch.labels.numel()


def test_the_synthetic_loader_reports_an_absolute_position_after_a_resume() -> None:
    """A checkpoint written *after* a resume must not lose the seek.

    The loader cannot seek, so resuming replays. If the saved position were the
    count of batches ``__iter__`` handed out, that count would start at zero on
    the resumed loader, and the batch right after the resume would save as
    batch 0 instead of its true index. The *next* resume would then rewind to
    it and re-train those batches -- a silent replay of already-seen data.
    """

    def loader() -> RandomTokenDataLoader:
        return RandomTokenDataLoader(
            seed=3, vocab_size=64, batch_size=1, seq_len=4, dp_rank=0, dp_world_size=1
        )

    def batch_id(batch) -> tuple[int, ...]:
        return tuple(batch.input_ids.flatten().tolist())

    reference = iter(loader())
    stream = [batch_id(next(reference)) for _ in range(12)]

    def consume(loader, count, offset):
        iterator = iter(loader)
        for index in range(count):
            assert batch_id(next(iterator)) == stream[offset + index]
        return offset + count

    position = 0
    run = loader()
    position = consume(run, 5, position)
    first = run.state_dict()
    assert first["steps"] == 5

    resumed = loader()
    resumed.load_state_dict(first)
    assert resumed.state_dict()["steps"] == 5
    position = consume(resumed, 3, position)

    second = resumed.state_dict()
    # The bug this guards: the pre-fix loader saved 0 here, not 8.
    assert second["steps"] == 8

    # And the checkpoint after the second save resumes at the right place --
    # this is the assertion that fails when the position is relative.
    final = loader()
    final.load_state_dict(second)
    position = consume(final, 4, position)
    assert position == 12


def test_the_synthetic_loader_still_refuses_to_rewind() -> None:
    """The absolute position must not turn a backwards resume into an overshoot.

    ``state_dict`` now carries an absolute index, so the seek drains the
    difference. A checkpoint from *behind* the live loader stays an error --
    replaying forward to a past batch is not something this loader can do.
    """
    loader = RandomTokenDataLoader(
        seed=3, vocab_size=64, batch_size=1, seq_len=4, dp_rank=0, dp_world_size=1
    )
    for _ in range(6):
        next(iter(loader))
    with pytest.raises(ValueError, match="already yielded 6"):
        loader.load_state_dict({"dp_world_size": 1, "steps": 5})


def test_exclude_from_loading_accepts_the_dataloader_key(tmp_path) -> None:
    """``exclude_from_loading=["dataloader"]`` must not raise for its absence.

    Every loader -- the synthetic one included -- is registered in ``states``
    now, so the key always exists; excluding it just skips the restore.
    """
    model, optimizer = _model_and_optimizer()
    loader = RandomTokenDataLoader(
        seed=3, vocab_size=16, batch_size=4, seq_len=6, dp_rank=0, dp_world_size=1
    )
    for _ in range(3):
        next(iter(loader))

    manager = CheckpointManager(
        CheckpointConfig(
            enable=True,
            folder="checkpoint",
            keep_latest_k=0,
            interval=1,
            exclude_from_loading=["dataloader"],
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_lr_scheduler(optimizer),
        states={TRAIN_STATE: _TrainState(), DATALOADER: loader},
        folder=str(tmp_path),
    )
    assert manager.save(3)
    manager.close()

    fresh_model, fresh_optimizer = _model_and_optimizer()
    fresh_loader = RandomTokenDataLoader(
        seed=3, vocab_size=16, batch_size=4, seq_len=6, dp_rank=0, dp_world_size=1
    )
    resumed = CheckpointManager(
        CheckpointConfig(
            enable=True,
            folder="checkpoint",
            keep_latest_k=0,
            interval=1,
            exclude_from_loading=["dataloader"],
        ),
        model_parts=[fresh_model],
        optimizer=fresh_optimizer,
        lr_scheduler=_lr_scheduler(fresh_optimizer),
        states={TRAIN_STATE: _TrainState(), DATALOADER: fresh_loader},
        folder=str(tmp_path),
    )
    assert resumed.load(-1) is True
    resumed.close()

    # Excluded: the fresh loader still sits at batch 0.
    first = next(iter(fresh_loader))
    reference = next(
        iter(RandomTokenDataLoader(seed=3, vocab_size=16, batch_size=4, seq_len=6))
    )
    assert torch.equal(first.input_ids, reference.input_ids)


def test_synthetic_batch_is_deterministic() -> None:
    """Two independent iterators over one config must agree.

    That is what makes two runs comparable and what makes every DP rank see the
    same global batch without a broadcast.
    """
    trainer = _bare_trainer(
        _cfg_with_batch(global_batch_size=8, max_seq_len=16, seed=42)
    )
    first = next(trainer._data_iterator())
    second = next(trainer._data_iterator())

    assert torch.equal(first.input_ids, second.input_ids)
    assert first.input_ids.shape == (8, 16)
    assert torch.equal(first.labels, first.input_ids)


def test_dp_slice_partitions_global_batch() -> None:
    """Two DP ranks' slices must concatenate back to the global batch.

    The slice math is driven directly (no process group): a gap or overlap here
    would drop or duplicate samples once per step, invisibly.
    """
    cfg = _cfg_with_batch(global_batch_size=8, max_seq_len=16, seed=42)
    batch = next(_bare_trainer(cfg)._data_iterator())

    per_rank = cfg.global_batch_size // 2
    rank_0 = batch.input_ids[0:per_rank]
    rank_1 = batch.input_ids[per_rank : 2 * per_rank]

    assert torch.equal(torch.cat([rank_0, rank_1]), batch.input_ids)


# -- the console entry point ----------------------------------------------------


def test_create_seed_checkpoint_writes_a_step_0_checkpoint_and_exits(tmp_path) -> None:
    """``--create_seed_checkpoint`` saves the untrained model at step 0.

    The flag existed in the config with no reader -- a run that set it simply
    trained. Wired up (torchtitan ``train.py``'s semantics), it must write the
    seed checkpoint and exit WITHOUT training: no step-5 checkpoint may appear.
    """
    import os
    import subprocess
    import sys

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    env = {**os.environ, "PYTHONPATH": repo_root}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hpmesh",
            "--steps",
            "5",
            "--max_seq_len",
            "32",
            "--global_batch_size",
            "4",
            "--enable",
            "--create_seed_checkpoint",
            "--dump_folder",
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Created seed checkpoint" in result.stdout
    assert (tmp_path / "checkpoint" / "step-0" / ".metadata").is_file()
    assert not (tmp_path / "checkpoint" / "step-5").exists()
