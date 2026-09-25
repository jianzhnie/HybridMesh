"""The combination matrix: every row's verdict, type, and config agreement.

The load-bearing property is the last one: a combination and the config's own
validation must never disagree -- the matrix is the single source both read.
"""

import pytest

from hpmesh.config import ParallelConfig
from hpmesh.errors import (
    ConfigError,
    EnvironmentUnsupportedError,
    UnsupportedCombinationError,
)
from hpmesh.parallel import matrix

# -- registry shape ------------------------------------------------------------


def test_every_row_has_a_verdict_reason_and_guard() -> None:
    for row in matrix.ENTRIES:
        assert row.name == row.fn.__name__
        assert row.reason == row.fn.__doc__.strip()
        assert row.phase in ("config", "assembly", "probe")
        assert issubclass(row.error, Exception)
        assert row.reason and row.guard


def test_every_guard_function_has_a_row() -> None:
    # The table and the functions cannot drift: every row references a real
    # function in this module, and every public guard function is tabled.
    import inspect

    import hpmesh.parallel.matrix as m

    public = {
        n
        for n, v in vars(m).items()
        if inspect.isfunction(v) and v.__module__ == m.__name__
        and not n.startswith("_")
        and n not in ("check_config", "check_training", "check_root")
    }
    assert {r.name for r in matrix.ENTRIES} == public


def test_config_rows_are_exactly_the_config_scope() -> None:
    config_rows = {r.name for r in matrix.ENTRIES if r.phase == "config"}
    assert config_rows == {
        "sequence_parallel_required",
        "tp_ep_cp",
        "deepep_hybridep",
        "dispatcher_requires_ep",
        "ptrr_load_balancer",
        "ulysses_no_load_balancer",
        "region_ac",
        "memory_budget_requires_compile",
        "cp_divides_seq_len",
        "async_tp_requires_compile",
        "async_tp_requires_tp",
    }


# -- config-phase rows: verdict on both sides ----------------------------------


def _cfg(**overrides) -> ParallelConfig:
    base = dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        expert_parallel_size=1,
        data_parallel_replicate_size=1,
        data_parallel_shard_size=1,
        enable_sequence_parallel=True,
        ep_token_dispatcher="alltoall",
        context_parallel_load_balancer="headtail",
        context_parallel_strategy="kv_allgather",
    )
    base.update(overrides)
    cfg = ParallelConfig.__new__(ParallelConfig)
    for key, value in base.items():
        setattr(cfg, key, value)
    return cfg


def test_supported_config_passes_every_row() -> None:
    matrix.check_config(_cfg())


@pytest.mark.parametrize(
    "check, overrides, error, match",
    [
        (matrix.sequence_parallel_required, {"enable_sequence_parallel": False},
         UnsupportedCombinationError, "sequence-parallel by construction"),
        (matrix.tp_ep_cp,
         {"tensor_parallel_size": 2, "expert_parallel_size": 2,
          "context_parallel_size": 2},
         UnsupportedCombinationError, "tp x ep x cp"),
        (matrix.deepep_hybridep,
         {"ep_token_dispatcher": "deepep", "expert_parallel_size": 2},
         EnvironmentUnsupportedError, "registered gap"),
        (matrix.dispatcher_requires_ep, {"ep_token_dispatcher": "torchao"},
         UnsupportedCombinationError, "expert_parallel_size=1"),
        (matrix.ptrr_load_balancer, {"context_parallel_load_balancer": "ptrr"},
         UnsupportedCombinationError, "ptrr"),
        (matrix.ulysses_no_load_balancer,
         {"context_parallel_strategy": "ulysses"},
         UnsupportedCombinationError, "load_balancer=None"),
    ],
)
def test_config_row_rejects_and_accepts(check, overrides, error, match) -> None:
    with pytest.raises(error, match=match):
        check(_cfg(**overrides))
    check(_cfg())  # the supported side does not raise


# -- assembly/probe rows: type and message -------------------------------------


def test_assembly_rows_reject_with_their_entry_type() -> None:
    class _Block:
        pass

    cases = [
        (matrix.pp_activation_checkpoint, (), "pp > 1 path"),
        (matrix.pp_validation, (), "pipeline parallelism"),
        (matrix.validation_once_requires_dp1, (4,), "data-parallel"),
        (matrix.validation_once_requires_finite_corpus, (), "infinite synthetic"),
        (matrix.ep_checkpoint, (2,), "checkpointing"),
        (matrix.chunked_loss_pp, (2, 2), "chunked_loss_num_chunks=2"),
        (matrix.pp_cp_ep, (), "does not compose"),
        (matrix.pp_real_corpus, (), "synthetic 'random' corpus"),
        (matrix.pp_weight_tying, (), "tied word embeddings"),
        (matrix.shared_expert_tp, ("layers.0.mlp", _Block()), "shared expert"),
        (matrix.tp_moe_specs_without_block, (2, _Block()), "no HF MoE block"),
        (matrix.tp_moe_non_tensor_output, (_Block(), _Block()), "not a bare"),
        (matrix.quantile_requires_ep, (), "ep=1 never runs"),
        (matrix.ptrr_load_balancer_backstop, (), "ptrr"),
        (matrix.gpt_oss_layout, (_Block(),), "bias vectors"),
        (matrix.group_limited_greedy, (), "single best expert"),
        (matrix.router_bias, (_Block(),), "router bias"),
        (matrix.quantile_requires_sigmoid, ("softmax", _Block()), "sigmoid"),
        (matrix.quantile_no_group_limit, (_Block(),), "group-limited"),
        (matrix.shared_expert_gate, (_Block(),), "shared_expert_gate"),
        (matrix.shared_expert_tp_ep, (_Block(),), "tp x ep"),
    ]
    for fn, args, match in cases:
        entry = next(r for r in matrix.ENTRIES if r.fn is fn)
        with pytest.raises(entry.error, match=match):
            fn(*args)


def test_validation_once_rows_are_config_errors() -> None:
    with pytest.raises(ConfigError):
        matrix.validation_once_requires_dp1(2)
    with pytest.raises(ConfigError):
        matrix.validation_once_requires_finite_corpus()
    # ... and therefore still ValueError, for the legacy assertions.
    with pytest.raises(ValueError):
        matrix.validation_once_requires_dp1(2)


# -- agreement: config and matrix give the same answer -------------------------


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"enable_sequence_parallel": False}, UnsupportedCombinationError),
        ({"tensor_parallel_size": 2, "expert_parallel_size": 2,
          "context_parallel_size": 2}, UnsupportedCombinationError),
        ({"ep_token_dispatcher": "deepep", "expert_parallel_size": 2},
         EnvironmentUnsupportedError),
        ({"ep_token_dispatcher": "torchao"}, UnsupportedCombinationError),
        ({"context_parallel_load_balancer": "ptrr"},
         UnsupportedCombinationError),
        ({"context_parallel_strategy": "ulysses"}, UnsupportedCombinationError),
    ],
)
def test_config_and_matrix_agree(overrides, error) -> None:
    """The same combination rejected by ParallelConfig must be the matrix row's
    verdict, with the same type -- never one raising and the other passing."""
    with pytest.raises(error):
        ParallelConfig(**overrides)
    with pytest.raises(error):
        matrix.check_config(_cfg(**overrides))
