"""The combination matrix: assembly/probe verdicts, types, messages, guards.

The matrix's scope is cross-layer combination verdicts (assembly and probe
phase). Config-phase combination checks live in the configs' own
``__post_init__`` and are covered by ``test_config.py`` -- there is no
config-vs-matrix agreement to test anymore because there is only one copy.
"""

import pytest

from hpmesh.parallel import matrix


def test_every_row_has_a_verdict_reason_and_guard() -> None:
    for row in matrix.ENTRIES:
        assert row.name == row.fn.__name__
        assert row.reason == row.fn.__doc__.strip()
        assert row.phase in ("assembly", "probe")
        assert issubclass(row.error, Exception)
        assert row.reason and row.guard


def test_every_guard_function_has_a_row() -> None:
    # The table and the functions cannot drift: every row references a real
    # function in this module, and every public guard function is tabled.
    import inspect

    public = {
        n
        for n, v in vars(matrix).items()
        if inspect.isfunction(v)
        and v.__module__ == matrix.__name__
        and not n.startswith("_")
    }
    assert {r.name for r in matrix.ENTRIES} == public


def test_config_phase_is_out_of_scope() -> None:
    # Config-phase checks live in config/* __post_init__ (see test_config.py);
    # the matrix carries only the cross-layer verdicts.
    assert {r.phase for r in matrix.ENTRIES} == {"assembly", "probe"}


def test_guards_point_at_real_files() -> None:
    from pathlib import Path

    root = Path(__file__).parents[4] / "hpmesh"
    for row in matrix.ENTRIES:
        path = row.guard.split("::")[0]
        assert (root / path).is_file(), row.guard


# -- every row rejects with its entry's type and message ------------------------


def test_rows_reject_with_their_entry_type() -> None:
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
    assert len(cases) == len(matrix.ENTRIES)
    for fn, args, match in cases:
        entry = next(r for r in matrix.ENTRIES if r.fn is fn)
        with pytest.raises(entry.error, match=match):
            fn(*args)


def test_validation_once_rows_are_config_errors() -> None:
    from hpmesh.errors import ConfigError

    with pytest.raises(ConfigError):
        matrix.validation_once_requires_dp1(2)
    with pytest.raises(ConfigError):
        matrix.validation_once_requires_finite_corpus()
    # ... and therefore still ValueError, for the legacy assertions.
    with pytest.raises(ValueError):
        matrix.validation_once_requires_dp1(2)
