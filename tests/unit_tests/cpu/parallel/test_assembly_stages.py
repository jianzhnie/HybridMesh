"""The assembly stage list: the order contract, pinned.

``parallelize_hf_transformers``'s correctness is its call order (TP before EP
before CP, AC before compile, FSDP last). That used to live in a docstring;
it is now the ``STAGES`` table, and this test is what keeps the table -- not
the prose -- as the contract.
"""

from hpmesh.parallel.stages import (
    PP_STAGE_ORDER,
    STAGE_ORDER,
    STAGES,
    _stage_enabled,
)


def test_the_full_order_is_the_documented_contract() -> None:
    assert STAGE_ORDER == ("tp", "ep", "cp", "ac", "compile", "fsdp")


def test_fsdp_wraps_last() -> None:
    assert STAGE_ORDER[-1] == "fsdp"


def test_sharding_wrappers_come_before_ac_and_compile() -> None:
    tp, ep, cp, ac, compile_ = (
        STAGE_ORDER.index(n) for n in ("tp", "ep", "cp", "ac", "compile")
    )
    assert tp < ep < cp < ac < compile_


def test_the_pp_order_is_a_subsequence_of_the_full_order() -> None:
    assert PP_STAGE_ORDER == ("tp", "compile", "fsdp")
    positions = [STAGE_ORDER.index(name) for name in PP_STAGE_ORDER]
    assert positions == sorted(positions)


def test_every_pp_stage_is_flagged_on_pp_and_vice_versa() -> None:
    assert PP_STAGE_ORDER == tuple(s.name for s in STAGES if s.on_pp)
    assert {s.name for s in STAGES} == set(STAGE_ORDER)


def test_only_compile_is_conditional() -> None:
    for name in STAGE_ORDER:
        assert _stage_enabled(name, compile=True) is True
        assert _stage_enabled(name, compile=False) is (name != "compile")


def test_every_stage_documents_why_it_sits_where_it_does() -> None:
    for stage in STAGES:
        assert stage.why_here
