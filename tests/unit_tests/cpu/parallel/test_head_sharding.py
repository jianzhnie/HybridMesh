"""Head-count divisibility: the guard TP and ulysses CP both lean on.

torchtitan rejects ``heads % (tp * cp) != 0`` once, while parsing the run config
(``validate_context_parallel``), because it builds the model from its own config.
llmtuner reads the counts off the HF config at wire-up, so the check fires from
``apply_tp`` (``% tp``) and the ulysses branch of ``apply_cp`` (``% tp*cp``);
together they are upstream's single check.

This module drives the shared helper directly, which is why it is not gated on
``spmd_types``: the two call sites are covered in ``test_tp.py`` / ``test_cp.py``,
both of which do need the full torch stack. The property worth pinning here is
that the guard is *loud* about a count it can see, and silent -- not wrong --
about a model that carries no attention config to check.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from llmtuner.parallel.head_sharding import (
    head_counts,
    require_heads_divisible_by,
)


def _model(num_attention_heads=None, num_key_value_heads=None) -> torch.nn.Module:
    """A stub shaped like the wrapper: config under ``.model``, nothing else."""
    config = SimpleNamespace()
    if num_attention_heads is not None:
        config.num_attention_heads = num_attention_heads
    if num_key_value_heads is not None:
        config.num_key_value_heads = num_key_value_heads
    model = torch.nn.Linear(4, 4)
    model.model = SimpleNamespace(config=config)
    return model


def _check(model, degree: int, divisor: str = "tp") -> None:
    require_heads_divisible_by(model, degree=degree, divisor=divisor, why="because")


def test_division_is_the_whole_rule() -> None:
    _check(_model(8, 8), 1)
    _check(_model(8, 8), 2)
    _check(_model(8, 8), 8)


def test_a_head_count_that_does_not_divide_is_refused() -> None:
    """The case the projection-level guard cannot see: 8 KV heads at
    head_dim=128 is 1024 features, which divides by 16 even though the heads do
    not -- so without this check the run gets as far as HF's head reshape."""
    with pytest.raises(ValueError, match=r"num_attention_heads \(8\).*tp \(16\)"):
        _check(_model(8, 8), 16)


def test_kv_heads_are_checked_with_their_own_count() -> None:
    """GQA: 8 query heads over tp=8 is fine, but 4 KV heads are not."""
    _check(_model(8, 4), 4)
    with pytest.raises(ValueError, match=r"num_key_value_heads \(4\)"):
        _check(_model(8, 4), 8)


def test_a_missing_kv_count_means_every_head_is_a_kv_head() -> None:
    """configs that omit ``num_key_value_heads`` are MHA, so the fallback is the
    query count -- not 'unchecked'."""
    assert head_counts(_model(8)) == [
        ("num_attention_heads", 8),
        ("num_key_value_heads", 8),
    ]
    _check(_model(8), 8)
    # Both entries fall back to 8, so tp=16 fails on the first one reported.
    with pytest.raises(ValueError, match=r"num_attention_heads \(8\)"):
        _check(_model(8), 16)


def test_a_model_with_no_attention_config_is_skipped() -> None:
    """Nothing to divide, nothing to say. The guard must not turn a metadata
    gap into a startup failure -- ``apply_cp``'s own ``layers`` check is what
    rejects a model that is not an HF wrapper."""
    plain = torch.nn.Linear(4, 4)
    assert head_counts(plain) == []
    _check(plain, 16)

    config_less = _model()
    assert head_counts(config_less) == []
    _check(config_less, 16)


def test_the_degree_one_axis_is_never_checked() -> None:
    """tp=1 / cp=1 is the disabled axis: an odd head count is legal there, and
    a '2 heads do not divide 1' style message would be nonsense."""
    _check(_model(3, 1), 1)


def test_both_counts_are_reported_independently() -> None:
    """One bad count must not mask the other, and the message carries the
    divisor the caller named -- 'tp*cp' for the ulysses call site."""
    with pytest.raises(ValueError, match=r"must be divisible by tp\*cp \(4\)"):
        _check(_model(4, 2), 4, divisor="tp*cp")
