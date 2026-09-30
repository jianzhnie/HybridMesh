"""The RegionAC vocabulary and save-pattern rule (the engine-free half).

``RegionAC`` retains the regions a block declares and recomputes the rest. In
torchtitan the declarations are ``torch_remat.region`` call sites in its own
model code, wired by ``Module.configure_remat_regions``; llmtuner runs HF models
and does not own their decoder code, so ``parallel/remat_regions.py`` derives the
vocabulary structurally -- the block's ``nn.Linear``s -- and applies upstream's
``fnmatch`` save rule to the configured patterns.

That half is testable everywhere, and it is the half a wrong change would break
silently: a renamed region makes a save pattern match nothing, which degrades the
policy to full checkpointing without an error. The ``torch_remat`` call shape
itself is pinned in ``test_activation_checkpoint.py``, which needs the AC engine.

The block below is a stand-in for an HF decoder layer rather than a real one:
building an HF model needs transformers' torch integration, which the torch
versions this repo develops against do not provide. Its submodule names are
LLaMA's (``self_attn.{q,k,v,o}_proj``, ``mlp.{gate,up,down}_proj``); the
same-vocabulary assertion against a real HF model lives in
``test_activation_checkpoint.py``.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from llmtuner.config import (
    VALID_AC_MODES,
    RegionACConfig,
    TrainingConfig,
)
from llmtuner.errors import ConfigError
from llmtuner.parallel.remat_regions import (
    region_names,
    region_policy,
    should_recompute,
    unmatched_save_patterns,
)


class _Attention(nn.Module):
    """The projections HF gives every dense attention."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class _Mlp(nn.Module):
    """The gated MLP HF gives every LLaMA-family block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.up_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)


class _DecoderLayer(nn.Module):
    """One block, with an HF decoder layer's module names."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(dim)
        self.self_attn = _Attention(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim)


_REGIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


# -- the vocabulary ------------------------------------------------------------


def test_the_vocabulary_is_the_blocks_linears() -> None:
    """Only the projections are regions: norms and the block itself are not."""
    assert region_names(_DecoderLayer()) == _REGIONS


def test_the_vocabulary_is_relative_to_the_block() -> None:
    """Block-relative names are what let one policy cover every block."""
    assert all(not name.startswith("layers.") for name in region_names(_DecoderLayer()))


def test_a_block_without_projections_has_no_regions() -> None:
    """An empty vocabulary is reported, not an error: the log names it."""
    assert region_names(nn.LayerNorm(4)) == ()


# -- the save rule -------------------------------------------------------------


def test_a_match_retains_and_a_miss_recomputes() -> None:
    assert not should_recompute("self_attn.q_proj", ["self_attn.q_proj"])
    assert should_recompute("self_attn.q_proj", ["mlp.down_proj"])


def test_patterns_are_shell_globs() -> None:
    """Upstream's rule: ``fnmatch`` against the qualified region name."""
    assert not should_recompute("self_attn.q_proj", ["self_attn.*"])
    assert not should_recompute("mlp.down_proj", ["*_proj"])
    assert not should_recompute("mlp.down_proj", ["mlp.?own_proj"])
    assert should_recompute("mlp.down_proj", ["self_attn.*"])


def test_no_patterns_recomputes_everything() -> None:
    """The default retains nothing, which is what full checkpointing means."""
    assert all(region_policy(_REGIONS, [])[region] for region in _REGIONS)


def test_the_policy_is_the_rule_applied_per_region() -> None:
    patterns = ["self_attn.*", "mlp.down_proj"]
    policy = region_policy(_REGIONS, patterns)
    assert policy == {region: should_recompute(region, patterns) for region in _REGIONS}
    assert [region for region, recompute in policy.items() if not recompute] == [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.down_proj",
    ]


def test_unmatched_patterns_are_reported() -> None:
    """A typo would otherwise be indistinguishable from a missing region."""
    assert unmatched_save_patterns(_REGIONS, ["mlp.nope", "self_attn.*"]) == (
        "mlp.nope",
    )
    assert unmatched_save_patterns(_REGIONS, ["self_attn.*"]) == ()


# -- config --------------------------------------------------------------------


def test_region_is_an_accepted_mode_with_its_own_config() -> None:
    assert "region" in VALID_AC_MODES
    cfg = TrainingConfig(activation_checkpoint_mode="region")
    assert cfg.region_ac.save_regions == []
    assert cfg.region_ac.determinism_check == "default"


def test_the_default_config_retains_nothing_and_keeps_upstream_defaults() -> None:
    cfg = RegionACConfig()
    assert cfg.save_regions == []
    assert cfg.preserve_rng_state is False


def test_preserve_rng_state_true_is_refused_with_the_hook_guidance() -> None:
    """torch_remat refuses it too; the config says so before the run starts."""
    with pytest.raises(ConfigError, match="RecomputeStateHook"):
        RegionACConfig(preserve_rng_state=True)


def test_an_unknown_mode_is_still_rejected() -> None:
    with pytest.raises(ConfigError, match="activation_checkpoint_mode"):
        TrainingConfig(activation_checkpoint_mode="region_ac")


def test_configuring_the_mode_does_not_import_torch_remat(monkeypatch) -> None:
    """The package is reached at apply time, so a config can name the mode without it.

    ``sys.modules["torch_remat"] = None`` makes any ``import torch_remat`` raise,
    so this fails the moment construction starts reaching for the package.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch_remat", None)
    cfg = TrainingConfig(
        activation_checkpoint_mode="region",
        region_ac=RegionACConfig(save_regions=["mlp.down_proj"]),
    )
    assert cfg.region_ac.save_regions == ["mlp.down_proj"]
