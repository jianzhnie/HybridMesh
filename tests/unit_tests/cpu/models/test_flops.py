"""``num_flops_per_token``: the MFU denominator, over the real architectures.

MFU is a number nobody cross-checks: a loss curve cannot see it, and a wrong
denominator produces output that looks exactly like a right one. The two ways
this function was wrong before are both silent -- an MoE counted as one dense
FFN (2x low for Qwen3-MoE, ~1.8x for DeepSeek-V3) and an MLA whose ``head_dim``
is the rope slice rather than the QK width -- so each is pinned here against an
independent arithmetic check on the same HF config.

Everything runs on CPU with no model and no ``LLMTunerConfig``: the function is
config-only on purpose, and the HF config classes are the same ones the trainer
resolves through ``AutoConfig.for_model``. Where a case cannot be expressed by a
real config class (a field HF validates rather than allows to be unset), an
attribute bag stands in and the test says so.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from llmtuner.models.hf import factory
from llmtuner.models.hf.factory import (
    flops_per_token,
    num_flops_per_token,
    quadratic_attention_flops_per_token,
)


class _Arch:
    """An attribute bag standing in for an HF config.

    HF 5.x configs are strict dataclasses: a field cannot be set to ``None`` to
    express "this architecture does not declare it". The refusal cases need
    exactly that, so they are stated on a bag -- the contract under test is the
    return value, not HF's validation.
    """

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


def _arch(name: str, **kwargs) -> PretrainedConfig:
    """A real HF config, so the field names are the ones the trainer reads."""
    kwargs.setdefault("max_position_embeddings", 64)
    kwargs.setdefault("vocab_size", 64)
    kwargs.setdefault("hidden_size", 32)
    kwargs.setdefault("intermediate_size", 64)
    kwargs.setdefault("num_hidden_layers", 2)
    kwargs.setdefault("num_attention_heads", 4)
    kwargs.setdefault("num_key_value_heads", 2)
    cfg = AutoConfig.for_model(name, **kwargs)
    # Pin the head width: several families default it to something other than
    # hidden_size / num_attention_heads, and the hand formulas spell it out.
    cfg.head_dim = kwargs.get("head_dim", 8)
    return cfg


# -- the dense path (the pre-existing formula, unchanged) ---------------------


def test_flops_are_affine_in_the_layer_count() -> None:
    """Every term scales with the layer count except the output projection, so
    the marginal cost of a layer is constant and the FLOPs are affine in the
    depth. Reading it as proportional would hide the lm_head offset."""
    one = flops_per_token(_arch("llama", num_hidden_layers=1), seq_len=8)
    two = flops_per_token(_arch("llama", num_hidden_layers=2), seq_len=8)
    four = flops_per_token(_arch("llama", num_hidden_layers=4), seq_len=8)

    assert four - two == 2 * (two - one)


def test_the_dense_formula_is_exact() -> None:
    """A fully specified model, so every term is checked rather than bounded."""
    hidden, intermediate, vocab, heads, kv_heads = 32, 64, 64, 4, 2
    head_dim, seq_len, layers = 8, 8, 2
    arch = _arch(
        "llama",
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
    )
    per_layer = (
        2 * hidden * heads * head_dim  # q
        + 2 * hidden * kv_heads * head_dim  # k
        + 2 * hidden * kv_heads * head_dim  # v
        + 2 * heads * head_dim * hidden  # o
        + 3 * 2 * hidden * intermediate  # gate, up, down
    )
    attention = 6 * heads * 2 * head_dim * seq_len

    assert flops_per_token(arch, seq_len=seq_len) == (
        3 * (layers * per_layer + 2 * vocab * hidden) + layers * attention
    )


def test_the_attention_term_grows_with_sequence_length() -> None:
    """The parameter term does not depend on the sequence length, so this
    isolates the attention term."""
    arch = _arch("llama")

    assert flops_per_token(arch, seq_len=32) > flops_per_token(arch, seq_len=8)


def test_kv_heads_default_to_the_head_count() -> None:
    """Missing GQA metadata must fall back to full multi-head attention for the
    K/V projection widths, not crash or count zero. The failure mode is a
    silently wrong magnitude, so it is pinned here."""
    unset = _arch("llama", num_key_value_heads=None)
    explicit = _arch("llama", num_key_value_heads=4)

    assert flops_per_token(unset, seq_len=8) == flops_per_token(explicit, seq_len=8)


def test_a_missing_size_suppresses_mfu_instead_of_guessing() -> None:
    """A config whose geometry the formula cannot read reports 0, which
    suppresses MFU and tflops rather than producing a guessed number."""

    class _Bare:
        pass

    assert flops_per_token(_Bare(), seq_len=4) == 0


def test_num_flops_per_token_passes_the_run_sequence_length_through(
    monkeypatch,
) -> None:
    """``num_flops_per_token`` only resolves the config and forwards
    ``cfg.max_seq_len``; the attention term must use the run's own sequence
    length, not the architecture's maximum."""
    monkeypatch.setattr(
        factory, "build_model_config_for", lambda cfg: _arch("llama")
    )
    short = num_flops_per_token(SimpleNamespace(max_seq_len=8))
    long = num_flops_per_token(SimpleNamespace(max_seq_len=32))

    assert short > 0
    assert long > short


# -- MoE ----------------------------------------------------------------------


def test_a_moe_layer_counts_only_its_top_k_experts() -> None:
    """The routed experts are stored ``num_experts`` times over but only
    ``top_k`` of them see each token. Counting all of them -- or, as before, one
    dense FFN's worth -- is off by the active-expert ratio."""
    hidden, heads, kv_heads, head_dim = 32, 4, 2, 8
    layers, vocab, seq_len = 2, 64, 16
    num_experts, top_k, expert_width = 8, 2, 16
    arch = _arch(
        "qwen3_moe",
        intermediate_size=expert_width,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=expert_width,
    )

    attention = 2 * (
        hidden * heads * head_dim  # q
        + 2 * hidden * kv_heads * head_dim  # k + v
        + heads * head_dim * hidden  # o
    )
    moe = 2 * (
        hidden * num_experts  # router
        + top_k * 3 * hidden * expert_width  # routed experts, top_k active
    )
    expected = (
        3 * (layers * (attention + moe) + 2 * vocab * hidden)
        + layers * 6 * heads * 2 * head_dim * seq_len
    )

    assert flops_per_token(arch, seq_len=seq_len) == expected


def test_shared_experts_are_counted_in_full() -> None:
    """A shared expert sees every token, so it is not scaled by the active
    ratio -- only the routed experts are."""
    hidden, layers, shared_width, seq_len = 32, 2, 16, 16
    kwargs = dict(
        intermediate_size=16,
        num_hidden_layers=layers,
        num_local_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
    )
    without = _arch("qwen3_moe", **kwargs)
    with_shared = _arch("qwen3_moe", **kwargs)
    with_shared.shared_expert_intermediate_size = shared_width

    delta = flops_per_token(with_shared, seq_len=seq_len) - flops_per_token(
        without, seq_len=seq_len
    )

    assert delta == 3 * layers * 2 * 3 * hidden * shared_width


def test_a_single_width_family_takes_the_expert_width_from_intermediate_size() -> None:
    """Mixtral and OLMoE declare one FFN width and no separate MoE width, and
    no dense layer for it to belong to, so that width is the experts'."""
    arch = _arch(
        "mixtral",
        intermediate_size=16,
        num_local_experts=8,
        num_experts_per_tok=2,
    )

    assert flops_per_token(arch, seq_len=16) > 0


def test_the_dense_prefix_of_a_mixed_stack_costs_the_dense_width() -> None:
    """DeepSeek runs ``first_k_dense_replace`` dense FFNs before its MoE layers,
    at a different width from the experts. Costing every layer as MoE would
    charge those layers a router and an expert width they do not have."""
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_shared_experts=0,
    )
    all_moe = _arch("deepseek_v3", first_k_dense_replace=0, **kwargs)
    two_dense = _arch("deepseek_v3", first_k_dense_replace=2, **kwargs)

    hidden, dense_width = 32, 64
    # Per converted layer, in units of 2 * in * out: it loses its router and its
    # top_k routed experts, and gains the dense gate/up/down triple.
    per_layer = 2 * (hidden * 8 + 2 * 3 * hidden * 16) - 3 * 2 * hidden * dense_width
    delta = flops_per_token(all_moe, seq_len=16) - flops_per_token(
        two_dense, seq_len=16
    )

    assert delta == 3 * 2 * per_layer


# -- MLA ----------------------------------------------------------------------


def test_an_mla_layer_uses_the_qk_and_v_widths_not_head_dim() -> None:
    """DeepSeek-V3 spells ``head_dim`` as the rope slice (64) against a 192-wide
    QK, so reading ``head_dim`` undercounts both the projections and the
    attention contractions."""
    hidden, heads, layers, vocab, seq_len = 128, 8, 2, 1024, 16
    qk_nope, qk_rope, v_head_dim = 16, 8, 16
    q_lora, kv_lora = 32, 16
    arch = _arch(
        "deepseek_v3",
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=256,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=4,  # deliberately not the QK width
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_shared_experts=0,
        first_k_dense_replace=0,
        q_lora_rank=q_lora,
        kv_lora_rank=kv_lora,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=v_head_dim,
        qk_head_dim=qk_nope + qk_rope,
    )
    qk = qk_nope + qk_rope
    attention = 2 * (
        hidden * q_lora
        + q_lora * heads * qk  # q_a + q_b
        + hidden * (kv_lora + qk_rope)
        + kv_lora * heads * (qk_nope + v_head_dim)  # kv_a + kv_b
        + heads * v_head_dim * hidden  # o
    )
    moe = 2 * (hidden * 8 + 2 * 3 * hidden * 32)
    expected = (
        3 * (layers * (attention + moe) + 2 * vocab * hidden)
        + layers * 6 * heads * (qk + v_head_dim) * seq_len
    )

    assert flops_per_token(arch, seq_len=seq_len) == expected


def test_the_deepseek_implied_active_params_match_the_published_figure() -> None:
    """The real V3 config, whose published activated-parameter count (37B) is an
    outside check on the MLA and MoE terms at once: dropping the low-rank MLA
    projections, or counting all 256 experts as active, cannot land on it."""
    arch = _arch(
        "deepseek_v3",
        hidden_size=7168,
        intermediate_size=18432,
        num_hidden_layers=61,
        num_attention_heads=128,
        num_key_value_heads=128,
        vocab_size=129280,
        max_position_embeddings=4096,
    )
    seq_len = 4096
    total = flops_per_token(arch, seq_len=seq_len)
    attention_ops = 61 * quadratic_attention_flops_per_token(
        num_heads=128,
        qk_head_dim=192,
        v_head_dim=128,
        seq_len=seq_len,
    )
    implied_active_params = (total - attention_ops) / 6

    assert 3.5e10 < implied_active_params < 3.8e10


# -- attention windows --------------------------------------------------------


def test_a_sliding_layer_attends_to_the_window_not_the_sequence() -> None:
    """GPT-OSS mixes full and 128-token sliding layers; charging every layer the
    full sequence overstates the attention term on the sliding ones."""
    kwargs = dict(
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=8,
        num_experts_per_tok=2,
    )
    full = _arch("gpt_oss", **kwargs)
    full.layer_types = ["full_attention"] * 4
    sliding = _arch("gpt_oss", **kwargs)
    sliding.layer_types = ["sliding_attention"] * 4
    sliding.sliding_window = 8

    heads, head_dim, seq_len, window = 4, 8, 32, 8
    delta = flops_per_token(full, seq_len=seq_len) - flops_per_token(
        sliding, seq_len=seq_len
    )

    assert delta == 4 * (
        quadratic_attention_flops_per_token(
            num_heads=heads, qk_head_dim=head_dim, v_head_dim=head_dim, seq_len=seq_len
        )
        - quadratic_attention_flops_per_token(
            num_heads=heads,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            seq_len=seq_len,
            sliding_window_size=window,
        )
    )


def test_an_uniform_window_caps_every_layer() -> None:
    """The older spelling (Mistral): ``sliding_window`` with no ``layer_types``."""
    kwargs = dict(sliding_window=4)
    windowed = _arch("mistral", **kwargs)
    unwindowed = _arch("mistral", **kwargs)
    unwindowed.sliding_window = None

    assert flops_per_token(unwindowed, seq_len=64) > flops_per_token(
        windowed, seq_len=64
    )


# -- the refusal cases --------------------------------------------------------


def test_a_linear_attention_layer_suppresses_the_count() -> None:
    """Qwen3-Next's delta-rule layers project through kernels whose parameter
    term is not derivable from these generic fields, so the whole count is
    suppressed rather than approximated."""
    arch = _arch("qwen3_next", max_position_embeddings=64)

    assert flops_per_token(arch, seq_len=64) == 0


def test_a_short_layer_types_list_suppresses_the_count() -> None:
    """A list that does not cover every layer leaves the mix unknown; upstream
    substitutes full attention, which is only safe when that is the default."""
    arch = _arch("llama", num_hidden_layers=4)
    arch.layer_types = ["full_attention", "sliding_attention"]
    arch.sliding_window = 8

    assert flops_per_token(arch, seq_len=64) == 0


def test_a_moe_without_a_resolvable_width_suppresses_the_count() -> None:
    """Experts declared, but no width for them and no single FFN width to
    borrow: a dense prefix means ``intermediate_size`` is the dense width, not
    the experts'."""
    arch = _Arch(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        first_k_dense_replace=2,
    )

    assert flops_per_token(arch, seq_len=64) == 0


def test_an_mla_without_its_ranks_suppresses_the_count() -> None:
    """MLA declared (``q_lora_rank``) without ``kv_lora_rank`` is half a
    geometry; the low-rank projections cannot be sized from the rest."""
    arch = _Arch(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
    )

    assert flops_per_token(arch, seq_len=64) == 0


@pytest.mark.parametrize("seq_len", [0, 1])
def test_a_degenerate_sequence_length_is_still_a_number(seq_len: int) -> None:
    """``seq_len`` only scales the attention term, so a one-token or empty
    sequence is a valid count rather than a division or a zero."""
    assert flops_per_token(_arch("llama"), seq_len=seq_len) > 0
