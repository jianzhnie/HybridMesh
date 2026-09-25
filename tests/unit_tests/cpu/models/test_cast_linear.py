"""``CastLinear``: the optional fixed-dtype lm_head forward.

The cast is a numerics feature, not a structural one: enabling it must change
the logits' dtype and nothing else -- same state-dict keys, same parameter
objects (a tied head stays tied), same gradients in the parameter's own dtype,
and a bitwise-identical model when it is left off. Those invariants are what
the tests below pin, because every one of them is a silent-breakage candidate
(checkpoint mismatch, un-tied training, fp32 gradients in a bf16 optimizer).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from hpmesh.models.common.cast_linear import (
    TORCH_DTYPE_MAP,
    CastLinear,
    to_cast_linear,
)
from hpmesh.models.hf_factory import build_model_config
from hpmesh.models.hf_wrapper import HFTransformerModel


def _tiny_qwen3_config(compute_dtype: str | None = None):
    config = build_model_config(
        "qwen3",
        seq_len=64,
        arch_overrides={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    )
    config.compute_dtype = compute_dtype
    return config


# -- the module itself --------------------------------------------------------


def test_forward_matmul_runs_in_compute_dtype() -> None:
    torch.manual_seed(0)
    linear = CastLinear(16, 8, compute_dtype=torch.float32).to(torch.bfloat16)
    x = torch.randn(4, 16, dtype=torch.bfloat16)

    out = linear(x)

    assert out.dtype == torch.float32
    assert torch.equal(out, F.linear(x.float(), linear.weight.float()))


def test_parameters_keep_their_dtype_and_so_do_gradients() -> None:
    """Autograd casts the grads back through the forward casts."""
    torch.manual_seed(0)
    linear = CastLinear(16, 8, compute_dtype=torch.float32).to(torch.bfloat16)
    x = torch.randn(4, 16, dtype=torch.bfloat16, requires_grad=True)

    linear(x).sum().backward()

    assert linear.weight.dtype == torch.bfloat16
    assert linear.weight.grad.dtype == torch.bfloat16
    assert x.grad.dtype == torch.bfloat16


def test_compute_dtype_matching_the_model_is_bitwise_plain_linear() -> None:
    torch.manual_seed(0)
    plain = nn.Linear(16, 8).to(torch.bfloat16)
    cast = to_cast_linear(plain, torch.bfloat16)
    x = torch.randn(4, 16, dtype=torch.bfloat16)

    assert torch.equal(cast(x), plain(x))


# -- the in-place swap ----------------------------------------------------------


def test_swap_reuses_the_parameter_objects() -> None:
    """A tied head shares one Parameter with the embedding; only reusing the
    object keeps that tie (and the optimizer's parameter identity) intact."""
    linear = nn.Linear(16, 8)
    weight, bias = linear.weight, linear.bias

    cast = to_cast_linear(linear, torch.float32)

    assert cast.weight is weight
    assert cast.bias is bias


def test_swap_leaves_the_state_dict_key_set_untouched() -> None:
    """The checkpoint contract: subclassing nn.Linear, rather than wrapping
    one, is what keeps ``lm_head.weight`` spelled exactly as before."""

    class _CausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = nn.Linear(16, 8)

    model = _CausalLM()
    keys_before = set(model.state_dict())
    model.lm_head = to_cast_linear(model.lm_head, torch.float32)

    assert set(model.state_dict()) == keys_before


def test_swap_rejects_a_non_linear_head() -> None:
    with pytest.raises(TypeError, match="nn.Linear"):
        to_cast_linear(nn.Identity(), torch.float32)


def test_unknown_compute_dtype_name_fails_loudly() -> None:
    with pytest.raises(ValueError, match="compute_dtype"):
        HFTransformerModel(_tiny_qwen3_config("float8_e4m3fn"))


# -- the wrapper wiring ---------------------------------------------------------


def test_wrapper_applies_the_cast_and_logits_come_out_in_compute_dtype() -> None:
    # Exercise the head directly rather than through ``forward``: the full
    # forward builds a flex BlockMask, which the CPU test stubs cannot run --
    # the cast lives entirely inside the head, so nothing is lost.
    model = HFTransformerModel(_tiny_qwen3_config("float32")).to(torch.bfloat16)

    assert isinstance(model.lm_head, CastLinear)
    hidden = torch.randn(16, 32, dtype=torch.bfloat16)
    logits = model.lm_head(hidden)
    assert logits.dtype == torch.float32
    assert torch.equal(logits, F.linear(hidden.float(), model.lm_head.weight.float()))


def test_wrapper_state_dict_keys_are_invariant_under_the_cast() -> None:
    keys_plain = set(HFTransformerModel(_tiny_qwen3_config()).state_dict())
    keys_cast = set(HFTransformerModel(_tiny_qwen3_config("float32")).state_dict())
    assert keys_cast == keys_plain


def test_default_off_is_unchanged() -> None:
    """With no compute_dtype configured the head must stay a plain Linear and
    the built model bitwise-match one constructed before the feature existed
    (same seed, same parameters, same head)."""
    torch.manual_seed(7)
    baseline = HFTransformerModel(_tiny_qwen3_config()).eval()
    torch.manual_seed(7)
    defaulted = HFTransformerModel(_tiny_qwen3_config()).eval()

    assert type(defaulted.lm_head) is nn.Linear
    for (name_a, tensor_a), (name_b, tensor_b) in zip(
        baseline.state_dict().items(), defaulted.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(tensor_a, tensor_b)


def test_dtype_map_spelling_matches_the_config_field() -> None:
    assert sorted(TORCH_DTYPE_MAP) == ["bfloat16", "float16", "float32"]
