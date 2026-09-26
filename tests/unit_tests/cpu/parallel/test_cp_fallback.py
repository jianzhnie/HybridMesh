"""The CPU eager fallback must refuse attention modifiers it cannot represent.

``CPFlexKernel`` has two arms. On CUDA it calls HF's
``flex_attention_forward``, whose ``score_mod`` applies ``softcap`` and the
post-hoc attention-sink (``s_aux``) renormalization. Off CUDA it calls
``torch.nn.attention.flex_attention.flex_attention`` directly -- whose signature
has no ``softcap`` parameter at all -- and forwards only ``scaling``.

So on CPU a model with ``attn_logit_softcapping`` set (Gemma-2/3) would compute
attention *without* the cap and produce a confidently wrong answer rather than
an error. These tests pin the refusal that closes that hole, and -- just as
important -- that it does not fire on the ordinary kwargs every flex call
carries, which would break the CPU equivalence tests this fallback exists for.
"""

from __future__ import annotations

from tests.caps import require_env

require_env('spmd_types')


import pytest

from hpmesh.parallel.context_parallel.cp_kernel import (
    reject_unrepresentable_attention_kwargs,
)


def test_the_cpu_fallback_refuses_softcap() -> None:
    """``softcap`` has no CPU counterpart, so it must raise, not be dropped.

    Non-vacuity: the assertion is that the call raises, so a no-op
    implementation (the pre-guard behavior, where the kwarg was silently
    dropped) fails this test rather than passing it.
    """
    with pytest.raises(NotImplementedError, match="softcap"):
        reject_unrepresentable_attention_kwargs(
            {"scaling": 0.25, "softcap": 50.0, "dropout": 0.0}
        )


def test_the_cpu_fallback_refuses_attention_sinks() -> None:
    """``s_aux`` is applied post-hoc around the LSE, which CPU flex omits."""
    import torch

    with pytest.raises(NotImplementedError, match="s_aux"):
        reject_unrepresentable_attention_kwargs(
            {"scaling": 0.25, "s_aux": torch.zeros(4)}
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"scaling": 0.25},
        {"scale": 0.25, "enable_gqa": True},
        {"scaling": 0.25, "dropout": 0.0, "kernel_options": None},
        # A None value is the "not configured" spelling, not a request.
        {"scaling": 0.25, "softcap": None, "s_aux": None},
    ],
)
def test_ordinary_flex_kwargs_pass_through(kwargs: dict) -> None:
    """The guard must not fire on what every flex call carries.

    This is the half that keeps the guard from being a blunt instrument: the
    kwargs HF passes on an ordinary run (``scaling``/``dropout``/
    ``kernel_options``) are all representable, and the CPU equivalence harnesses
    drive exactly this path.
    """
    reject_unrepresentable_attention_kwargs(kwargs)
