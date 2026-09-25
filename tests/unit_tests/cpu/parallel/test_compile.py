"""The compile step: gating, defaults, and the four compile-side toggles.

``hpmesh/parallel/compile.py`` ports torchtitan's ``distributed/compile.py``:
per-block compile, async TP, regional_inductor and capture_scalar_outputs,
each behind its own switch so the default is the historical whole-model
``torch.compile(model)``.

These tests pin, on CPU with no process group:

* the default: whole-model compile, dense models never touch
  ``capture_scalar_outputs``, and ``CompileConfig()`` reproduces the old
  plain ``torch.compile(model)`` path;
* per-block compile wraps every decoder layer and not the model;
* the config-time loud errors: async TP without compile, async TP at tp=1;
* the assembly-time loud errors: async TP without a TP mesh, async TP on a
  torch without ``_micro_pipeline_tp``, a flex model on a non-inductor /
  non-aot_eager backend, and regional_inductor on a torch that lacks it;
* the MoE condition for ``capture_scalar_outputs`` (set when the model part
  carries token-choice MoE blocks, unset for a dense one), with the global
  restored afterwards.

The model-level tests need the flex/spmd import surface, which this
machine's torch (2.2.2) only provides under the stub preamble; without it
they skip rather than error, so the default-collected suite is unchanged.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from hpmesh.config import (
    CompileConfig,
    HybridMeshConfig,
    ParallelConfig,
    TrainingConfig,
)

try:
    from torch._dynamo import OptimizedModule

    from hpmesh.models.hf_factory import build_model_config
    from hpmesh.models.hf_wrapper import HFTransformerModel
    from hpmesh.parallel import compile as compile_mod
    from hpmesh.parallel.compile import apply_compile, maybe_regional_inductor

    _IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - env gate, see module docstring
    _IMPORT_ERROR = e

requires_runtime = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"this torch lacks the flex/spmd import surface: {_IMPORT_ERROR}",
)

try:
    ParallelConfig()
    _PARALLEL_CONFIG_ERROR = None
except Exception as e:  # noqa: BLE001 - env gate (torch 2.2 has no pipelining)
    _PARALLEL_CONFIG_ERROR = e

requires_parallel_config = pytest.mark.skipif(
    _PARALLEL_CONFIG_ERROR is not None,
    reason=f"this torch cannot build a ParallelConfig: {_PARALLEL_CONFIG_ERROR}",
)

_VOCAB = 32
_HIDDEN = 16
_SEQ = 24
_NUM_LAYERS = 3


def _model(seed: int = 0):
    torch.manual_seed(seed)
    config = build_model_config(
        "llama",
        seq_len=_SEQ,
        arch_overrides={
            "vocab_size": _VOCAB,
            "hidden_size": _HIDDEN,
            "intermediate_size": 32,
            "num_hidden_layers": _NUM_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )
    return HFTransformerModel(config)


@contextlib.contextmanager
def _preserve_dynamo_flag(name: str):
    old = getattr(torch._dynamo.config, name)
    try:
        yield
    finally:
        setattr(torch._dynamo.config, name, old)


# -- config-time validation ---------------------------------------------------


def test_compile_config_defaults_reproduce_whole_model_compile() -> None:
    cfg = CompileConfig()
    assert cfg.per_block is False
    assert cfg.backend == "inductor"
    assert cfg.enable_async_tensor_parallel is False


def test_compile_config_rejects_empty_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        CompileConfig(backend="")


@requires_parallel_config
def test_async_tp_requires_compile() -> None:
    with pytest.raises(ValueError, match="requires training.compile=True"):
        HybridMeshConfig(
            parallel=ParallelConfig(tensor_parallel_size=2),
            training=TrainingConfig(
                compile=False,
                compile_config=CompileConfig(enable_async_tensor_parallel=True),
            ),
        )


@requires_parallel_config
def test_async_tp_requires_tp() -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size > 1"):
        HybridMeshConfig(
            parallel=ParallelConfig(tensor_parallel_size=1),
            training=TrainingConfig(
                compile=True,
                compile_config=CompileConfig(enable_async_tensor_parallel=True),
            ),
        )


@requires_parallel_config
def test_async_tp_valid_combination_passes() -> None:
    HybridMeshConfig(
        parallel=ParallelConfig(tensor_parallel_size=2),
        training=TrainingConfig(
            compile=True,
            compile_config=CompileConfig(enable_async_tensor_parallel=True),
        ),
    )


# -- the compile step itself ----------------------------------------------------


@requires_runtime
def test_default_compile_is_whole_model_and_leaves_dynamo_flags() -> None:
    """compile_config defaults: whole-model compile, dense flags untouched."""
    with _preserve_dynamo_flag("capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = False
        model = apply_compile(_model())
        assert isinstance(model, OptimizedModule)
        assert not any(isinstance(layer, OptimizedModule) for layer in model.layers)
        # Dense model: the MoE-only flag is untouched.
        assert torch._dynamo.config.capture_scalar_outputs is False


@requires_runtime
def test_per_block_compile_wraps_each_layer_not_the_model() -> None:
    model = apply_compile(_model(), compile_config=CompileConfig(per_block=True))
    assert not isinstance(model, OptimizedModule)
    # Module.compile is in place: same objects, compiled call impl attached.
    assert all(layer._compiled_call_impl is not None for layer in model.layers)


@requires_runtime
def test_async_tp_without_tp_mesh_raises() -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size > 1"):
        apply_compile(
            _model(),
            compile_config=CompileConfig(enable_async_tensor_parallel=True),
            tp_mesh=None,
        )


@requires_runtime
@pytest.mark.skipif(
    hasattr(__import__("torch._inductor.config", fromlist=["x"]), "_micro_pipeline_tp"),
    reason="this torch carries _micro_pipeline_tp; the loud-raise is unreachable",
)
def test_async_tp_on_unsupported_torch_loud_raises() -> None:
    with pytest.raises(NotImplementedError, match="_micro_pipeline_tp"):
        apply_compile(
            _model(),
            compile_config=CompileConfig(enable_async_tensor_parallel=True),
            tp_mesh=object(),  # never reached: the capability check fires first
        )


@requires_runtime
def test_regional_inductor_not_triggered_without_flex() -> None:
    """CPU/sdpa model + aot_eager: the backend passes through unwrapped."""
    backend = compile_mod._maybe_regional_inductor_backend(_model(), "aot_eager")
    assert backend == "aot_eager"
    assert compile_mod._regional_inductor_enabled is False


@requires_runtime
def test_flex_model_rejects_unknown_backend(monkeypatch) -> None:
    model = _model()
    monkeypatch.setattr(
        type(model), "uses_flex_attention", property(lambda self: True)
    )
    with pytest.raises(ValueError, match="neither 'inductor' nor 'aot_eager'"):
        apply_compile(model, compile_config=CompileConfig(backend="eager"))


@requires_runtime
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch.fx.passes.regional_inductor")
    is not None,
    reason="this torch carries regional_inductor; the loud-raise is unreachable",
)
def test_regional_inductor_on_unsupported_torch_loud_raises(monkeypatch) -> None:
    model = _model()
    monkeypatch.setattr(
        type(model), "uses_flex_attention", property(lambda self: True)
    )
    with pytest.raises(NotImplementedError, match="regional_inductor"):
        apply_compile(model, compile_config=CompileConfig(backend="aot_eager"))


@requires_runtime
def test_maybe_regional_inductor_is_null_when_disabled() -> None:
    assert not compile_mod._regional_inductor_enabled
    with maybe_regional_inductor({}):
        pass  # must be a no-op context on the default path


@requires_runtime
def test_capture_scalar_outputs_set_only_for_token_choice_moe(monkeypatch) -> None:
    """The flag follows the MoE-dispatch condition, and only that condition.

    A real swapped MoE cannot be built in this environment (the installed
    transformers' Qwen3Moe layout predates the swap's probe), so the
    detection seam itself is stubbed; ``_iter_moe_layers`` is pinned by the
    EP tests.
    """
    with _preserve_dynamo_flag("capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = False
        monkeypatch.setattr(compile_mod, "_iter_moe_layers", lambda m: [object()])
        apply_compile(_model(), compile_config=CompileConfig())
        assert torch._dynamo.config.capture_scalar_outputs is True

    monkeypatch.undo()  # restore the real _iter_moe_layers for the dense case

    with _preserve_dynamo_flag("capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = False
        apply_compile(_model(), compile_config=CompileConfig())
        assert torch._dynamo.config.capture_scalar_outputs is False
