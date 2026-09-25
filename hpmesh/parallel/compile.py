"""``torch.compile`` application: whole-model or per-block, plus its toggles.

This is the hpmesh counterpart of torchtitan's ``distributed/compile.py``.
That file does four independent things beyond the compile call itself, and
each lands here behind its own switch on ``CompileConfig`` (or its upstream
condition), so the default -- ``CompileConfig()`` -- is exactly the old
behavior: one whole-model ``torch.compile(model, backend="inductor")``.

* **Per-block compile** (``per_block=True``): compile each decoder layer
  separately with ``fullgraph=True``. The repeated block structure is then
  traced once and the graph reused, which is what makes compile time scale
  on deep models. Whole-model compile stays the default because it is the
  path every existing hpmesh run has taken.
* **Async TP** (``enable_async_tensor_parallel=True``): Inductor's
  micro-pipeline pass overlaps the TP collectives with the GEMMs inside
  compiled regions. It needs a TP mesh (``tensor_parallel_size > 1``),
  compile, symmetric memory registered for the TP group, and a torch that
  carries ``torch._inductor.config._micro_pipeline_tp`` -- any missing
  piece is a loud error at assembly time, never a silent skip.
* **regional_inductor**: under a non-inductor backend (``aot_eager``),
  flex attention has no lowering of its own and would decompose to eager
  aten ops. When the model routes through hpmesh's flex path, the backend
  is wrapped so just the annotated flex region is scooped into an inductor
  sub-compile (the annotation is ``maybe_regional_inductor`` at the flex
  call site in ``models/hf_wrapper.py``). A flex model with any other
  non-inductor backend raises, as upstream does.
* **capture_scalar_outputs**: token-choice MoE dispatch has data-dependent
  shapes (per-expert token counts), which dynamo cannot trace without
  ``torch._dynamo.config.capture_scalar_outputs``. Set -- with the same
  condition upstream documents -- when the compiled model part actually
  carries hpmesh MoE blocks; dense models keep the flag untouched.

Ordering: this runs after activation checkpointing and before FSDP, the
same slot the whole-model compile it replaces occupied.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable

import torch
import torch.nn as nn

from hpmesh.config import CompileConfig
from hpmesh.errors import EnvironmentUnsupportedError

from ..accelerator.capabilities import has
from ..models.common.moe import _iter_moe_layers
from ..utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ["apply_compile", "maybe_regional_inductor"]


# Toggled on by ``_maybe_regional_inductor_backend`` when the model compiles
# with a non-inductor backend and its flex regions must be scooped into an
# inductor sub-compile. Read by ``maybe_regional_inductor`` at trace time;
# left False on the default inductor / eager paths so no annotation metadata
# is emitted.
_regional_inductor_enabled: bool = False

# TP group names already registered for symmetric memory, so the per-chunk
# calls under pipeline parallelism register each group exactly once.
_symm_mem_enabled_groups: set[str] = set()


def apply_compile(
    model: nn.Module,
    *,
    compile_config: CompileConfig | None = None,
    tp_mesh=None,
) -> nn.Module:
    """Compile ``model`` according to ``compile_config``; returns the model.

    ``compile_config=None`` is the default run: whole-model inductor compile,
    identical to the plain ``torch.compile(model)`` this replaces. With
    ``per_block=True`` each decoder layer is compiled in place instead and
    the model is returned unwrapped -- either way the caller rebinds the
    result, so a wrapping transform cannot leave a stale module behind.

    ``tp_mesh`` is the tensor-parallel mesh (``parallel_dims``' ``tp``
    axis), consulted only when async TP is configured.
    """
    if compile_config is None:
        compile_config = CompileConfig()

    _maybe_enable_async_tp(compile_config, tp_mesh)

    if _iter_moe_layers(model):
        # Token-choice dispatch sizes its expert splits from the routing,
        # so the compiled graph has data-dependent shapes. Dense models
        # never touch the flag, keeping their trace bitwise unchanged.
        if not has("dynamo_capture_scalar_outputs"):
            raise EnvironmentUnsupportedError(
                "This torch has no torch._dynamo.config.capture_scalar_outputs, "
                "which compiling a token-choice MoE dispatch needs for its "
                "data-dependent shapes. Upgrade torch, or run this model "
                "without compile."
            )
        torch._dynamo.config.capture_scalar_outputs = True
        logger.info("capture_scalar_outputs is enabled (token-choice MoE)")

    backend = _maybe_regional_inductor_backend(model, compile_config.backend)

    if compile_config.per_block:
        for layer_id, block in model.layers.named_children():
            # Module.compile, not torch.compile(block): it sets the compiled
            # call impl in place, so the layer keeps its identity for FSDP
            # and the state dict.
            block.compile(backend=backend, fullgraph=True)
            logger.info(f"Compiling decoder layer {layer_id} (fullgraph)")
        return model

    logger.info(f"Compiling the whole model (backend={backend!r})")
    return torch.compile(model, backend=backend)


def _maybe_enable_async_tp(compile_config: CompileConfig, tp_mesh) -> None:
    """Configure Inductor's async TP pass for the TP mesh.

    Every precondition is checked here and fails loudly: async TP asked for
    without a TP mesh (``tensor_parallel_size = 1`` or a single-process run),
    or a torch too old to carry the pass, are config errors, not situations
    to degrade through.
    """
    if not compile_config.enable_async_tensor_parallel:
        return
    if tp_mesh is None:
        raise ValueError(
            "compile_config.enable_async_tensor_parallel requires "
            "tensor_parallel_size > 1 with a real process group: async TP "
            "pipelines the TP collectives, and there is no TP group here."
        )

    import torch._inductor.config as inductor_config

    if not has("inductor_micro_pipeline_tp"):
        raise EnvironmentUnsupportedError(
            "compile_config.enable_async_tensor_parallel needs "
            "torch._inductor.config._micro_pipeline_tp, which this torch "
            f"({torch.__version__}) does not carry. Upgrade torch, or run "
            "without async TP."
        )
    if not has("symm_mem"):
        raise EnvironmentUnsupportedError(
            "compile_config.enable_async_tensor_parallel needs "
            "torch.distributed._symmetric_memory.enable_symm_mem_for_group, "
            f"which this torch ({torch.__version__}) does not carry. Upgrade "
            "torch, or run without async TP."
        )
    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    group_name = tp_mesh.get_group().group_name
    if group_name not in _symm_mem_enabled_groups:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            enable_symm_mem_for_group(group_name)
        _symm_mem_enabled_groups.add(group_name)

    inductor_config._micro_pipeline_tp = True
    logger.info("Async TP is enabled")


def _maybe_regional_inductor_backend(
    model: nn.Module, backend: str
) -> str | Callable:
    """Wrap ``aot_eager`` so the flex regions are scooped into inductor.

    ``regional_inductor`` lowers just the regions annotated with
    ``compile_with_inductor`` (see ``maybe_regional_inductor`` and the flex
    call site in ``models/hf_wrapper.py``) while the rest stays in
    aot_eager. Only applied for ``aot_eager`` on a model that actually runs
    flex attention: the default inductor backend already lowers flex
    directly, and a model on the sdpa fallback has no flex region to scoop,
    so both keep the unmodified backend.

    Flex has only an inductor lowering. Under any other non-inductor
    backend it would decompose to eager aten ops (no Triton kernel), which
    cannot be transparently scooped -- fail loudly, as upstream does.
    """
    uses_flex = getattr(model, "uses_flex_attention", False)
    if not uses_flex or backend == "inductor":
        return backend

    if backend != "aot_eager":
        raise ValueError(
            f"This model runs flex attention but compile backend {backend!r} "
            "is neither 'inductor' nor 'aot_eager'; the flex region would "
            "decompose to eager aten ops (no Triton kernel). Use 'inductor' "
            "or 'aot_eager'."
        )

    if not has("fx_regional_inductor"):
        raise EnvironmentUnsupportedError(
            "compile backend 'aot_eager' on a flex-attention model needs "
            "torch.fx.passes.regional_inductor to scoop the flex region "
            f"into inductor, which this torch ({torch.__version__}) does "
            "not carry. Upgrade torch, use backend='inductor', or run "
            "without compile."
        )
    from torch._dynamo.backends.common import aot_autograd
    from torch.fx.passes.regional_inductor import regional_inductor

    global _regional_inductor_enabled
    _regional_inductor_enabled = True

    logger.info("regional_inductor is enabled")
    return aot_autograd(fw_compiler=regional_inductor, bw_compiler=regional_inductor)


def maybe_regional_inductor(
    inductor_configs: dict,
) -> contextlib.AbstractContextManager:
    """Context manager that marks the wrapped region for ``regional_inductor``.

    Returns a null context unless regional inductor is enabled (see
    ``_maybe_regional_inductor_backend``). When enabled, the region is tagged
    with ``compile_with_inductor`` so a non-inductor outer compile lowers
    just this region to inductor with ``inductor_configs``.
    """
    if not _regional_inductor_enabled:
        return contextlib.nullcontext()
    import torch.fx.traceback as fx_traceback

    return fx_traceback.annotate(
        {"compile_with_inductor": {"inductor_configs": inductor_configs}}
    )
