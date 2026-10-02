"""llmtuner's exception hierarchy: three fail-fast semantics, told apart by type.

Zero-dependency by design -- every layer (config, models, parallel, trainer)
imports this module, so it must import nothing itself.

The three leaf classes answer different operator questions:

* ``ConfigError`` -- "the config is wrong; fix a flag or a field value."
  Field-level validation and cross-field contradictions, raised in
  ``llmtuner/config/*``'s ``__post_init__`` (and equivalent guard points).
* ``UnsupportedCombinationError`` -- "each half is legal alone; the
  combination is rejected." The composition-matrix refusals: PP x AC,
  PP x validation, shared-expert x tp, ulysses x load-balancer, an HF MoE
  layout the swap/TP path does not express. Not a dependency problem -- the
  combination is unverified or meaningless, and the fix is to change the
  configuration, not the environment.
* ``EnvironmentUnsupportedError`` -- "this build/host lacks the dependency."
  A torch too old for a knob (``_micro_pipeline_tp``,
  ``activation_memory_budget``, ...), an unvendored CUDA-only backend
  (deepep/hybridep). The message must carry the unlock condition (which
  torch version, which package). Optional *packages* keep raising plain
  ``ImportError`` instead (Python convention: renderers, torchao,
  torchvision) -- their install hint lives in the message.

Each leaf multiply-inherits the builtin its guard points raised before this
hierarchy existed (``ValueError`` / ``NotImplementedError``), so existing
``pytest.raises`` assertions and callers keep catching what they caught.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "EnvironmentUnsupportedError",
    "LLMTunerError",
    "UnsupportedCombinationError",
]


class LLMTunerError(Exception):
    """Base class for every error llmtuner raises deliberately."""


class ConfigError(LLMTunerError, ValueError):
    """Configuration validation failed; fix the config, not the environment."""


class UnsupportedCombinationError(LLMTunerError, NotImplementedError):
    """A combination of individually valid options is rejected."""


class EnvironmentUnsupportedError(LLMTunerError, NotImplementedError):
    """The build or host lacks a required dependency; message names the unlock."""
