"""hpmesh configuration: grouped dataclasses composed into one HybridMeshConfig.

Design notes (see README): grouped by concern (model / parallel / optimizer /
training), then COMPOSED -- not mixed in via multiple inheritance -- so each
group's __post_init__ validation runs automatically via default_factory, with no
fragile manual chaining. The single entry point is ``HybridMeshConfig``; CLI/YAML
parsing is done by ``transformers.HfArgumentParser`` in ``hpmesh.train``.

Kept deliberately small for the learning path: one optimizer (adamw), one LR
schedule, deterministic seeding. Add knobs only when a learning step needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from transformers import AutoConfig

from hpmesh.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "HybridMeshConfig",
    "ModelArguments",
    "OptimizerArguments",
    "ParallelArguments",
    "TrainingArguments",
]


@dataclass
class ModelArguments:
    """Model architecture. None fields are auto-filled from the HF config.

    Offline path: when ``model_name_or_path`` has no '/', build a tiny local model
    with AutoConfig.for_model using the explicit sizes below. Online path: a hub id
    ("org/name") pulls real architecture via AutoConfig.from_pretrained.
    """

    model_name_or_path: str = field(
        default="llama",
        metadata={"help": "HF architecture name (offline) or hub id 'org/name' (online)"},
    )
    vocab_size: int = field(default=128, metadata={"help": "Vocabulary size"})
    hidden_size: int = field(default=64, metadata={"help": "Hidden dimension"})
    intermediate_size: int = field(default=128, metadata={"help": "FFN inner dimension"})
    num_hidden_layers: int = field(default=2, metadata={"help": "Number of decoder layers"})
    num_attention_heads: int = field(default=4, metadata={"help": "Number of attention heads"})
    num_key_value_heads: int = field(default=4, metadata={"help": "Number of KV heads (GQA)"})


@dataclass
class ParallelArguments:
    """Hybrid-parallelism degrees. world_size = dp * cp * tp * pp (ep stays 1 until MoE).

    dp = -1 means "derive from world_size" once the other degrees are known.
    """

    dp: int = field(default=-1, metadata={"help": "Data-parallel (FSDP) degree; -1 = derive"})
    tp: int = field(default=1, metadata={"help": "Tensor-parallel degree"})
    pp: int = field(default=1, metadata={"help": "Pipeline-parallel degree"})
    cp: int = field(default=1, metadata={"help": "Context-parallel degree"})
    ep: int = field(default=1, metadata={"help": "Expert-parallel degree (MoE)"})
    backend: str = field(default="nccl", metadata={"help": "Distributed backend"})

    def __post_init__(self) -> None:
        for name in ("tp", "pp", "cp", "ep"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.dp < 1 and self.dp != -1:
            raise ValueError(f"dp must be >= 1 or -1 (derive), got {self.dp}")
        if self.backend not in {"nccl", "gloo", "hccl"}:
            raise ValueError(f"backend must be one of {{nccl, gloo, hccl}}, got {self.backend}")

    def non_dp_degrees(self) -> int:
        """Product of the non-data-parallel degrees (dp is carved out of the rest)."""
        return self.tp * self.pp * self.cp * self.ep

    def derive_dp(self, world_size: int) -> int:
        """Resolve the data-parallel degree against world_size."""
        denom = self.non_dp_degrees()
        if self.dp == -1:
            if world_size % denom != 0:
                raise ValueError(
                    f"world_size={world_size} not divisible by tp*pp*cp*ep={denom}"
                )
            return world_size // denom
        if self.dp * denom != world_size:
            raise ValueError(
                f"dp*tp*pp*cp*ep = {self.dp * denom} != world_size={world_size}"
            )
        return self.dp


@dataclass
class OptimizerArguments:
    """Optimizer (adamw only for now -- the learning path needs just one)."""

    learning_rate: float = field(default=3e-4, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay"})


@dataclass
class TrainingArguments:
    """Training loop hyperparameters and reproducibility."""

    global_batch_size: int = field(
        default=8, metadata={"help": "Sequences per step across ALL DP ranks"}
    )
    max_seq_len: int = field(default=64, metadata={"help": "Sequence length"})
    steps: int = field(default=20, metadata={"help": "Number of optimizer steps"})
    seed: int = field(default=42, metadata={"help": "Base RNG seed"})
    compile: bool = field(default=False, metadata={"help": "torch.compile the model"})
    deterministic: bool = field(
        default=True,
        metadata={"help": "Deterministic algorithms -- required for bit-exact comparison"},
    )
    log_freq: int = field(default=1, metadata={"help": "Log every N steps"})

    def __post_init__(self) -> None:
        if self.global_batch_size < 1:
            raise ValueError(f"global_batch_size must be >= 1, got {self.global_batch_size}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")


@dataclass
class HybridMeshConfig:
    """Single entry point: composes the argument groups (no multiple inheritance).

    Each group validates itself in its own __post_init__ (run by default_factory).
    """

    model: ModelArguments = field(default_factory=ModelArguments)
    parallel: ParallelArguments = field(default_factory=ParallelArguments)
    optimizer: OptimizerArguments = field(default_factory=OptimizerArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)

    def __post_init__(self) -> None:
        # Cross-group check: CP must divide the sequence length.
        if self.training.max_seq_len % self.parallel.cp != 0:
            raise ValueError(
                f"max_seq_len ({self.training.max_seq_len}) must be divisible by "
                f"cp ({self.parallel.cp})"
            )

    # -- Flat view: lets mesh/bundle/trainer read cfg.dp / cfg.lr / ... uniformly. --
    @property
    def hf_model(self) -> str: return self.model.model_name_or_path
    @property
    def vocab_size(self) -> int: return self.model.vocab_size
    @property
    def hidden_size(self) -> int: return self.model.hidden_size
    @property
    def intermediate_size(self) -> int: return self.model.intermediate_size
    @property
    def num_hidden_layers(self) -> int: return self.model.num_hidden_layers
    @property
    def num_attention_heads(self) -> int: return self.model.num_attention_heads
    @property
    def num_key_value_heads(self) -> int: return self.model.num_key_value_heads
    @property
    def dp(self) -> int: return self.parallel.dp
    @property
    def tp(self) -> int: return self.parallel.tp
    @property
    def pp(self) -> int: return self.parallel.pp
    @property
    def cp(self) -> int: return self.parallel.cp
    @property
    def ep(self) -> int: return self.parallel.ep
    @property
    def lr(self) -> float: return self.optimizer.learning_rate
    @property
    def weight_decay(self) -> float: return self.optimizer.weight_decay
    @property
    def global_batch_size(self) -> int: return self.training.global_batch_size
    @property
    def max_seq_len(self) -> int: return self.training.max_seq_len
    @property
    def steps(self) -> int: return self.training.steps
    @property
    def seed(self) -> int: return self.training.seed
    @property
    def compile(self) -> bool: return self.training.compile
    @property
    def deterministic(self) -> bool: return self.training.deterministic
    @property
    def log_freq(self) -> int: return self.training.log_freq

    def derive_dp(self, world_size: int) -> int:
        """Flat passthrough so callers use cfg.derive_dp(world_size) uniformly."""
        return self.parallel.derive_dp(world_size)

    def auto_fill_model(self) -> None:
        """Fill model arch fields from a HF hub config when given a hub id.

        Called explicitly (not in __post_init__) so offline runs never hit the network.
        """
        mp = self.model.model_name_or_path
        if mp.count("/") != 1:
            return  # offline architecture name; explicit sizes are authoritative
        try:
            hf_config = AutoConfig.from_pretrained(mp)
        except Exception as e:  # noqa: BLE001 - surface as a warning, keep explicit values
            logger.warning("Could not load AutoConfig for '%s': %s", mp, e)
            return
        m = self.model
        m.vocab_size = getattr(hf_config, "vocab_size", m.vocab_size)
        m.hidden_size = getattr(hf_config, "hidden_size", m.hidden_size)
        m.intermediate_size = getattr(hf_config, "intermediate_size", m.intermediate_size)
        m.num_hidden_layers = getattr(hf_config, "num_hidden_layers", m.num_hidden_layers)
        m.num_attention_heads = getattr(hf_config, "num_attention_heads", m.num_attention_heads)
        m.num_key_value_heads = getattr(hf_config, "num_key_value_heads", m.num_key_value_heads)
