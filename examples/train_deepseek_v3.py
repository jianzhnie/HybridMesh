"""DeepSeek-V3 training: a routed MoE behind multi-head latent attention.

Same shape as ``train_qwen3.py`` -- build a config, hand it to ``Trainer`` -- with
one wrinkle that is the point of this example. DeepSeek-V3's architecture needs
settings that ``ModelConfig`` has no field for: how many experts to route to
(``n_routed_experts``), how wide each expert is (``moe_intermediate_size``), and
the MLA compression ranks (``q_lora_rank``, ``kv_lora_rank``). Those go in
``arch_overrides``, which is applied on top of the explicit sizes.

That is not cosmetic. ``AutoConfig.for_model("deepseek_v3")`` fills every field
this config does not set from the *published* DeepSeek-V3 hyperparameters, so
leaving ``n_routed_experts`` out builds 256 experts and a 671B-shaped model
rather than a toy. The overrides are what make the architecture small.

Run it (from the repo root; see ``train_qwen3.py`` for why ``-m``):
    python -m examples.train_deepseek_v3

For real weights, point ``model_name_or_path`` at a hub id
("deepseek-ai/DeepSeek-V3"): the Hub's config.json carries all of the settings
below, and ``arch_overrides`` is ignored so it cannot disagree with the weights.
"""

from __future__ import annotations

from hpmesh import Trainer
from hpmesh.config import (
    HybridMeshConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)


def deepseek_v3_config() -> HybridMeshConfig:
    """A tiny offline DeepSeek-V3: 2 dense layers, then 2 routed MoE layers."""
    return HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="deepseek_v3",
            # The dense-decoder widths. Everything MoE- or MLA-specific is below.
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            arch_overrides={
                # Multi-head latent attention: Q and K/V are compressed through
                # low-rank projections, and the per-head dims are not
                # hidden_size / num_attention_heads. Sized to the tiny width
                # above, not to the published 128 / 64 / 128.
                "q_lora_rank": 32,
                "kv_lora_rank": 16,
                "qk_nope_head_dim": 16,
                "qk_rope_head_dim": 8,
                "v_head_dim": 16,
                # Routed MoE. Without these the architecture defaults to 256
                # experts per layer at 2048 wide, which is the real model.
                "n_routed_experts": 4,
                "num_experts_per_tok": 1,
                "n_shared_experts": 1,
                "moe_intermediate_size": 64,
                "n_group": 1,  # one group: plain top-k, no group-limited routing
                "topk_group": 1,
                # Layers 0-1 dense, 2-3 routed. This split is what makes the
                # model a DeepSeek-V3 rather than a plain decoder with MoE on
                # every layer.
                "first_k_dense_replace": 2,
            },
        ),
        parallel=ParallelConfig(data_parallel_shard_size=-1),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=8,
            max_seq_len=32,
            steps=20,
            seed=42,
            deterministic=True,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def main() -> None:
    Trainer(deepseek_v3_config()).train()


if __name__ == "__main__":
    main()
