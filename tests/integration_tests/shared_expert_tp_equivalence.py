"""shared-expert x TP check: sharded shared expert equals the unsharded model.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \\
        tests/integration_tests/shared_expert_tp_equivalence.py

A tiny offline DeepSeek-V3 (with a shared expert) runs apply_tp at tp=2.
The shared expert shards featurewise with no collectives -- inside the MoE
sequence boundary the token stream is already gathered, so gate/up shard on
the output features and down on the input features, leaving a partial output
that the boundary reduce-scatter sums. The whole model's logits must match
the unsharded reference computed in-process before apply_tp.

Environment note: needs torch >= 2.12 and a transformers that builds
deepseek_v3 offline. Written 2026-10-02 with the shared-expert x TP unlock;
environment not covered on the development host -- awaiting a multi-rank run
on the target torch.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from transformers import AutoConfig

from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.tensor_parallel import apply_tp
from llmtuner.trainer import ParallelConfig

VOCAB = 128
TOKENS = 64
TOL = 1e-9


def _config() -> AutoConfig:
    return AutoConfig.for_model(
        "deepseek_v3",
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        max_position_embeddings=256,
        experts_implementation="eager",
    )


def _model() -> HFTransformerModel:
    torch.manual_seed(0)
    return HFTransformerModel(_config()).to(torch.float64).eval()


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    from torch.distributed.device_mesh import init_device_mesh

    tp_mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("tp",))["tp"]

    model = _model()
    ref = _model()

    ids = torch.randint(VOCAB, (TOKENS,), generator=torch.Generator().manual_seed(7))
    positions = torch.arange(TOKENS)
    with torch.no_grad():
        ref_logits = ref(ids, positions=positions)

    apply_tp(model, tp_mesh, ParallelConfig(tensor_parallel_size=world))

    failures: list[str] = []

    # Non-vacuity: the shared expert's weights are halved on every rank.
    sparse_layers = [
        layer
        for layer in model.layers
        if isinstance(getattr(layer.mlp, "shared_experts", None), torch.nn.Module)
    ]
    if not sparse_layers:
        failures.append("no shared expert found to shard -- check is vacuous")
    for layer in sparse_layers:
        gate = layer.mlp.shared_experts.gate_proj
        ref_gate = ref.layers[model.layers.index(layer)].mlp.shared_experts.gate_proj
        if gate.weight.shape[0] * world != ref_gate.weight.shape[0]:
            failures.append(
                f"rank {rank}: shared gate rows {gate.weight.shape[0]}, "
                f"reference {ref_gate.weight.shape[0]} -- not sharded"
            )

    with torch.no_grad():
        local = model(ids, positions=positions)
        full = torch.cat(
            [torch.empty_like(local) for _ in range(world)], dim=-2
        )
        dist.all_gather_into_tensor(full, local, group=tp_mesh.get_group())
    diff = (full - ref_logits).abs().max().item()
    if diff > TOL:
        failures.append(f"rank {rank}: tp-joined logits diff {diff:.3e}")

    if rank == 0:
        print(f"shared-expert tp={world} tokens={TOKENS} tol={TOL:.0e}")
        print(f"tp-joined vs reference max diff = {diff:.3e}")
        for f in failures:
            print(f"  FAIL {f}")
        print("all checks passed" if not failures else "CHECKS FAILED")
    verdict = torch.tensor(len(failures), dtype=torch.int64)
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    assert int(verdict) == 0, f"{int(verdict)} check(s) failed -- see rank 0 output"


if __name__ == "__main__":
    main()
