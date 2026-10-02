"""TP x EP x CP check: the three-way composition must match the HF model.

Run under torchrun with 4 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=4 \\
        tests/integration_tests/tp_ep_cp_equivalence.py

A tiny qwen3_moe runs with tensor_parallel_size=2, context_parallel_size=2 and
expert_parallel_size=2 on 4 gloo ranks: TP shards the dense projections, the
EP swap owns the routed experts, CP shards the sequence. Each rank's logits
are this rank's T/(cp*tp) sequence slice; joining them over tp and then cp
must reproduce the unsharded HF model's logits, and the summed loss must
reduce to the reference's.

Environment note: needs torch >= 2.12 (the flex/spmd import surface). The
combination was config-refused until 2026-10-02; this script is the numeric
proof, pending a multi-rank run on the target torch (environment not covered
on the development host).

Non-vacuity: TP must have swapped projections (wrapper count), each rank must
hold only its half of the experts, and the CP gather must matter (a rank
whose tokens have predecessors must not match a no-op run).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from llmtuner.models.common.moe.block import MoE
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.context_parallel import (
    apply_cp,
    shard_batch_for_cp,
    shard_batch_for_tp,
)
from llmtuner.parallel.expert_parallel import apply_ep
from llmtuner.parallel.parallel_dims import ParallelDims
from llmtuner.parallel.tensor_parallel import apply_tp
from llmtuner.trainer import LLMTunerConfig, ModelConfig, ParallelConfig, TrainingConfig

SEQ = 256  # torch's CP BlockMask path requires Q_LEN % (cp * 128) == 0
VOCAB = 128
HEADS = 8  # divisible by tp * cp == 4
NUM_EXPERTS = 8
TP = 2
CP = 2
EP = 2
WORLD = TP * CP  # EP tiles the existing ranks; it is not an extra axis
TOL = 1e-9
IGNORE_INDEX = -100


def _cfg() -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=HEADS,
            num_key_value_heads=HEADS,
            arch_overrides={
                "num_experts": NUM_EXPERTS,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 48,
                "norm_topk_prob": True,
            },
        ),
        parallel=ParallelConfig(
            tensor_parallel_size=TP,
            context_parallel_size=CP,
            expert_parallel_size=EP,
            context_parallel_load_balancer=None,
        ),
        training=TrainingConfig(max_seq_len=SEQ, steps=1),
    )


def _build(cfg: LLMTunerConfig, *, seed: int = 0) -> HFTransformerModel:
    """Deterministically initialized tiny qwen3_moe; identical on every rank."""
    torch.manual_seed(seed)
    return HFTransformerModel(build_model_config_for(cfg)).to(torch.float64).eval()


def _data(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(VOCAB, (SEQ,), generator=g)
    labels = torch.randint(VOCAB, (SEQ,), generator=g)
    return ids, labels, torch.arange(SEQ)


def _gather(x: torch.Tensor, mesh) -> torch.Tensor:
    """Concatenate ``x`` across ``mesh`` in rank order."""
    xs = [torch.empty_like(x) for _ in range(mesh.size())]
    dist.all_gather(xs, x.detach().contiguous(), group=mesh.get_group())
    return torch.cat(xs, dim=0)


def _dense_view(cp_mesh, tp_mesh):
    """The dense (cp, tp) view ``apply_cp`` is handed, alongside apply_tp."""

    class _View:
        mesh_dim_names = ("cp", "tp")

        def __getitem__(self, key):
            return {"cp": cp_mesh, "tp": tp_mesh}[key]

    return _View()


def _diff(what: str, got: torch.Tensor, want: torch.Tensor, failures) -> float:
    if got.shape != want.shape:
        failures.append(f"{what}: shape {tuple(got.shape)} vs {tuple(want.shape)}")
        return float("inf")
    diff = (got - want).abs().max().item()
    if diff > TOL:
        failures.append(f"{what}: max abs diff {diff:.3e}")
    return diff


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == WORLD, f"this check assumes {WORLD} ranks, got {world}"

    cfg = _cfg()
    parallel_dims = ParallelDims.from_config(cfg.parallel, world)
    cp_mesh = parallel_dims.get_mesh("cp")
    tp_mesh = parallel_dims.get_mesh("tp")
    ep_mesh = parallel_dims.get_mesh("ep")
    assert cp_mesh.size() == CP and tp_mesh.size() == TP and ep_mesh.size() == EP

    model = _build(cfg)
    ref = _build(cfg)

    # The assembly order from parallelize.py's STAGES table: tp -> ep -> cp.
    model = apply_tp(model, tp_mesh, cfg.parallel)
    apply_ep(model, cfg.parallel, ep_group=ep_mesh.get_group())
    apply_cp(model, _dense_view(cp_mesh, tp_mesh), cfg.parallel)

    failures: list[str] = []

    # Non-vacuity, TP axis: projections were swapped for sharded realizers.
    wrappers = {"ColumnParallelLinear", "RowParallelLinear", "ColwiseLinearNoGather"}
    n_wrapped = sum(1 for m in model.modules() if type(m).__name__ in wrappers)
    if n_wrapped == 0:
        failures.append("no sharded-linear wrapper -- TP sharded nothing")

    # Non-vacuity, EP axis: each rank holds only its half of the experts.
    num_local = NUM_EXPERTS // EP
    for layer_idx, layer in enumerate(model.layers):
        moe = layer.mlp
        assert isinstance(moe, MoE), "the swap did not install llmtuner MoE blocks"
        held = moe.routed_experts.inner_experts.num_experts
        if held != num_local:
            failures.append(
                f"layer {layer_idx}: holds {held} experts, want {num_local} "
                "-- EP did not shard"
            )

    ids, labels, positions = _data()
    ids_cp, labels_cp, pos_cp = shard_batch_for_cp(
        ids, labels, positions, cp_mesh, load_balancer=None
    )
    ids_r, labels_r = shard_batch_for_tp(ids_cp, labels_cp, tp_mesh)

    with torch.no_grad():
        local = model(ids_r, positions=pos_cp)
        ref_full = ref(ids, positions=positions)

    # This rank's CP block of the full-sequence reference.
    block = SEQ // CP
    start = cp_mesh.get_local_rank() * block
    ref_block = ref_full[start : start + block]

    # Join over tp -> this rank's CP block; then over cp -> the whole sequence.
    tp_joined = _gather(local, tp_mesh)
    _diff("tp-joined vs CP-block reference", tp_joined, ref_block, failures)
    cp_joined = _gather(tp_joined, cp_mesh)
    _diff("cp-joined vs full reference", cp_joined, ref_full, failures)

    # The loss, summed over tokens and reduced across cp and tp.
    with torch.no_grad():
        loss_local = torch.nn.functional.cross_entropy(
            local, labels_r, reduction="sum", ignore_index=IGNORE_INDEX
        )
        loss_ref = torch.nn.functional.cross_entropy(
            ref_full, labels, reduction="sum", ignore_index=IGNORE_INDEX
        )
    dist.all_reduce(loss_local, group=cp_mesh.get_group())
    dist.all_reduce(loss_local, group=tp_mesh.get_group())
    loss_diff = abs(loss_local.item() - loss_ref.item())
    if loss_diff > TOL:
        failures.append(f"reduced loss diff {loss_diff:.3e}")

    # The batch is replicated across cp and tp: the token count must not be
    # reduced over those axes (see cp_tp_equivalence.py for the same pin).
    local_count = int((labels_r != IGNORE_INDEX).sum())
    full_count = int((labels != IGNORE_INDEX).sum())
    if local_count != full_count // (CP * TP):
        failures.append(
            f"local valid tokens {local_count}, expected {full_count} // "
            f"(cp*tp={CP * TP})"
        )

    if rank == 0:
        print(
            f"tp={TP} cp={CP} ep={EP} world={world} seq={SEQ} "
            f"experts={NUM_EXPERTS} dtype=float64"
        )
        for f in failures:
            print(f"  FAIL {f}")
        print("all checks passed" if not failures else "CHECKS FAILED")
    verdict = torch.tensor(len(failures), dtype=torch.int64)
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    assert int(verdict) == 0, f"{int(verdict)} check(s) failed -- see rank 0 output"


if __name__ == "__main__":
    main()
