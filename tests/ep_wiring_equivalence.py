"""EP wiring check: ep=2 through ``apply_cp_ep`` must match the HF model.

Run under torchrun with 2 ranks:

    torchrun --nproc_per_node=2 tests/ep_wiring_equivalence.py

Where ``ep_equivalence.py`` checks the dispatcher arithmetic on hand-built
MoEs, this checks the WIRING end to end: a real (tiny, offline, randomly
initialized) qwen3_moe goes through ``apply_cp_ep`` with a 2-rank EP group,
and each rank's forward over its own token shard must reproduce
(a) the unmodified HF model's forward and (b) the EP=1 swap's forward.

Non-vacuity: under EP=2 each rank must hold only half the experts, and the
slice must be THIS rank's slice -- rank 1 holding experts 0-3 instead of 4-7
is exactly the wiring bug the weight check catches.

Forward-only for the EP=2 model: the checks run under gloo on CPU, and the
all-to-all's autograd is not what is being pinned here. The aux-loss
injection (a forward-time accumulation plus a backward-time gradient) is
exercised on the EP=1 swap, where the backward is plain local compute.

Everything runs in float64; routing is fp32 in both implementations (see
tests/test_ep_swap.py), so the tolerance sits above the fp32 routing noise.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig

from hpmesh.models.common.aux_loss import AuxLoss
from hpmesh.models.common.moe import MoE
from hpmesh.models.hf_wrapper import HFTransformerModel
from hpmesh.parallel.cp_ep import apply_cp_ep
from hpmesh.parallel.ep import swap_hf_moe_blocks

NUM_EXPERTS = 8
TOKENS = 40
TOL = 1e-6


class _Cfg:
    """``apply_cp_ep`` reads exactly these two attributes on the EP path."""

    cp = 1
    ep = 2


def _config() -> AutoConfig:
    return AutoConfig.for_model(
        "qwen3_moe",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=256,
    )


def _model(seed: int = 0) -> HFTransformerModel:
    """Deterministically initialized tiny qwen3_moe; identical on every rank."""
    torch.manual_seed(seed)
    return HFTransformerModel(_config()).to(torch.float64).eval()


def _data(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Each rank owns its own token shard -- that is what makes EP a form of
    # parallelism (see tests/ep_equivalence.py for why sharing the shard
    # would double-count).
    g = torch.Generator().manual_seed(1000 + rank)
    ids = torch.randint(128, (TOKENS,), generator=g)
    return ids, torch.arange(TOKENS)


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    ep_mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("ep",))
    ep_group = ep_mesh.get_group()
    failures: list[str] = []

    ref = _model()
    ep1 = _model()
    swap_hf_moe_blocks(ep1)
    ep2 = _model()
    apply_cp_ep(ep2, None, _Cfg(), ep_group=ep_group)

    ids, positions = _data(rank)
    with torch.no_grad():
        ref_logits = ref(ids, positions=positions)
        ep1_logits = ep1(ids, positions=positions)
        ep2_logits = ep2(ids, positions=positions)

    diff_hf = (ep2_logits - ref_logits).abs().max().item()
    diff_ep1 = (ep2_logits - ep1_logits).abs().max().item()
    if diff_hf > TOL:
        failures.append(f"rank {rank}: EP=2 vs HF diff {diff_hf:.3e}")
    if diff_ep1 > TOL:
        failures.append(f"rank {rank}: EP=2 vs EP=1 diff {diff_ep1:.3e}")

    # Non-vacuity: each rank holds only its own half of the experts, and the
    # slice matches the reference weights for exactly those experts.
    num_local = NUM_EXPERTS // world
    for layer_idx, (layer_ep2, layer_ref) in enumerate(
        zip(ep2.layers, ref.layers, strict=True)
    ):
        moe = layer_ep2.mlp
        assert isinstance(moe, MoE)
        grouped = moe.routed_experts.inner_experts
        if grouped.num_experts != num_local:
            failures.append(
                f"rank {rank} layer {layer_idx}: holds {grouped.num_experts} "
                f"experts, want {num_local} -- EP did not shard"
            )
        lo = rank * num_local
        for e in range(num_local):
            hf_w = layer_ref.mlp.experts[lo + e].gate_proj.weight
            if not torch.equal(grouped.w1_EFD[e], hf_w):
                failures.append(
                    f"rank {rank} layer {layer_idx}: local expert {e} is not "
                    f"global expert {lo + e}"
                )

    # Aux loss under training mode: the metric accumulates in the forward and
    # the gradient reaches the router on backward. Run on the EP=1 swap: what
    # is pinned here is the injection, which is dispatcher-independent.
    AuxLoss.set_step_denominator(torch.tensor(float(TOKENS - 1)))
    ep1.train()
    ep1(ids, positions=positions).sum().backward()
    aux = ep1.layers[0].mlp.router.aux_loss
    if aux.instance_acc.item() <= 0:
        failures.append(f"rank {rank}: aux loss metric not accumulated")
    gate_grad = ep1.layers[0].mlp.router.gate.weight.grad
    if gate_grad is None or gate_grad.abs().max().item() == 0:
        failures.append(f"rank {rank}: no gradient reached the router")

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"tokens={TOKENS} experts={NUM_EXPERTS} ep_size={world} tol={TOL:.0e}")
        print(f"EP=2 vs HF  max abs diff = {diff_hf:.3e}")
        print(f"EP=2 vs EP=1 max abs diff = {diff_ep1:.3e}")
        print(f"aux metric (rank 0)       = {aux.instance_acc.item():.4f}")
        print(f"failed ranks = {int(local_ok.item())}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "EP wiring equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
