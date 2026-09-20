"""Ulysses CP wiring check: all-to-all CP must match the full-sequence run.

Run under torchrun with 2 ranks:

    torchrun --nproc_per_node=2 tests/cp_ulysses_equivalence.py

Where ``cp_wiring_equivalence.py`` pins the kv_allgather strategy, this pins
the ulysses one end to end: a tiny offline qwen3 goes through ``apply_cp``
with ``context_parallel_strategy="ulysses"``, gets fed contiguously CP-sharded
inputs (ulysses forbids the load balancer -- the all-to-all reassembles the
sequence by concatenating rank shards in rank order), and every rank's forward
must reproduce the single-rank full-sequence forward exactly.

Three layers of checks:

* the all-to-all itself: ``(b, h, s/cp, d) -> (b, h/cp, s, d)`` must land each
  rank on its own head slice of the full sequence, the inverse swap must
  restore the shard exactly, and the backward must place gradients on the
  sending rank's token shard (a pass-through backward would silently misplace
  them);
* the wiring: sharded logits gathered in rank order equal the reference
  full-sequence logits, and the summed token losses agree. Non-vacuity:
  attending each shard against itself must differ materially, so a kernel that
  secretly skipped the all-to-all could not pass;
* the refusals: heads not divisible by cp, ulysses + load balancer, and
  ulysses + packed sequences must all raise rather than compute a confidently
  wrong answer.

Forward-only, like ``cp_wiring_equivalence.py``: torch's flex attention has no
CPU backward. Everything runs in float64 so the comparison is arithmetic, not
noise.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config_for
from hpmesh.parallel.context_parallel import apply_cp, shard_batch_for_cp
from hpmesh.parallel.context_parallel.cp_kernel import (
    CPFlexKernel,
    _HeadToSeq,
    _SeqToHead,
)
from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)

SEQ = 256  # torch's CP BlockMask path requires Q_LEN % (cp * 128) == 0
VOCAB = 128

# float64 comparisons: wiring differences are O(1), arithmetic noise is O(1e-13).
TOL = 1e-9


def _cfg(num_kv_heads: int = 4) -> HybridMeshConfig:
    return HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=num_kv_heads,
        ),
        parallel=ParallelConfig(
            context_parallel_degree=2,
            context_parallel_strategy="ulysses",
            context_parallel_load_balancer=None,
            backend="gloo",
        ),
        training=TrainingConfig(max_seq_len=SEQ, steps=1),
    )


def _build_model(cfg: HybridMeshConfig, *, flex: bool, seed: int = 0):
    """Deterministically initialized tiny qwen3; identical on every rank."""
    torch.manual_seed(seed)
    model = HFTransformerModel(build_model_config_for(cfg)).to(torch.float64)
    if flex:
        # The machine-level fallback picks sdpa off CUDA; CP needs the flex
        # path, so the test opts in explicitly.
        model.model.config._attn_implementation = "flex_torchtitan"
    return model.eval()


def _data(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(VOCAB, (SEQ,), generator=g)
    labels = torch.randint(VOCAB, (SEQ,), generator=g)
    positions = torch.arange(SEQ)
    return ids, labels, positions


def _causal_additive_mask(t: int) -> torch.Tensor:
    """A dense causal additive (1, 1, t, t) float64 mask."""
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
    additive = torch.zeros(t, t, dtype=torch.float64)
    return additive.masked_fill(~causal, float("-inf"))[None, None]


def _ref_forward(ref, ids, positions):
    """Full-sequence forward through HF's own sdpa path with a dense mask."""
    hidden = ref.model.model(
        ids.unsqueeze(0),
        position_ids=positions.unsqueeze(0),
        attention_mask=_causal_additive_mask(ids.shape[0]),
        use_cache=False,
    ).last_hidden_state.squeeze(0)
    return ref.model.lm_head(hidden)


def _gather_cp(x_local: torch.Tensor, cp_mesh) -> torch.Tensor:
    """Concatenate a tensor across CP ranks, in rank order."""
    xs = [torch.empty_like(x_local) for _ in range(cp_mesh.size())]
    dist.all_gather(xs, x_local.detach().contiguous(), group=cp_mesh.get_group())
    return torch.cat(xs, dim=0)


def _run_ulysses(
    name: str,
    num_kv_heads: int,
    mesh,
    failures: list[str],
) -> dict[str, float]:
    cfg = _cfg(num_kv_heads)
    model = _build_model(cfg, flex=True)
    ref = _build_model(cfg, flex=False)
    apply_cp(model, mesh, cfg)

    ids, labels, positions = _data()
    ids_sh, labels_sh, pos_sh = shard_batch_for_cp(
        ids, labels, positions, mesh["cp"], load_balancer=None
    )

    with torch.no_grad():
        logits_local = model(ids_sh, positions=pos_sh)
        ref_logits = _ref_forward(ref, ids, positions)

    gathered = _gather_cp(logits_local, mesh["cp"])
    logit_diff = (gathered - ref_logits).abs().max().item()
    if logit_diff > TOL:
        failures.append(f"{name}: logits max abs diff {logit_diff:.3e}")

    # Non-vacuity: shard-against-itself attention must differ materially.
    with torch.no_grad():
        local_only = _ref_forward(ref, ids_sh, pos_sh)
    noop_diff = (logits_local - local_only).abs().max().item()
    # With a contiguous causal split, rank 0's shard genuinely attends only
    # within itself; the check lands on the ranks whose tokens need the swap.
    if mesh["cp"].get_local_rank() > 0 and noop_diff < 1e-3:
        failures.append(f"{name}: no-op all-to-all matches ({noop_diff:.3e}), vacuous")

    # The loss sums over tokens, so the CP ranks' summed losses all-reduced
    # over the CP group must equal the reference's.
    with torch.no_grad():
        loss_local = F.cross_entropy(logits_local, labels_sh, reduction="sum")
        loss_ref = F.cross_entropy(ref_logits, labels, reduction="sum")
    dist.all_reduce(loss_local, group=mesh["cp"].get_group())
    loss_diff = abs(loss_local.item() - loss_ref.item())
    if loss_diff > TOL:
        failures.append(f"{name}: loss diff {loss_diff:.3e}")

    return {"logits": logit_diff, "loss": loss_diff}


def _check_all_to_all(cp_mesh, failures: list[str]) -> dict[str, float]:
    """The token<->head swap: placement, round trip, and backward placement."""
    rank = cp_mesh.get_local_rank()
    cp = cp_mesh.size()
    group = cp_mesh.get_group()
    b, h, s, d = 2, 4, 16, 8
    g = torch.Generator().manual_seed(0)
    full = torch.randn(b, h, s, d, generator=g, dtype=torch.float64)
    w_full = torch.randn(b, h, s, d, generator=g, dtype=torch.float64)

    # The contiguous token shard (ulysses runs without a load balancer).
    lo, hi = rank * s // cp, (rank + 1) * s // cp
    shard = full[:, :, lo:hi].clone().requires_grad_(True)

    swapped = _SeqToHead.apply(shard, group)
    # Full sequence, this rank's head slice -- anything else is a misplacement.
    head_lo, head_hi = rank * h // cp, (rank + 1) * h // cp
    want = full[:, head_lo:head_hi]
    fwd_diff = (swapped - want).abs().max().item()
    if fwd_diff > TOL:
        failures.append(f"all-to-all forward: max abs diff {fwd_diff:.3e}")
    if swapped.shape != (b, h // cp, s, d):
        failures.append(f"all-to-all forward: shape {tuple(swapped.shape)}")

    # Backward: grad of (swapped * w_slice).sum() must land on the sending
    # rank's token shard of w -- a pass-through backward would leave it on the
    # head slice instead.
    (swapped * w_full[:, head_lo:head_hi]).sum().backward()
    grad_diff = (shard.grad - w_full[:, :, lo:hi]).abs().max().item()
    if grad_diff > TOL:
        failures.append(f"all-to-all backward: grad diff {grad_diff:.3e}")

    restored = _HeadToSeq.apply(swapped.detach(), group)
    rt_diff = (restored - shard.detach()).abs().max().item()
    if rt_diff > TOL:
        failures.append(f"all-to-all round trip: {rt_diff:.3e}")

    return {"forward": fwd_diff, "backward": grad_diff, "round_trip": rt_diff}


def _check_refusals(mesh, failures: list[str]) -> None:
    """Every misconfiguration must raise, not compute a wrong answer."""

    def _expect(exc_type, substring, what, fn):
        try:
            fn()
            failures.append(f"{what}: no error raised")
        except exc_type as e:
            if substring not in str(e):
                failures.append(f"{what}: wrong error: {e}")

    cp_mesh = mesh["cp"]

    # Unknown strategies still refuse at kernel construction.
    _expect(
        NotImplementedError,
        "not wired",
        "unknown strategy",
        lambda: CPFlexKernel(cp_mesh=cp_mesh, strategy="ring"),
    )

    # Config-level validation of the strategy field and the ulysses /
    # load-balancer combination.
    _expect(
        ValueError,
        "context_parallel_strategy",
        "bad strategy value",
        lambda: ParallelConfig(context_parallel_strategy="ring"),
    )
    _expect(
        ValueError,
        "load_balancer=None",
        "ulysses + headtail",
        lambda: ParallelConfig(
            context_parallel_degree=2,
            context_parallel_strategy="ulysses",
            context_parallel_load_balancer="headtail",
        ),
    )

    # The kernel's forward guard: heads must divide the CP degree. This raises
    # before any collective, so both ranks fail uniformly.
    kernel = CPFlexKernel(cp_mesh=cp_mesh, strategy="ulysses")
    bad = torch.randn(1, 3, 8, 4, dtype=torch.float64)
    _expect(
        ValueError,
        "does not divide evenly",
        "odd head count",
        lambda: kernel(bad, bad, bad, module=None),
    )

    # Attach-time guards: KV heads not divisible by cp, and packed sequences.
    cfg = _cfg(num_kv_heads=1)
    model = _build_model(cfg, flex=True)
    _expect(
        ValueError,
        "num_key_value_heads",
        "attach with indivisible kv heads",
        lambda: apply_cp(model, mesh, cfg),
    )
    model = _build_model(_cfg(), flex=True)
    # As in a packed run: ``build_model_config_for`` derives this from the
    # corpus, and the attach guard below reads it.
    model.model.config.attn_mask_type = "block_causal"
    _expect(
        ValueError,
        "packed",
        "attach with packed sequences",
        lambda: apply_cp(model, mesh, _cfg()),
    )


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("cp",))
    failures: list[str] = []
    stats: dict[str, dict[str, float]] = {}

    stats["all_to_all"] = _check_all_to_all(mesh["cp"], failures)
    stats["ulysses/mha"] = _run_ulysses("ulysses/mha", 4, mesh, failures)
    stats["ulysses/gqa"] = _run_ulysses("ulysses/gqa", 2, mesh, failures)
    _check_refusals(mesh, failures)

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"seq={SEQ} cp_size={world} dtype=float64 tol={TOL:.0e}")
        a2a = stats["all_to_all"]
        print(
            f"all-to-all fwd={a2a['forward']:.3e} bwd={a2a['backward']:.3e} "
            f"round_trip={a2a['round_trip']:.3e}"
        )
        for name in ("ulysses/mha", "ulysses/gqa"):
            s = stats[name]
            print(f"{name:20s} logits={s['logits']:.3e} loss={s['loss']:.3e}")
        print(f"failed ranks   = {int(local_ok.item())}")
        if failures:
            for f in failures:
                print(f"  FAIL {f}")
        else:
            print("all checks passed")

    assert local_ok.item() == 0, "CP ulysses equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
