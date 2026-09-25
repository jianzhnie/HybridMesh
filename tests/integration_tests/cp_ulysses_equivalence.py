"""Ulysses CP wiring check: all-to-all CP must match the full-sequence run.

Run under torchrun with 2 ranks:

    torchrun --nproc_per_node=2 tests/cp_ulysses_equivalence.py

Where ``cp_wiring_equivalence.py`` pins the kv_allgather strategy, this pins
the ulysses one end to end: a tiny offline qwen3 goes through ``apply_cp``
with ``context_parallel_strategy="ulysses"``, gets fed contiguously CP-sharded
inputs (ulysses forbids the load balancer -- it attends the full sequence in
whatever order the all-to-all delivers, and the causal mask it rebuilds is
only the right mask for one unpermuted order), and every rank's forward
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
* the refusals: heads not divisible by cp and ulysses + load balancer must
  raise rather than compute a confidently wrong answer, while ulysses + packed
  sequences must attach cleanly (varlen coverage lives in
  ``cp_ulysses_varlen_equivalence.py``);
* the asymmetry behind that second refusal: kv_allgather with a load balancer
  is *correct* (the rearrangement lands in the mask too, so it cancels), while
  ulysses with one reproduces the model run over the rearranged corpus. That
  difference is the entire reason the refusal is written for ulysses and not
  for the strategy that is actually allowed to take a balancer.

Forward-only, like ``cp_wiring_equivalence.py``: torch's flex attention has no
CPU backward. Everything runs in float64 so the comparison is arithmetic, not
noise.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.attention.flex_attention import create_block_mask

from hpmesh.models.common.masks import get_causal_mask_mod
from hpmesh.models.hf_factory import build_model_config_for
from hpmesh.models.hf_wrapper import HFTransformerModel
from hpmesh.parallel.context_parallel import (
    apply_cp,
    shard_attention_mask_for_cp,
    shard_batch_for_cp,
)
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
from hpmesh.utils.batch_invariant import is_in_batch_invariant_mode

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
            context_parallel_size=2,
            context_parallel_strategy="ulysses",
            context_parallel_load_balancer=None,
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


def _headtail_indices(seq: int, cp: int) -> torch.Tensor:
    """The permutation a headtail balancer applies, matching torch's own.

    Reimplemented rather than imported so the expected rearrangement is stated
    here; ``_HeadTailLoadBalancer._generate_indices`` is the definition it
    follows (chunk r pairs with chunk 2*cp-1-r).
    """
    chunk = seq // (cp * 2)
    chunks = torch.arange(seq).view(cp * 2, chunk)
    head = torch.arange(cp)
    return torch.stack([chunks[head], chunks[2 * cp - 1 - head]], dim=1).reshape(-1)


def _run_ulysses(
    name: str,
    num_kv_heads: int,
    mesh,
    failures: list[str],
) -> dict[str, float]:
    cfg = _cfg(num_kv_heads)
    model = _build_model(cfg, flex=True)
    ref = _build_model(cfg, flex=False)
    apply_cp(model, mesh, cfg.parallel)

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
            context_parallel_size=2,
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

    # Attach-time guard: KV heads not divisible by cp. Packed sequences are no
    # longer refused -- ulysses x varlen is supported (the wrapper passes the
    # document mask full-length) -- so the packed attach must now succeed and
    # latch the strategy for the wrapper's mask handling.
    cfg = _cfg(num_kv_heads=1)
    model = _build_model(cfg, flex=True)
    _expect(
        ValueError,
        "num_key_value_heads",
        "attach with indivisible kv heads",
        lambda: apply_cp(model, mesh, cfg.parallel),
    )
    model = _build_model(_cfg(), flex=True)
    # As in a packed run: ``build_model_config_for`` derives this from the
    # corpus. Attaching must NOT raise -- the varlen/packed coverage lives in
    # cp_ulysses_varlen_equivalence.py.
    model.model.config.attn_mask_type = "block_causal"
    try:
        apply_cp(model, mesh, _cfg().parallel)
    except Exception as e:  # noqa: BLE001 - any raise here is the failure
        failures.append(f"attach with packed sequences raised: {e}")
    if model._cp_strategy != "ulysses":
        failures.append(
            f"packed ulysses attach latched strategy {model._cp_strategy!r}"
        )


def _check_ulysses_under_a_load_balancer(mesh, failures: list[str]) -> dict[str, float]:
    """Why the load-balancer refusal is written for ulysses and not for the other.

    The pairing is refused *only* for ulysses, and that asymmetry is easy to
    misread as arbitrary -- particularly because both strategies concatenate
    rank shards in rank order, which sounds like the same constraint for each.
    It is not the same, and the difference is what this pins:

    * kv_allgather is fine with a load balancer, and ``cp_wiring_equivalence.py``
      asserts that end to end. The all-gather inverts the rearrangement, so the
      full sequence it assembles is the one the mask was built and Q-sharded in,
      and the rearrangement cancels between the queries and the mask;
    * ulysses has no such inversion. Every rank attends the full sequence in
      whatever order the shards arrived, against a causal mask rebuilt from the
      length alone. That mask describes the unpermuted order, so the attention
      runs over the rearranged corpus -- and nothing raises. No shape, no
      finiteness check, no loss spike: the model that comes out is the model
      trained on ``ids[perm]``.

    Driven directly, because this is precisely the configuration ``apply_cp``
    refuses to build, and asserted against the *permuted* corpus rather than
    against the truth: agreeing with the wrong run is what identifies the
    mechanism, where merely drifting from the right one would not.
    """
    cp = mesh["cp"].size()
    rank = mesh["cp"].get_local_rank()
    ref = _build_model(_cfg(), flex=False)
    ids, labels, positions = _data()
    perm = _headtail_indices(SEQ, cp)

    with torch.no_grad():
        truth = _ref_forward(ref, ids, positions)
        permuted = _ref_forward(ref, ids[perm], positions[perm])

    lo, hi = rank * (SEQ // cp), (rank + 1) * (SEQ // cp)
    # What attending the rearrangement gives, at the rows this rank holds.
    wrong = permuted[lo:hi]
    gap = (truth[perm[lo:hi]] - wrong).abs().max().item()
    if gap < 1e-3:
        failures.append(f"ulysses + headtail: permutation is a no-op ({gap:.3e})")

    ids_ht, _, pos_ht = shard_batch_for_cp(
        ids, labels, positions, mesh["cp"], load_balancer="headtail"
    )
    # The Q-sharded mask the wrapper would hand the kernel: full-length, then
    # rearranged to match the tokens this rank holds.
    full_mask = create_block_mask(
        get_causal_mask_mod(),
        1,
        None,
        SEQ,
        SEQ,
        device=torch.device("cpu"),
        BLOCK_SIZE=128,
        separate_full_blocks=not is_in_batch_invariant_mode(),
    )
    sharded_mask = shard_attention_mask_for_cp(full_mask, mesh["cp"], "headtail")

    model = _build_model(_cfg(), flex=True)
    model.set_cp_mesh(mesh["cp"], load_balancer="headtail")
    for layer in model.layers:
        layer.self_attn._titan_flex_kernel = CPFlexKernel(
            cp_mesh=mesh["cp"], strategy="ulysses"
        )
    with torch.no_grad():
        out = model(ids_ht, positions=pos_ht, attention_masks=sharded_mask)

    diff = (out - wrong).abs().max().item()
    if diff > TOL:
        failures.append(
            f"ulysses + headtail: {diff:.3e} from the permuted-corpus run, "
            "expected the two to agree exactly"
        )

    return {"vs_permuted": diff, "permutation_gap": gap}


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
    stats["ulysses_load_balancer"] = _check_ulysses_under_a_load_balancer(
        mesh, failures
    )
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
        lb = stats["ulysses_load_balancer"]
        print(
            f"ulysses + headtail: vs permuted={lb['vs_permuted']:.3e} "
            f"(permutation gap={lb['permutation_gap']:.3e})"
        )
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
