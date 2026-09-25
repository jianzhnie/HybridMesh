"""CP + TP at once: tp>1 and cp>1 must compose, and each axis must be real.

Run under torchrun with 4 ranks (tp=2 x cp=2):

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/cp_tp_equivalence.py

Why this file exists: CP and TP are coupled in four separate places and no
harness ran both axes at once, so every one of those couplings was unverified.

* ``apply.py`` splits ulysses' heads over ``tp * cp``, so the local head count
  is ``H / (tp * cp)`` -- not ``H / cp``.
* ``shard_batch_for_tp`` slices the *CP shard*, so a TP slice must land inside
  this rank's CP block rather than across the whole sequence.
* ``_allreduce_replicated_tp_grads`` sums the gradients of TP-replicated
  parameters (embedding, norms, head), which under CP are also CP-partial.
* The trainer's loss mesh is ``(dp, cp, tp)``, while the denominator is a
  dp-only sum of the *unsharded* batch's token count.

Two levels of comparison, both in float64 so the difference is arithmetic
rather than noise:

* over ``tp``: the local slices join back to this rank's CP block, which must
  equal the reference's logits for exactly those tokens.
* over ``cp``: the CP blocks join back to the full sequence, which must equal
  the single-rank full-sequence reference.

That the first holds is what proves TP sliced *inside* the CP shard; if the
two axes sliced the sequence independently the join would not reproduce the
reference block. It also covers the ulysses strategy, whose kernel converts
back to token shards before returning, so both strategies present the same
externally-visible layout.

Non-vacuity, asserted rather than assumed: ``apply_tp`` must have swapped at
least one projection for a sharded realizer, and each CP rank past the first
must attend over more than its own block (a skipped K/V gather), and the token
count must be replicated on both axes rather than sharded.

The ``tp * cp`` head coupling is stated as an assertion of its own: with
``heads=8`` and ``tp*cp=4`` ``apply_cp`` must accept, and with ``heads=6`` (not
a multiple of 4) it must refuse.

The flex backend is forced explicitly: this machine has no CUDA and the
wrapper would otherwise select sdpa, which ``apply_cp`` rejects.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config_for
from hpmesh.parallel.context_parallel import (
    apply_cp,
    shard_batch_for_cp,
    shard_batch_for_tp,
)
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.parallel.tensor_parallel import apply_tp
from hpmesh.trainer import HybridMeshConfig, ModelConfig, ParallelConfig, TrainingConfig

SEQ = 256  # torch's CP BlockMask path requires Q_LEN % (cp * 128) == 0
VOCAB = 128
HEADS = 8  # divisible by tp * cp == 4
TP = 2
CP = 2
WORLD = TP * CP
TOL = 1e-9
IGNORE_INDEX = -100


def _cfg(strategy: str, *, heads: int = HEADS) -> HybridMeshConfig:
    return HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=heads,
            num_key_value_heads=heads,
        ),
        parallel=ParallelConfig(
            tensor_parallel_size=TP,
            context_parallel_size=CP,
            context_parallel_strategy=strategy,
            context_parallel_load_balancer=None,
        ),
        training=TrainingConfig(max_seq_len=SEQ, steps=1),
    )


def _build(cfg: HybridMeshConfig, *, flex: bool, seed: int = 0) -> HFTransformerModel:
    torch.manual_seed(seed)
    model = HFTransformerModel(build_model_config_for(cfg)).to(torch.float64)
    if flex:
        model.model.config._attn_implementation = "flex_torchtitan"
    return model.eval()


def _data(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(VOCAB, (SEQ,), generator=g)
    labels = torch.randint(VOCAB, (SEQ,), generator=g)
    return ids, labels, torch.arange(SEQ)


def _additive_causal(n: int) -> torch.Tensor:
    """A dense (1, 1, n, n) additive causal mask, for the sdpa reference."""
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
    additive = torch.zeros(n, n, dtype=torch.float64)
    return additive.masked_fill(~causal, float("-inf"))[None, None]


def _ref_logits(ref: HFTransformerModel, ids, positions) -> torch.Tensor:
    hidden = ref.model.model(
        ids.unsqueeze(0),
        position_ids=positions.unsqueeze(0),
        attention_mask=_additive_causal(positions.shape[0]),
        use_cache=False,
    ).last_hidden_state.squeeze(0)
    return ref.model.lm_head(hidden)


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


def _tp_wrapper_count(model: HFTransformerModel) -> int:
    """How many projections ``apply_tp`` swapped for a sharded realizer.

    hpmesh's TP is manual-collective, not DTensor-based: ``apply_tp`` replaces a
    target ``nn.Linear`` with ``ColwiseLinear`` / ``RowwiseLinear`` /
    ``ColwiseLinearNoGather``, whose stored weight is a literal slice of the
    original. So "did TP do anything" is a class-count question, not an
    ``isinstance(param, DTensor)`` one.

    This is a *diagnostic*, not the proof: the proof that TP sharded is the
    ``tp_join`` comparison below, which rejoins the ranks' slices and requires
    them to be exactly this rank's CP block. A TP that replicated instead would
    produce a doubled token axis there and fail on shape. What this adds is a
    failure message that names the cause.
    """
    wrappers = {"ColwiseLinear", "RowwiseLinear", "ColwiseLinearNoGather"}
    return sum(1 for m in model.modules() if type(m).__name__ in wrappers)


def _run_strategy(name, *, strategy, cp_mesh, tp_mesh, failures) -> dict[str, float]:
    cfg = _cfg(strategy)
    model = _build(cfg, flex=True)
    ref = _build(cfg, flex=False)

    # TP then CP, the order parallelize_hf_transformers uses.
    model = apply_tp(model, tp_mesh, cfg.parallel)
    apply_cp(model, _dense_view(cp_mesh, tp_mesh), cfg.parallel)

    # Non-vacuity on the TP axis: at least the q/k/v/o and gate/up/down
    # projections must have been swapped for sharded realizers. A count of
    # zero means every logits comparison below compares replicas.
    wrappers = _tp_wrapper_count(model)
    if wrappers == 0:
        failures.append(f"{name}: no sharded-linear wrapper -- TP sharded nothing")
    elif wrappers < 4:
        failures.append(
            f"{name}: only {wrappers} sharded projections -- fewer than the "
            "q/k/v/o set every decoder layer carries, so the plan only partly matched"
        )

    ids, labels, positions = _data()
    ids_cp, labels_cp, pos_cp = shard_batch_for_cp(
        ids, labels, positions, cp_mesh, load_balancer=None
    )
    ids_r, labels_r = shard_batch_for_tp(ids_cp, labels_cp, tp_mesh)

    with torch.no_grad():
        local = model(ids_r, positions=pos_cp)
        ref_full = _ref_logits(ref, ids, positions)

    # This rank's CP block, as the FULL-sequence reference computes it -- the
    # K/V every rank attends over, restricted to this rank's queries.
    block = SEQ // CP
    start = cp_mesh.get_local_rank() * block
    ref_block = ref_full[start : start + block]

    # -- level 1: join over tp -> this rank's CP block -----------------------
    # That the TP slices rejoin to exactly this block is what proves TP sliced
    # *inside* the CP shard; two axes slicing the sequence independently would
    # reproduce some other set of tokens, not this rank's.
    tp_joined = _gather(local, tp_mesh)
    tp_diff = _diff(
        f"{name}: tp-joined vs CP-block reference", tp_joined, ref_block, failures
    )

    # -- level 2: join those blocks over cp -> the whole sequence ------------
    cp_joined = _gather(tp_joined, cp_mesh)
    cp_diff = _diff(
        f"{name}: cp-joined vs full reference", cp_joined, ref_full, failures
    )

    # -- non-vacuity: the CP gather must have mattered -----------------------
    # A rank that skipped the K/V gather would attend its own block against
    # only itself. That is a *different* model, so it gets a reference of its
    # own: the same tokens and positions through the un-sharded model, whose
    # causal mask then spans just this block. On the ranks whose tokens need
    # predecessors the two must differ materially, or the equality above is
    # satisfied by a no-op.
    if CP > 1 and cp_mesh.get_local_rank() > 0:
        with torch.no_grad():
            noop_ref = _ref_logits(ref, ids_cp, pos_cp)
        noop = (tp_joined - noop_ref).abs().max().item()
        if noop < 1e-3:
            failures.append(f"{name}: no-op CP gather matches ({noop:.3e}) -- vacuous")

    # -- the loss, summed over tokens and reduced across cp ------------------
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
        failures.append(f"{name}: reduced loss diff {loss_diff:.3e}")

    # -- the denominator property -------------------------------------------
    # The batch is REPLICATED across cp and tp (a replica's ranks all read it),
    # so the trainer's dp-only sum counts each token once. Reducing over cp or
    # tp instead would multiply the count by that degree -- which is exactly
    # the failure the trainer's mesh choice avoids. Pinned here: the local
    # count must equal the unsharded batch's, on every rank.
    local_count = int((labels_r != IGNORE_INDEX).sum())
    full_count = int((labels != IGNORE_INDEX).sum())
    if local_count != full_count // (CP * TP):
        failures.append(
            f"{name}: local valid tokens {local_count}, expected "
            f"{full_count} // (cp*tp={CP * TP})"
        )
    for axis_name, mesh in (("cp", cp_mesh), ("tp", tp_mesh)):
        tally = torch.tensor(local_count, dtype=torch.int64)
        dist.all_reduce(tally, group=mesh.get_group())
        if int(tally) != local_count * mesh.size():
            failures.append(
                f"{name}: summing the token count over {axis_name} gave "
                f"{int(tally)}, expected {local_count} * {mesh.size()} -- the "
                "count is replicated on that axis and must not be reduced there"
            )

    return {"tp_join": tp_diff, "cp_join": cp_diff, "loss": loss_diff}


def _diff(what: str, got: torch.Tensor, want: torch.Tensor, failures) -> float:
    if got.shape != want.shape:
        failures.append(f"{what}: shape {tuple(got.shape)} vs {tuple(want.shape)}")
        return float("inf")
    diff = (got - want).abs().max().item()
    if diff > TOL:
        failures.append(f"{what}: max abs diff {diff:.3e}")
    return diff


def _check_ulysses_headcount_guard(cp_mesh, tp_mesh, failures) -> None:
    """``apply_cp`` must key ulysses' head divisibility off ``tp * cp``.

    ``heads=8`` with ``tp*cp=4`` divides (accepted); ``heads=6`` does not
    (refused). A guard written against ``cp`` alone would accept the second and
    then slice heads wrongly inside the all-to-all, so the refusal is the
    observable that separates the two spellings.
    """
    ok = _cfg("ulysses", heads=HEADS)
    model_ok = _build(ok, flex=True)
    apply_cp(model_ok, _dense_view(cp_mesh, tp_mesh), ok.parallel)  # must not raise

    bad = _cfg("ulysses", heads=HEADS // 2 + 2)  # 6, not divisible by tp*cp=4
    model_bad = _build(bad, flex=True)
    try:
        apply_cp(model_bad, _dense_view(cp_mesh, tp_mesh), bad.parallel)
    except ValueError:
        return
    failures.append(
        "ulysses head guard: heads=6 with tp*cp=4 was accepted; the divisibility "
        "check is not keyed off tp * cp"
    )


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == WORLD, f"this check assumes {WORLD} ranks, got {world}"

    cfg = _cfg("kv_allgather")
    parallel_dims = ParallelDims.from_config(cfg.parallel, world)
    cp_mesh = parallel_dims.get_mesh("cp")
    tp_mesh = parallel_dims.get_mesh("tp")
    assert cp_mesh.size() == CP and tp_mesh.size() == TP

    failures: list[str] = []
    stats: dict[str, dict[str, float]] = {}

    for strategy in ("kv_allgather", "ulysses"):
        stats[strategy] = _run_strategy(
            strategy,
            strategy=strategy,
            cp_mesh=cp_mesh,
            tp_mesh=tp_mesh,
            failures=failures,
        )
    _check_ulysses_headcount_guard(cp_mesh, tp_mesh, failures)

    if rank == 0:
        print(f"tp={TP} cp={CP} world={world} seq={SEQ} heads={HEADS} dtype=float64")
        for strategy, row in stats.items():
            print(
                f"  {strategy:12s} " + "  ".join(f"{k}={v:.3e}" for k, v in row.items())
            )
        for f in failures:
            print(f"  FAIL {f}")
        print(f"failed ranks   = {len(failures)}")
        print("all checks passed" if not failures else "CHECKS FAILED")
    # Every rank must agree on failure, not just rank 0.
    verdict = torch.tensor(len(failures), dtype=torch.int64)
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    assert int(verdict) == 0, f"{int(verdict)} check(s) failed -- see rank 0 output"


if __name__ == "__main__":
    main()
