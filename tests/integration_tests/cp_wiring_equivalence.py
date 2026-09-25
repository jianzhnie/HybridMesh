"""CP wiring check: cp>1 through ``apply_cp`` must match the full-sequence run.

Run under torchrun with 2 ranks:

    torchrun --nproc_per_node=2 tests/cp_wiring_equivalence.py

Where ``cp_equivalence.py`` checks the redistribution primitives, this checks
the WIRING end to end: a real (tiny, offline, randomly initialized) qwen3 goes
through ``apply_cp``, gets fed CP-sharded inputs from ``shard_batch_for_cp``,
and every rank's forward must reproduce the single-rank full-sequence forward
exactly -- logits gathered back from the CP ranks equal the reference logits,
and the summed token losses agree across the sharded and full computations.

Forward-only, and deliberately so: torch's flex attention has no CPU backward
(``NotImplementedError: FlexAttention does not support backward on CPU``), and
this machine has no CUDA. The one backward that IS checkable here -- the K/V
gather's reduce-scatter gradient, a torch custom op independent of flex -- is
exercised directly in ``_check_gather_backward``.

Three scenarios:

* causal, no load balancer: the contiguous split; gathered logits concatenate
  in rank order.
* causal, "headtail": ranks hold one head chunk and one tail chunk, so the
  gathered logits come out rearranged and are un-permuted before comparing.
* packed (three documents under a block_causal BlockMask), "headtail": the
  mask is built full-length and Q-sharded by ``shard_attention_mask_for_cp``,
  then handed to ``forward`` -- the path packing must take, since the wrapper
  cannot rebuild document structure from a positions shard.

Non-vacuity: attending each shard against itself -- what a no-op K/V gather
would compute -- must differ materially from the CP result, so a wiring that
secretly skipped the gather could not pass.

Everything runs in float64 so the comparison is arithmetic, not noise. The
flex backend is forced explicitly because this machine has no CUDA: the
wrapper would otherwise select sdpa, which ``apply_cp`` rejects -- that
rejection is what keeps a real CPU run from silently computing attention over
each rank's own shard only.
"""

from __future__ import annotations

from unittest import mock

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from hpmesh.models.hf_factory import build_model_config_for
from hpmesh.models.hf_wrapper import HFTransformerModel
from hpmesh.parallel.context_parallel import (
    apply_cp,
    shard_attention_mask_for_cp,
    shard_batch_for_cp,
)
from hpmesh.parallel.context_parallel.cp_kernel import CPFlexKernel
from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)

SEQ = 256  # torch's CP BlockMask path requires Q_LEN % (cp * 128) == 0
VOCAB = 128
DOC_LENS = (100, 96, 60)  # the packed scenario; sums to SEQ

# float64 comparisons: wiring differences are O(1), arithmetic noise is O(1e-13).
TOL = 1e-9


def _cfg(load_balancer: str | None) -> HybridMeshConfig:
    return HybridMeshConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            context_parallel_size=2,
            context_parallel_load_balancer=load_balancer,
        ),
        training=TrainingConfig(max_seq_len=SEQ, steps=1),
    )


def _build_model(cfg: HybridMeshConfig, *, flex: bool, seed: int = 0):
    """Deterministically initialized tiny qwen3; identical on every rank."""
    torch.manual_seed(seed)
    model = HFTransformerModel(build_model_config_for(cfg)).to(torch.float64)
    if flex:
        # The machine-level fallback picks sdpa off CUDA; CP needs the flex
        # path (its mask is a BlockMask), so the test opts in explicitly.
        model.model.config._attn_implementation = "flex_torchtitan"
    return model.eval()


def _data(packed: bool, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(VOCAB, (SEQ,), generator=g)
    labels = torch.randint(VOCAB, (SEQ,), generator=g)
    if packed:
        positions = torch.cat([torch.arange(n) for n in DOC_LENS])
    else:
        positions = torch.arange(SEQ)
    return ids, labels, positions


def _additive_mask(positions: torch.Tensor) -> torch.Tensor:
    """The reference mask as a dense additive (1, 1, T, T) float64 tensor."""
    doc = torch.cumsum((positions == 0).int(), dim=0) - 1
    causal = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))
    allow = (doc[:, None] == doc[None, :]) & causal
    additive = torch.zeros(SEQ, SEQ, dtype=torch.float64)
    return additive.masked_fill(~allow, float("-inf"))[None, None]


def _ref_forward(ref, ids, positions):
    """Full-sequence forward through HF's own sdpa path with a dense mask."""
    hidden = ref.model.model(
        ids.unsqueeze(0),
        position_ids=positions.unsqueeze(0),
        attention_mask=_additive_mask(positions),
        use_cache=False,
    ).last_hidden_state.squeeze(0)
    return ref.model.lm_head(hidden)


def _local_only_forward(ref, ids_shard, positions_shard):
    """The shard attended against itself -- what a no-op gather would compute."""
    t = ids_shard.shape[0]
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
    additive = torch.zeros(t, t, dtype=torch.float64)
    additive = additive.masked_fill(~causal, float("-inf"))[None, None]
    hidden = ref.model.model(
        ids_shard.unsqueeze(0),
        position_ids=positions_shard.unsqueeze(0),
        attention_mask=additive,
        use_cache=False,
    ).last_hidden_state.squeeze(0)
    return ref.model.lm_head(hidden)


def _gather_cp(x_local: torch.Tensor, cp_mesh) -> torch.Tensor:
    """Concatenate a tensor across CP ranks, in rank order."""
    xs = [torch.empty_like(x_local) for _ in range(cp_mesh.size())]
    dist.all_gather(xs, x_local.detach().contiguous(), group=cp_mesh.get_group())
    return torch.cat(xs, dim=0)


def _unpermute_headtail(x: torch.Tensor, cp: int) -> torch.Tensor:
    """Undo the headtail rearrangement of a rank-order-concatenated tensor."""
    t = x.shape[0]
    k = t // (2 * cp)
    out = torch.empty_like(x)
    for r in range(cp):
        shard = x[r * 2 * k : (r + 1) * 2 * k]
        out[r * k : (r + 1) * k] = shard[:k]
        out[t - (r + 1) * k : t - r * k] = shard[k:]
    return out


def _run_scenario(
    name: str,
    *,
    packed: bool,
    load_balancer: str | None,
    mesh,
    failures: list[str],
) -> dict[str, float]:
    cfg = _cfg(load_balancer)
    model = _build_model(cfg, flex=True)
    ref = _build_model(cfg, flex=False)
    apply_cp(model, mesh, cfg.parallel)

    ids, labels, positions = _data(packed)
    ids_sh, labels_sh, pos_sh = shard_batch_for_cp(
        ids, labels, positions, mesh["cp"], load_balancer=load_balancer
    )
    masks = None
    if packed:
        # Packing under CP: the wrapper cannot rebuild document structure from
        # a positions shard, so the mask is built full-length and Q-sharded.
        masks = _masks_for(positions, mesh, load_balancer)

    with torch.no_grad():
        logits_local = model(ids_sh, positions=pos_sh, attention_masks=masks)
        ref_logits = _ref_forward(ref, ids, positions)

    gathered = _gather_cp(logits_local, mesh["cp"])
    if load_balancer == "headtail":
        gathered = _unpermute_headtail(gathered, mesh["cp"].size())
    logit_diff = (gathered - ref_logits).abs().max().item()
    if logit_diff > TOL:
        failures.append(f"{name}: logits max abs diff {logit_diff:.3e}")

    # Non-vacuity: shard-against-itself attention must differ materially.
    with torch.no_grad():
        local_only = _local_only_forward(ref, ids_sh, pos_sh)
    noop_diff = (logits_local - local_only).abs().max().item()
    # With a contiguous causal split, rank 0's shard genuinely attends only
    # within itself; the check lands on the ranks whose tokens need the gather.
    expects_difference = load_balancer == "headtail" or mesh["cp"].get_local_rank() > 0
    if expects_difference and noop_diff < 1e-3:
        failures.append(f"{name}: no-op gather matches ({noop_diff:.3e}), vacuous")

    # The loss sums over tokens (permutation-invariant), so the CP ranks'
    # summed losses all-reduced over the CP group must equal the reference's.
    with torch.no_grad():
        loss_local = F.cross_entropy(logits_local, labels_sh, reduction="sum")
        loss_ref = F.cross_entropy(ref_logits, labels, reduction="sum")
    group = mesh["cp"].get_group()
    dist.all_reduce(loss_local, group=group)
    loss_diff = abs(loss_local.item() - loss_ref.item())
    if loss_diff > TOL:
        failures.append(f"{name}: loss diff {loss_diff:.3e}")

    return {"logits": logit_diff, "loss": loss_diff}


def _check_gather_backward(cp_mesh, failures: list[str]) -> float:
    """``flex_cp_allgather`` backward must reduce-scatter the gradient back.

    With a sum over the gathered K/V as the loss, every rank's contribution to
    every other rank's output is 1, so each rank's local gradient is cp copies
    of ones. A gather whose backward silently passed the gradient through
    would return ones -- off by exactly the CP degree.
    """
    cp = cp_mesh.size()
    k = torch.randn(1, 2, 8, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, 8, 4, dtype=torch.float64, requires_grad=True)
    kernel = CPFlexKernel(cp_mesh=cp_mesh)
    k_full, v_full = kernel._flex_cp_allgather(k, v, 2, kernel._cp_pg_name)
    (k_full.sum() + v_full.sum()).backward()
    expected = torch.full_like(k, float(cp))
    grad_diff = max(
        (k.grad - expected).abs().max().item(),
        (v.grad - expected).abs().max().item(),
    )
    if grad_diff > TOL:
        failures.append(f"gather backward: grad diff {grad_diff:.3e}")
    if k_full.shape[2] != 8 * cp:
        failures.append(
            f"gather backward: gathered seq {k_full.shape[2]}, want {8 * cp}"
        )
    return grad_diff


def _masks_for(positions: torch.Tensor, mesh, load_balancer: str | None):
    """The packed CP mask, built with the wrapper's mods in the TEST's frame.

    Uncompiled on purpose: hpmesh's compiler wrapper owns a module-level
    ``torch.compile(create_block_mask)``, and this CPU-only run reaches it with
    a graph whose document-id gather trips an inductor vectorizer bug
    (``VecMask<int>`` vs ``VecMask<float>``). CUDA runs are unaffected.
    """
    from torch.nn.attention.flex_attention import and_masks, create_block_mask

    from hpmesh.models.common.masks import get_causal_mask_mod, get_document_mask_mod

    full = create_block_mask(
        and_masks(get_causal_mask_mod(), get_document_mask_mod(positions)),
        1,
        None,
        SEQ,
        SEQ,
        device="cpu",
        BLOCK_SIZE=128,
    )
    return shard_attention_mask_for_cp(full, mesh["cp"], load_balancer)


def _uncompiled_create_attention_mask(*args, **kwargs):
    """``masks.create_attention_mask`` with the ``torch.compile`` taken out.

    The module-level compile is the one thing about the mask builder that
    cannot run on this CPU-only machine (the document-id gather trips an
    inductor vectorizer bug). Everything the seam does around it -- which
    positions it passes, and when -- is still the production code, which is
    what this check is for.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask(*args, **kwargs)


def _check_preprocess_inputs_matches(mesh, failures: list[str]) -> float:
    """The seam the trainer drives must shard exactly as ``_run_scenario`` does.

    ``_run_scenario`` calls ``shard_batch_for_cp`` directly, which is the
    *library* contract; ``HFTransformerModel.preprocess_inputs`` is the
    *runtime* path -- the trainer calls it, and it owns the CP shard, the
    full-length positions default and the packed BlockMask. The end-to-end CP
    trainer run needs CUDA (``apply_cp`` rejects sdpa), so this is where the
    runtime path's CP branch gets its numerical check: if the two ever disagree
    about the shard or the mask, the equivalence proven above stops describing
    what a real run does.

    It runs the packed batch only. That is the case the trainer could not reach
    before -- the mask must be built from the FULL-length positions, before
    they are sharded -- and it is the only one that needs the mask path at all
    (a non-packed batch falls back to arange plus the causal mod, which is
    unaffected by where the call is made from).

    The mask is compared by *feeding it back through the forward* rather than
    by its block counts: the mask is sparse metadata, so two masks that differ
    only in where a block boundary landed can hold the same tensors and still
    attend differently.
    """
    cfg = _cfg(load_balancer="headtail")
    model = _build_model(cfg, flex=True)
    # The batch is packed, so the model has to be told to mask document
    # boundaries; ``get_attention_masks`` picks its mods off this config, and
    # left at the default it would build a causal-only mask and quietly attend
    # across documents. In a real run ``build_model_config_for`` derives this
    # from the corpus (any non-random dataset is packed); here the model is
    # built without one, so it is set explicitly to the same value.
    model.model.config.attn_mask_type = "block_causal"
    apply_cp(model, mesh, cfg.parallel)
    ids, labels, positions = _data(packed=True)

    # ``preprocess_inputs`` reaches the mesh through ``parallel_dims``, so it
    # gets a stand-in whose only job is to answer ``get_optional_mesh``: "cp"
    # with the CP mesh, "tp" (asked since the TP sequence shard moved into the
    # seam) with None -- this is a CP-only run.
    class _CpOnlyDims:
        @staticmethod
        def get_optional_mesh(name: str):
            assert name in ("cp", "tp"), f"preprocess_inputs asked for '{name}'"
            return mesh["cp"] if name == "cp" else None

    with mock.patch(
        "hpmesh.models.hf_wrapper.create_attention_mask",
        _uncompiled_create_attention_mask,
    ):
        inputs, out_labels, extra = model.preprocess_inputs(
            {"input": ids, "labels": labels, "positions": positions},
            parallel_dims=_CpOnlyDims(),
        )
    ids_sh, labels_sh, pos_sh = shard_batch_for_cp(
        ids, labels, positions, mesh["cp"], load_balancer="headtail"
    )

    for tag, got, want in (
        ("ids", inputs, ids_sh),
        ("labels", out_labels, labels_sh),
        ("positions", extra["positions"], pos_sh),
    ):
        diff = (got - want).abs().max().item()
        if diff != 0:
            failures.append(f"preprocess_inputs sharded {tag}: {diff:.3e}")

    # The mask the seam produced must forward to the same logits as the one
    # built in the test's own frame -- and, since that frame is the one this
    # file already proved equal to the dense reference, that transitively pins
    # the seam's mask to the reference too.
    with torch.no_grad():
        got_logits = model(
            ids_sh, positions=pos_sh, attention_masks=extra["attention_masks"]
        )
        ref_logits = model(
            ids_sh,
            positions=pos_sh,
            attention_masks=_masks_for(positions, mesh, "headtail"),
        )

    diff = (got_logits - ref_logits).abs().max().item()
    if diff > TOL:
        failures.append(f"preprocess_inputs: forward logits diff {diff:.3e}")
    return diff


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("cp",))
    failures: list[str] = []
    stats: dict[str, dict[str, float]] = {}

    stats["causal/contiguous"] = _run_scenario(
        "causal/contiguous",
        packed=False,
        load_balancer=None,
        mesh=mesh,
        failures=failures,
    )
    stats["causal/headtail"] = _run_scenario(
        "causal/headtail",
        packed=False,
        load_balancer="headtail",
        mesh=mesh,
        failures=failures,
    )
    stats["packed/headtail"] = _run_scenario(
        "packed/headtail",
        packed=True,
        load_balancer="headtail",
        mesh=mesh,
        failures=failures,
    )
    # Same numbers, driven through the seam the trainer actually calls -- this
    # is what pins the trainer's CP path, which cannot run here (it needs CUDA).
    stats["packed/via_preprocess_inputs"] = {
        "logits": _check_preprocess_inputs_matches(mesh, failures),
        "loss": 0.0,
    }

    gather_grad = _check_gather_backward(mesh["cp"], failures)

    # Ulysses is wired (see cp_ulysses_equivalence.py for its numerical
    # equivalence); an unknown strategy must still refuse loudly.
    CPFlexKernel(cp_mesh=mesh["cp"], strategy="ulysses")
    try:
        CPFlexKernel(cp_mesh=mesh["cp"], strategy="ring")
        failures.append("unknown strategy: no error raised")
    except NotImplementedError:
        pass

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"seq={SEQ} cp_size={world} dtype=float64 tol={TOL:.0e}")
        for name, s in stats.items():
            print(f"{name:20s} logits={s['logits']:.3e} loss={s['loss']:.3e}")
        print(f"gather backward grad diff = {gather_grad:.3e}")
        print(f"failed ranks   = {int(local_ok.item())}")
        if failures:
            for f in failures:
                print(f"  FAIL {f}")
        else:
            print("all checks passed")

    assert local_ok.item() == 0, "CP wiring equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
