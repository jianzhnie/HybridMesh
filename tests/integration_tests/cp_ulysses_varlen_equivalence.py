"""Ulysses CP x packed (varlen) sequences: full-length mask, full-sequence math.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/cp_ulysses_varlen_equivalence.py

Where ``cp_ulysses_equivalence.py`` pins the single-document ulysses wiring,
this pins the varlen/packed combination that used to be refused at attach
time. The premise: ulysses all-to-all's every rank's token shard into a head
shard BEFORE attention, so each rank attends the full, unpermuted token stream
with ``heads / cp`` heads -- and the packed document structure (hpmesh's
varlen metadata, baked into the BlockMask from the full positions) therefore
applies to the kernel FULL-LENGTH and unsharded, exactly as upstream's ulysses
``cp_shard`` lifts ``attention_masks`` out of the sharded inputs.

Three layers of checks:

* the seam: ``preprocess_inputs`` on a multi-document packed row must hand the
  forward a full-length (unsharded) document mask under ulysses -- Q-sharded
  would index the wrong queries;
* the wiring: sharded logits gathered in rank order equal the single-rank
  dense block-causal reference (documents of uneven lengths, one of length 1,
  boundaries off the CP split and off the mask's 128-block grid). Non-vacuity:
  a causal-only reference -- attention crossing document boundaries -- must
  differ materially, so a kernel that silently dropped the document structure
  could not pass;
* the gradients: torch's flex attention has no CPU backward, so the collective
  duality is pinned around an SDPA inner attention with the same dense
  document mask: loss and q/k/v gradients of the sharded path
  (``SeqToHead`` -> attention -> ``HeadToSeq``) must equal the full-sequence
  reference's shards, and must be nonzero (non-vacuity).

CPU/gloo, float64, flex forced explicitly. Everything forward-only except the
SDPA gradient check, which needs no flex kernel.
"""

from __future__ import annotations

from unittest import mock

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from hpmesh.models.hf_factory import build_model_config_for
from hpmesh.models.hf_wrapper import HFTransformerModel
from hpmesh.parallel.context_parallel import apply_cp
from hpmesh.parallel.context_parallel.cp_kernel import HeadToSeq, SeqToHead
from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)

SEQ = 256  # torch's CP BlockMask path requires Q_LEN % (cp * 128) == 0
VOCAB = 128
DOC_LENS = (90, 1, 165)  # uneven, off the cp split (128) and the block grid

TOL = 1e-9  # float64: wiring differences are O(1), arithmetic noise is O(1e-13)
NONVACUITY = 1e-3


def _cfg() -> HybridMeshConfig:
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
            context_parallel_strategy="ulysses",
            context_parallel_load_balancer=None,
        ),
        training=TrainingConfig(max_seq_len=SEQ, steps=1),
    )


def _build_model(cfg: HybridMeshConfig, *, flex: bool, seed: int = 0):
    torch.manual_seed(seed)
    model = HFTransformerModel(build_model_config_for(cfg)).to(torch.float64)
    # The packed mask is a document mask; declare the corpus shape the way
    # ``build_model_config_for`` would for any non-random dataset.
    model.model.config.attn_mask_type = "block_causal"
    if flex:
        model.model.config._attn_implementation = "flex_torchtitan"
    return model.eval()


def _packed_data(seed: int = 0):
    """A multi-document packed row: ids/labels plus positions resetting per doc."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(VOCAB, (SEQ,), generator=g)
    labels = torch.randint(VOCAB, (SEQ,), generator=g)
    positions = torch.cat([torch.arange(n) for n in DOC_LENS])
    assert positions.shape[0] == SEQ
    return ids, labels, positions


def _packed_additive_mask(positions: torch.Tensor, *, document: bool) -> torch.Tensor:
    """Dense (1, 1, T, T) block-causal additive mask, float64.

    ``document=False`` drops the document half -- the cross-boundary reference
    the non-vacuity check needs.
    """
    t = positions.shape[0]
    allow = torch.tril(torch.ones(t, t, dtype=torch.bool))
    if document:
        # Same doc-id convention as masks.get_document_mask_mod.
        doc_ids = torch.cumsum((positions == 0).int(), dim=0) - 1
        allow &= doc_ids[:, None] == doc_ids[None, :]
    additive = torch.zeros(t, t, dtype=torch.float64)
    return additive.masked_fill(~allow, float("-inf"))[None, None]


def _ref_forward(ref, ids, positions, *, document: bool = True):
    """Full-sequence forward through HF's own sdpa path with a dense mask."""
    hidden = ref.model.model(
        ids.unsqueeze(0),
        position_ids=positions.unsqueeze(0),
        attention_mask=_packed_additive_mask(positions, document=document),
        use_cache=False,
    ).last_hidden_state.squeeze(0)
    return ref.model.lm_head(hidden)


def _uncompiled_create_attention_mask(*args, **kwargs):
    """The production mask builder minus ``torch.compile`` (CPU inductor bug)."""
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask(*args, **kwargs)


class _CpOnlyDims:
    def __init__(self, mesh):
        self._mesh = mesh

    def get_optional_mesh(self, name: str):
        assert name in ("cp", "tp"), f"preprocess_inputs asked for '{name}'"
        return self._mesh["cp"] if name == "cp" else None


def _run_ulysses_packed(mesh, failures: list[str]) -> dict[str, float]:
    cfg = _cfg()
    model = _build_model(cfg, flex=True)
    ref = _build_model(cfg, flex=False)
    apply_cp(model, mesh, cfg.parallel)

    ids, labels, positions = _packed_data()
    with mock.patch(
        "hpmesh.models.hf_wrapper.create_attention_mask",
        _uncompiled_create_attention_mask,
    ):
        inputs, out_labels, extra = model.preprocess_inputs(
            {"input": ids, "labels": labels, "positions": positions},
            parallel_dims=_CpOnlyDims(mesh),
        )

    # THE pin: under ulysses the packed mask must reach the forward FULL-LENGTH
    # (unsharded) -- the varlen input contract. Q-sharded would be (SEQ/cp, SEQ).
    mask = extra.get("attention_masks")
    if mask is None:
        failures.append("preprocess_inputs built no mask for a packed ulysses row")
        return {}
    if tuple(mask.seq_lengths) != (SEQ, SEQ):
        failures.append(
            f"packed ulysses mask seq_lengths {tuple(mask.seq_lengths)}, "
            f"expected the full-length ({SEQ}, {SEQ})"
        )

    with torch.no_grad():
        logits_local = model(
            inputs,
            positions=extra["positions"],
            attention_masks=mask,
        )
        ref_logits = _ref_forward(ref, ids, positions)
        cross_boundary = _ref_forward(ref, ids, positions, document=False)

    gathered = [torch.empty_like(logits_local) for _ in range(mesh["cp"].size())]
    dist.all_gather(
        gathered, logits_local.detach().contiguous(), group=mesh["cp"].get_group()
    )
    logit_diff = (torch.cat(gathered, dim=0) - ref_logits).abs().max().item()
    if logit_diff > TOL:
        failures.append(f"ulysses/packed: logits max abs diff {logit_diff:.3e}")

    # Non-vacuity: dropping the document structure must change the answer
    # materially, so a kernel attending across boundaries cannot pass above.
    doc_gap = (ref_logits - cross_boundary).abs().max().item()
    if doc_gap < NONVACUITY:
        failures.append(f"document mask is a no-op ({doc_gap:.3e}), vacuous")

    # Summed token losses agree after the CP reduction.
    with torch.no_grad():
        loss_local = F.cross_entropy(logits_local, out_labels, reduction="sum")
        loss_ref = F.cross_entropy(ref_logits, labels, reduction="sum")
    dist.all_reduce(loss_local, group=mesh["cp"].get_group())
    loss_diff = abs(loss_local.item() - loss_ref.item())
    if loss_diff > TOL:
        failures.append(f"ulysses/packed: loss diff {loss_diff:.3e}")

    return {"logits": logit_diff, "loss": loss_diff, "doc_gap": doc_gap}


def _check_gradients_under_varlen_mask(
    cp_mesh, failures: list[str]
) -> dict[str, float]:
    """Loss and q/k/v grads: sharded path vs full-sequence reference, varlen mask.

    Flex has no CPU backward, so the inner attention here is SDPA against the
    same dense document mask -- what is pinned is the collective duality
    (``SeqToHead`` / ``HeadToSeq`` forward-backward pairing) carrying correct
    gradients under a varlen mask, not the flex kernel.
    """
    rank = cp_mesh.get_local_rank()
    cp = cp_mesh.size()
    group = cp_mesh.get_group()
    b, h, s, d = 1, 4, SEQ, 16
    g = torch.Generator().manual_seed(0)
    qkv_full = [
        torch.randn(b, h, s, d, generator=g, dtype=torch.float64) for _ in "qkv"
    ]
    w = torch.randn(b, h, s, d, generator=g, dtype=torch.float64)
    _, _, positions = _packed_data()
    mask = _packed_additive_mask(positions, document=True)

    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.expand(b, h, s, s)
        )

    # Reference: full sequence, per-rank token shard of the loss weight.
    lo, hi = rank * s // cp, (rank + 1) * s // cp
    ref_qkv = [x.clone().requires_grad_(True) for x in qkv_full]
    ref_out = sdpa(*ref_qkv)
    (ref_out * w).sum().backward()

    # Sharded: token shard -> head shard -> attention -> token shard.
    sh_qkv = [x[:, :, lo:hi].clone().requires_grad_(True) for x in qkv_full]
    a2a_qkv = [SeqToHead.apply(x, group) for x in sh_qkv]
    out = sdpa(*a2a_qkv)
    out = HeadToSeq.apply(out, group)
    (out * w[:, :, lo:hi]).sum().backward()

    out_diff = (out - ref_out[:, :, lo:hi]).abs().max().item()
    if out_diff > TOL:
        failures.append(f"varlen grad path: output diff {out_diff:.3e}")
    grad_diff = max(
        (sh.grad - ref.grad[:, :, lo:hi]).abs().max().item()
        for sh, ref in zip(sh_qkv, ref_qkv, strict=True)
    )
    if grad_diff > TOL:
        failures.append(f"varlen grad path: grad diff {grad_diff:.3e}")
    grad_norm = min(sh.grad.abs().max().item() for sh in sh_qkv)
    if grad_norm == 0.0:
        failures.append("varlen grad path: an all-zero gradient, vacuous")

    return {"output": out_diff, "grad": grad_diff}


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("cp",))
    failures: list[str] = []

    stats = _run_ulysses_packed(mesh, failures)
    grad_stats = _check_gradients_under_varlen_mask(mesh["cp"], failures)

    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"seq={SEQ} cp_size={world} doc_lens={DOC_LENS} dtype=float64 tol={TOL:.0e}"
        )
        if stats:
            print(
                f"ulysses/packed logits={stats['logits']:.3e} "
                f"loss={stats['loss']:.3e} doc_gap={stats['doc_gap']:.3e}"
            )
        print(
            f"varlen grad path out={grad_stats['output']:.3e} "
            f"grad={grad_stats['grad']:.3e}"
        )
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "CP ulysses varlen equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
