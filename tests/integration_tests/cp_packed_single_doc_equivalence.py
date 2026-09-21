"""CP + packed single-document batch: the mask must be prebuilt, restart or not.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/cp_packed_single_doc_equivalence.py

The packing collator splits an overlong document into single-document rows,
whose positions are a plain monotonic arange -- there is no restart for a
``positions[1:] < positions[:-1]`` scan to find. ``preprocess_inputs`` used to
decide "packed, prebuild the mask" by exactly that scan, so this batch shape
took the no-mask CP branch and the forward raised in
``_get_cp_attention_masks`` (block_causal cannot be rebuilt from a positions
shard). The decision is now keyed off ``attn_mask_type``.

This pins the fixed path: a single-document packed row
(``attn_mask_type="block_causal"``, ``positions=arange``) must come out of
``preprocess_inputs`` carrying a prebuilt, Q-sharded BlockMask, and the
forward driven by the seam's own outputs must match the dense causal
reference -- one document, so block_causal and causal coincide.

CPU/gloo, float64, flex forced explicitly (sdpa cannot consume a BlockMask;
``apply_cp`` rejects it). Modeled on ``cp_wiring_equivalence.py``, which
covers the multi-document packed row.
"""

from __future__ import annotations

from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config_for
from hpmesh.parallel.context_parallel import apply_cp, shard_batch_for_cp
from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)

SEQ = 256  # torch's CP BlockMask path requires Q_LEN % (cp * 128) == 0
VOCAB = 128

TOL = 1e-9  # float64: wiring differences are O(1), arithmetic noise is O(1e-13)


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
            # Contiguous split: the gathered logits then concatenate in rank
            # order and compare against the dense reference directly.
            context_parallel_load_balancer=None,
            backend="gloo",
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


def _uncompiled_create_attention_mask(*args, **kwargs):
    """The production mask builder minus ``torch.compile`` (CPU inductor bug)."""
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask(*args, **kwargs)


def _ref_forward(ref, ids, positions):
    """Dense sdpa reference; one document, so the additive mask is plain causal."""
    causal = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))
    additive = torch.zeros(SEQ, SEQ, dtype=torch.float64)
    additive = additive.masked_fill(~causal, float("-inf"))[None, None]
    hidden = ref.model.model(
        ids.unsqueeze(0),
        position_ids=positions.unsqueeze(0),
        attention_mask=additive,
        use_cache=False,
    ).last_hidden_state.squeeze(0)
    return ref.model.lm_head(hidden)


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this check assumes 2 ranks, got {world}"

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("cp",))
    cfg = _cfg()
    model = _build_model(cfg, flex=True)
    ref = _build_model(cfg, flex=False)
    apply_cp(model, mesh, cfg.parallel)

    # One document filling the whole row: the packing collator's output for an
    # overlong document. Monotonic positions -- no restart anywhere.
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, VOCAB, (SEQ,), generator=g)
    labels = torch.randint(0, VOCAB, (SEQ,), generator=g)
    positions = torch.arange(SEQ)
    assert not bool((positions[1:] < positions[:-1]).any())

    class _CpOnlyDims:
        @staticmethod
        def get_optional_mesh(name: str):
            assert name in ("cp", "tp"), f"preprocess_inputs asked for '{name}'"
            return mesh["cp"] if name == "cp" else None

    failures: list[str] = []

    with mock.patch(
        "hpmesh.models.hf_wrapper.create_attention_mask",
        _uncompiled_create_attention_mask,
    ):
        inputs, out_labels, extra = model.preprocess_inputs(
            {"input": ids, "labels": labels, "positions": positions},
            parallel_dims=_CpOnlyDims(),
        )

    # THE pin: the seam must have prebuilt and Q-sharded the mask even though
    # the positions carry no restart. Without it the flex forward would raise
    # in ``_get_cp_attention_masks``.
    if "attention_masks" not in extra:
        failures.append(
            "preprocess_inputs built no mask for a single-document packed row"
        )

    ids_sh, labels_sh, pos_sh = shard_batch_for_cp(ids, labels, positions, mesh["cp"])
    for tag, got, want in (
        ("ids", inputs, ids_sh),
        ("labels", out_labels, labels_sh),
        ("positions", extra["positions"], pos_sh),
    ):
        if not torch.equal(got, want):
            failures.append(f"preprocess_inputs sharded {tag} differently")

    logit_diff = float("nan")
    if not failures:
        with torch.no_grad():
            logits_local = model(
                inputs,
                positions=extra["positions"],
                attention_masks=extra["attention_masks"],
            )
            ref_logits = _ref_forward(ref, ids, positions)
        gathered = [torch.empty_like(logits_local) for _ in range(world)]
        dist.all_gather(
            gathered, logits_local.detach().contiguous(), group=mesh["cp"].get_group()
        )
        logit_diff = (torch.cat(gathered, dim=0) - ref_logits).abs().max().item()
        if logit_diff > TOL:
            failures.append(f"logits max abs diff {logit_diff:.3e}")

    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"seq={SEQ} cp_size={world} single-document packed row, dtype=float64")
        print(f"logits max abs diff = {logit_diff:.3e}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "CP packed single-document check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
