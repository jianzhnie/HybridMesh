"""TP x FSDP check: the combined mesh must train like the single-rank model.

Run under torchrun with 4 ranks (from the repo root):

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/tp_fsdp_equivalence.py

tp=2 + dp_shard=2: every rank holds a T/tp sequence shard (the sequence-
parallelism premise) AND a dp_shard parameter shard. A tiny offline qwen3
(random init, fixed seed) goes through the real ``Trainer`` for 4 optimizer
steps and is compared against a single-process reference over the same global
batches -- the loss trajectory AND the final parameters.

Two seams are pinned at once:

* the reported loss: each rank's loss sum covers only its ``T / tp`` token
  shard, so the loss reduce-group must span the tp axis (the ``loss`` view
  includes it); without that the report is 1/tp of the true value. The token
  count needs no tp factor: TP ranks of a DP group read the same rows, and
  the count is taken from the unsharded batch.
* the gradient wiring: TP-sharded weights reduce inside the fused GEMMs,
  TP-replicated ones (embedding, norms, head) through the trainer's
  ``_allreduce_replicated_tp_grads``, and FSDP reduces/shards everything over
  dp_shard -- which must NOT include the tp axis, or replicas' partial
  gradients would be averaged instead of summed.

Non-vacuity: TP weight shards are literal slices of the reference weights and
the two tp ranks hold DIFFERENT slices; FSDP parameters are DTensors whose
local shard is half the global tensor. A combination that silently skipped
either reduction would diverge from the reference within the tolerances below.

Everything runs in fp32 on CPU/gloo (the fallback path of the TP collectives).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from hpmesh.components.loss import IGNORE_INDEX, cross_entropy_loss
from hpmesh.datasets.random_data import Batch, RandomTokenSource, batch_iterator
from hpmesh.models.hf_wrapper import HFTransformerModel, build_model_config_for
from hpmesh.parallel.collectives import clip_grad_norm_
from hpmesh.parallel.parallel_dims import ParallelDims
from hpmesh.parallel.tensor_parallel.tp import (
    ColwiseLinear,
    ColwiseLinearNoGather,
    RowwiseLinear,
)
from hpmesh.trainer import (
    HybridMeshConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from hpmesh.trainer.config import MetricsConfig
from hpmesh.trainer.trainer import Trainer

STEPS = 4
GLOBAL_BATCH = 8
SEQ = 32
VOCAB = 128
SEED = 42

TP = 2
DP_SHARD = 2

# fp32; the sanctioned divergences are summation orders (gathered GEMMs,
# per-rank reduce-scatter), which sit at ~1e-6 relative. A wiring error (a
# missing tp reduction, an averaged-instead-of-summed gradient) is O(1).
TOL = 1e-4


def _cfg() -> HybridMeshConfig:
    return HybridMeshConfig(
        model=ModelConfig(
            # llama, not qwen3: qwen3's HF tp_plan carries
            # ``replicated_with_grad_allreduce`` norm entries that the minimal
            # TP engine deliberately rejects (apply_tp fails loudly on them).
            model_name_or_path="llama",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            tensor_parallel_size=TP,
            data_parallel_shard_size=DP_SHARD,
        ),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            max_seq_len=SEQ,
            steps=STEPS,
            seed=SEED,
            deterministic=True,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def _reference_trajectory(cfg: HybridMeshConfig) -> tuple[list[float], dict]:
    """The same training with no parallelism: one process, the whole batch.

    Mirrors the trainer's non-PP step arithmetic exactly. One subtlety: dp
    sharding turns the global batch into per-group flattened sequences (each
    group's rows collapse into one causal sequence with its own arange
    positions), so the reference must run the DP groups as SEPARATE forwards
    and sum -- a single flat forward over the union would give the second
    group's rows different positions and a different causal boundary. The
    target shift is row-local, so the full-batch targets still double as the
    token count.
    """
    torch.manual_seed(cfg.seed)
    model = HFTransformerModel(build_model_config_for(cfg))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    batches = batch_iterator(
        RandomTokenSource(
            seed=cfg.seed,
            vocab_size=cfg.vocab_size,
            batch_size=cfg.global_batch_size,
            seq_len=cfg.max_seq_len,
        )
    )
    rows_per_dp = cfg.global_batch_size // DP_SHARD
    losses = []
    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        batch = next(batches)
        num_valid = int(
            (
                model.preprocess_inputs(batch, parallel_dims=None)[1] != IGNORE_INDEX
            ).sum()
        )
        loss_sum = None
        for d in range(DP_SHARD):
            rows = slice(d * rows_per_dp, (d + 1) * rows_per_dp)
            sub = Batch(input_ids=batch.input_ids[rows], labels=batch.labels[rows])
            inputs, targets, _ = model.preprocess_inputs(sub, parallel_dims=None)
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            loss_sum = loss if loss_sum is None else loss_sum + loss
        # Normalized BEFORE backward, as the trainer does, so the clip below
        # reads the gradient the optimizer actually applied.
        (loss_sum / num_valid).backward()
        clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm, foreach=True)
        optimizer.step()
        losses.append(float(loss_sum / num_valid))
    params = {name: p.detach().clone() for name, p in model.named_parameters()}
    return losses, params


def _full(t: torch.Tensor) -> torch.Tensor:
    """Global tensor at this rank's TP coordinate: redistribute the FSDP shard."""
    return t.full_tensor() if isinstance(t, DTensor) else t.detach()


def main() -> None:
    cfg = _cfg()
    failures: list[str] = []

    # Trainer init owns the process group (torchrun env): tp=2, dp_shard=2.
    trainer = Trainer(cfg)
    rank = trainer.rank
    world = trainer.world_size
    assert world == TP * DP_SHARD, f"this check assumes 4 ranks, got {world}"
    parallel_dims = trainer.parallel_dims
    assert isinstance(parallel_dims, ParallelDims)
    tp_rank = parallel_dims.get_mesh("tp").get_local_rank()

    reference, ref_params = _reference_trajectory(cfg)

    # -- the trajectory, through the real Trainer -----------------------------
    data_iterator = trainer._data_iterator()
    losses = []
    for _ in range(STEPS):
        trainer.step += 1
        metrics = trainer.train_step(data_iterator)
        assert metrics is not None  # log_freq=1: every step reports
        # The reported loss is the global one: summed over every rank's token
        # shard (the loss mesh spans dp and tp) over the global token count.
        losses.append(metrics["loss"])

    max_loss_diff = 0.0
    for step, (got, want) in enumerate(zip(losses, reference, strict=True), 1):
        diff = abs(got - want)
        max_loss_diff = max(max_loss_diff, diff)
        if diff > TOL:
            failures.append(
                f"rank {rank}: step {step} loss {got:.6f} vs reference "
                f"{want:.6f} (diff {diff:.3e})"
            )

    # -- final parameters ------------------------------------------------------
    model = trainer.model_parts[0]
    # The TP layout of each sharded projection, keyed by its parameter name:
    # colwise shards rows, rowwise shards columns.
    tp_layouts = {}
    for path, mod in model.named_modules():
        if isinstance(mod, ColwiseLinear | ColwiseLinearNoGather):
            tp_layouts[f"{path}.weight"] = "row"
        elif isinstance(mod, RowwiseLinear):
            tp_layouts[f"{path}.weight"] = "col"
    if not tp_layouts:
        failures.append(f"rank {rank}: no TP-sharded projections -- vacuous")

    max_param_diff = 0.0
    saw_fsdp_shard = False
    for name, p in model.named_parameters():
        full = _full(p)
        want = ref_params[name]
        layout = tp_layouts.get(name)
        if layout == "row":
            n = want.shape[0]
            want = want[tp_rank * n // TP : (tp_rank + 1) * n // TP]
        elif layout == "col":
            k = want.shape[1]
            want = want[:, tp_rank * k // TP : (tp_rank + 1) * k // TP]
        elif full.numel() != want.numel():
            # A replicated parameter must arrive whole: FSDP redistributes it.
            failures.append(f"rank {rank}: {name} not replicated whole")
            continue
        # Non-vacuity, FSDP half: a sharded parameter is a DTensor whose local
        # shard is smaller than the global tensor.
        if isinstance(p, DTensor) and p.to_local().numel() < p.numel():
            saw_fsdp_shard = True
        diff = (full - want).abs().max().item()
        max_param_diff = max(max_param_diff, diff)
        if diff > TOL:
            failures.append(f"rank {rank}: param {name} diff {diff:.3e}")
    if not saw_fsdp_shard:
        failures.append(f"rank {rank}: no parameter was FSDP-sharded -- vacuous")

    # Non-vacuity, TP half: the two tp ranks of a dp group must hold different
    # slices of the same weight.
    some_tp_name = next(iter(tp_layouts))
    shard = _full(dict(model.named_parameters())[some_tp_name]).contiguous()
    gathered = [torch.zeros_like(shard) for _ in range(TP)]
    dist.all_gather(gathered, shard, group=parallel_dims.get_mesh("tp").get_group())
    if torch.equal(gathered[0], gathered[1]):
        failures.append(f"rank {rank}: tp ranks hold identical shards (vacuous)")

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0])
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"tp={TP} dp_shard={DP_SHARD} steps={STEPS} "
            f"global_batch={GLOBAL_BATCH} seq={SEQ} tol={TOL:.0e}"
        )
        print(f"reference losses = {[f'{x:.6f}' for x in reference]}")
        print(f"tp+fsdp losses   = {[f'{x:.6f}' for x in losses]}")
        print(f"loss     max abs diff = {max_loss_diff:.3e}")
        print(f"params   max abs diff = {max_param_diff:.3e}")
        print(f"failed ranks       = {int(local_ok.item())}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "TP x FSDP equivalence check failed"
    trainer.checkpointer.close()
    trainer.metrics.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
