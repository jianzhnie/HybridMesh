"""Trainer -- the single training loop, shared by every learning step.

Learning note: this is deliberately small and linear. The distributed complexity
lives in parallel/ and mesh.py; the loop itself should stay readable end to end.

Order of operations mirrors Titan (parallelize -> compile -> fsdp), but here each
parallelism dimension is an explicit, individually-understandable call.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .. import parallel
from ..bundle import Batch, ModelBundle, build_bundle
from ..mesh import build_mesh, build_parallel_dims, init_distributed
from .config import HybridMeshConfig


class Trainer:
    def __init__(self, cfg: HybridMeshConfig):
        self.cfg = cfg
        self.rank, self.local_rank, self.world_size = init_distributed()

        # Deterministic seeding BEFORE model build so all ranks build identical
        # initial weights -- the precondition for bit-exact DP comparisons.
        self._seed_everything(cfg.seed, deterministic=cfg.deterministic)

        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # 1. mesh (the process topology every dimension is built on). ``parallel_dims``
        #    is the same resolved degrees the mesh was built from, kept so the
        #    trainer can ask "how many DP ranks?" without re-indexing the mesh.
        self.parallel_dims = build_parallel_dims(cfg, self.world_size)
        self.mesh = build_mesh(self.parallel_dims)

        # 2. build the model bundle
        self.bundle: ModelBundle = build_bundle(cfg, device=self.device)
        model = self.bundle.model

        # 3. parallelism, in Titan's order: tp/pp/cp/ep declared first, fsdp last
        #    (outer wraps inner). Each is a no-op when its degree is 1.
        model = parallel.apply_tp(model, self.mesh, cfg)
        model = parallel.apply_cp_ep(model, self.mesh, cfg)
        parallel.apply_pp(model, self.mesh, cfg)  # returns schedule later (step 3)
        if cfg.compile:
            model = torch.compile(model)
        model = parallel.apply_fsdp(model, self.mesh, cfg)
        self.model = model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    @staticmethod
    def _seed_everything(seed: int, *, deterministic: bool) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=False)

    def _make_batch(self, step: int) -> Batch:
        """Synthetic random-token batch. The SAME generator seed on every rank yields
        identical data; DP ranks then take disjoint slices (see _dp_slice)."""
        g = torch.Generator(device="cpu").manual_seed(self.cfg.seed * 100_000 + step)
        ids = torch.randint(
            0,
            self.cfg.vocab_size,
            (self.cfg.global_batch_size, self.cfg.max_seq_len),
            generator=g,
        )
        labels = ids.clone()
        return Batch(input_ids=ids, labels=labels)

    def _dp_slice(self, batch: Batch) -> Batch:
        """Give each DP rank its shard of the global batch (data parallel semantics)."""
        if self.parallel_dims is None:
            dp, dp_rank = 1, 0
        else:
            # The dense DP group spans replicate * shard; unsplit on torchrun
            # it is a plain 1-D mesh, so ``mesh["dp"]`` sizes the batch.
            dp_mesh = self.parallel_dims.get_optional_mesh(
                "dp", include_singleton_axes=True
            )
            dp = dp_mesh.size()
            dp_rank = dp_mesh.get_local_rank()
        per = self.cfg.global_batch_size // dp
        sl = slice(dp_rank * per, (dp_rank + 1) * per)
        return Batch(
            input_ids=batch.input_ids[sl].to(self.device),
            labels=batch.labels[sl].to(self.device),
        )

    def train_step(self, step: int) -> float:
        batch = self._dp_slice(self._make_batch(step))
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.model(batch.input_ids, batch.labels)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def _all_reduce_loss(self, loss: float) -> float:
        """Average the loss across DP ranks so logging reflects the global batch."""
        if self.world_size == 1:
            return loss
        t = torch.tensor([loss], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        return float(t.item())

    def train(self) -> None:
        for step in range(self.cfg.steps):
            loss = self.train_step(step)
            if step % self.cfg.log_freq == 0 and self.rank == 0:
                global_loss = self._all_reduce_loss(loss)
                print(f"step {step:4d} | loss {global_loss:.6f}")
        if dist.is_initialized():
            dist.destroy_process_group()
