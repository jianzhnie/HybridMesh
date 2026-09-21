# Optimizer checkpoint format migration

This note records the one breaking change in the `OptimizersContainer` change
(commit `0fd6cbe`): **optimizer state on disk has a new layout.**

## Why

The old layout keyed optimizer state by *positional* parameter index. Under
pipeline parallelism every stage's optimizer numbers its own parameters from 0,
so two stages wrote the same key into one shared checkpoint and one of them was
lost. The old `OptimizerWrapper` worked around this by re-keying to FQNs, but
only when explicitly asked (`fqn_keying=True`), and the non-FFN path kept the
positional layout that cannot be safe there at all.

`OptimizersContainer` spans every model part by construction, so it has no
positional layout to fall back to. Its state dict is always flat and FQN-keyed.

## What changed

| | before | after |
|---|---|---|
| state key | `state/weight` -> nested `{exp_avg, exp_avg_sq, step}` | `state/weight/exp_avg`, `state/weight/exp_avg_sq`, `state/weight/step` |
| param group key | `param_groups/0/lr` (one shared group) | `param_groups/weight/lr` (what `state_dict()` reports for each group) |

The second row is a genuine improvement for checkpoint portability: an FQN-keyed
param group is unambiguous per parameter, where a positional `0` is not.

## Impact

**Checkpoints written before `0fd6cbe` will not load into a build after it.**
There is no conversion script, and none is planned: the mapping is only
recoverable from the *old* checkpoint's own metadata, and this is a
pre-1.0 research framework.

A run resumed from an old checkpoint will fail loudly rather than silently
training with a cold optimizer, because DCP cannot match the nonexistent keys.
Delete or re-export old checkpoints.

## Model weights are unaffected

Only the optimizer subtree moved. `model/...` keys, the trainer's
`train_state` counters, and the dataloader cursor are unchanged, so model
weights export and reload exactly as before.

## Verified

- Default run reproduces the pre-change baseline bitwise (`loss` 4.85817 /
  4.85671 / 4.85931 / 4.85672, `grad_norm` 0.5583 / 0.5612 / 0.5572 / 0.5487)
  at `--steps 4 --seed 42 --deterministic`. The `fused` default was confirmed
  bit-identical to the for-loop kernel on CPU; on CUDA the fused kernel is a
  different implementation and is expected to differ in the last bits.
- PP checkpoint round-trip is bitwise exact: `tests/integration_tests/pp_checkpoint_equivalence.py`
  reports `max abs diff = 0.000e+00`.
- `tests/integration_tests/pp_equivalence.py` (1F1B and Interleaved1F1B) both pass.
- `tests/integration_tests/cp_wiring_equivalence.py` passes, including the
  `preprocess_inputs` seam.

## A bug this change exposed

While re-checking the CP path, a separate defect surfaced:
**`attn_mask_type` had no config path.** Nothing in `trainer/config.py` set it,
and only two equivalence tests assigned it by hand. It falls back to
`"causal"` at the mask site (`hf_wrapper.py`, `context_parallel/apply.py`).

That matters because every non-random corpus is packed: `datasets/build.py`
always runs samples through `ConcatThenSplitPackingConfig`, so a row holds
several documents and attention must not cross a boundary. The consequences:

- on **flex** (the CUDA path) an unset flag silently builds a causal-only mask
  and attends across document boundaries -- no error, wrong model;
- on **sdpa** (CPU) the wrapper's packed-sequence guard raises, so the failure
  is loud but arrives as "packing requires CUDA" rather than naming the cause.

`build_model_config_for` now derives the flag from the dataset selector
(`"causal"` for the synthetic random corpus, `"block_causal"` otherwise), so it
cannot disagree with the corpus the trainer loaded. There is deliberately no CLI
knob: packing is not independently configurable today, and a knob that could
contradict the data is the bug, not the fix.

**Why the existing tests missed it:** `cp_wiring_equivalence.py` builds its model
directly and sets the flag itself, so it was testing the mask machinery in
isolation rather than the seam a real run goes through. The new
`test_the_mask_type_follows_the_corpus_rather_than_being_configured` pins the
seam.
