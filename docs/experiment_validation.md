# Shared experiment validation — 2026-10-14 (Asia/Tokyo)

All nine validation cases trained on the NVIDIA GeForce RTX 4060 Ti (16 GiB,
driver 570.211.01) and restored in a separate Python process. The complete
local evidence is [validation_summary.json](../artifacts/experiments_validation/validation_summary.json).
Each case also has its own `validation.json`, resolved YAML, asset snapshots,
manifest, checkpoint, and saved observation probes in that directory.

The validation uses Torch 2.7.1 with CUDA runtime 12.6, Ray 2.47.1, RDKit
2025.3.6, NumPy 1.26.4, and Python 3.12.7. Run manifests record base revision
`414692672e5fb078606229f4fdf3500601d51e11`, the working tree status, and
per-file SHA256 hashes of the implementation used by each case. Model assets,
vocabulary, resolved configuration, and the restored model state have SHA256
checks. This records work that had not yet been committed when measured.

Reproduce the GPU suite from the repository root:

```bash
.venv/bin/python -m scripts.validate_experiment --require-cuda \
  --output artifacts/experiments_validation_new
```

Use `--device cpu` for CPU-only PPO validation. `--require-cuda` fails if CUDA
cannot be accessed; it never substitutes a CPU result. `--cases actor_value
mpnn_trainable` selects cases, and `--resume` reuses completed cases only when
their saved validation settings match. Every training case and restoration is
a separate subprocess, and cases run sequentially on one GPU.

The measured smoke settings were seed 0, one PPO iteration, batch 64,
minibatch 32, one optimization epoch, zero remote rollout workers, atom limit
20, attachment cap 128, decision limit 12, QED threshold 1.1, and the eight
fragments listed by the validation script. Policy hidden width remained 256;
MPNN used three layers with hidden/output width 128, 4096-atom chunks, and
gradient checkpointing for joint training. The frozen-GNN case enabled a
32-entry fragment embedding cache. These runs verify execution and learning
mechanics; the larger seed/worker sweeps provide throughput and short quality
comparisons.

| Case | Reported PPO loss | U auxiliary loss | Encoder absolute change | U absolute change | Peak CUDA allocated MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed Mol2Vec, actor, rich context | 0.01951 | disabled | fixed vectors | disabled | 71.6 |
| Fixed Mol2Vec, actor, classic context | -0.07022 | disabled | fixed vectors | disabled | 66.8 |
| Fixed Mol2Vec, actor plus U | 0.02007 | 0.01739 | fixed vectors | 6.3478 | 82.9 |
| Fixed Mol2Vec, auxiliary U only | 0.01995 | 0.01739 | fixed vectors | 6.3482 | 82.9 |
| Frozen MPNN, actor plus U | 0.03832 | 0.04698 | 0.0000 | 2.1413 | 232.5 |
| Joint MPNN, actor plus U | 0.03298 | 0.04577 | 28.7304 | 2.1217 | 290.0 |
| Joint MPNN, VAE fragment IDs | 0.50110 | 0.44309 | 26.7980 | 6.0515 | 281.0 |
| Explicit legacy environment | 0.31398 | disabled | fixed vectors | disabled | 22.4 |
| Old E2E model and registration | -0.07022 | disabled | fixed vectors | disabled | 70.5 |

Absolute change is the sum of absolute parameter differences before and after
the PPO update. Every case changed policy parameters, and every enabled U head
changed. The frozen encoder retained every parameter exactly. The reported
PPO loss is RLlib's ordinary learner statistic; U's separately recorded custom
loss is added for optimization. All recorded policy/value losses, entropy, KL,
parameters, and actual learner gradients were finite. Peak allocations include
construction, rollout validation, and PPO; they are Torch allocation peaks,
rather than total driver memory or whole-host memory.

The validator sampled a real RLlib rollout before learning and compared saved
`SampleBatch.ACTION_LOGP` and `ACTION_DIST_INPUTS` with the same model's
recomputed distribution. All chosen actions were live mask entries. The largest
log-probability difference was `7.1526e-7`. Candidate-value tests additionally
verify selected phase-B supervision against `value_targets`, Huber coefficient
0.5, detached U contributions to actor logits, auxiliary-only selection, and
training-batch recomputation after an unrelated value forward. CUDA learner
metrics are exported as Python scalars for RLlib's NumPy tower reduction.

Restoration verified exact SHA256 model state equality, exact logits and state
values for both phases, and the identical seeded final molecule and QED score.
Legacy has one observation phase. Each restored sample was a valid RDKit
molecule. New checkpoints restore through saved configuration and the shared
factory. The old E2E case restores through `Algorithm.from_checkpoint` with
the original `moldr_pointer` registration and original model class; it is never
implicitly converted to `moldr_pointer_v2`.

The final version-2 schema uses masked **mean and max** option aggregation in
rich policy context. Padding, negative embeddings, and an empty live set are
handled explicitly. GNN graph readout still projects concatenated **mean and
sum**, as specified. Graph atoms, bonds, counts, and fragment IDs use signed
32-bit integer observations; encoder indexing converts live records to 64-bit
integers. Completed graph buffers retain the `max_attach` axis independently
of vocabulary size. Tests verify equivalent outputs and gradients for 32-bit
and 64-bit records, capacity errors without graph truncation, and rejection of
integer feature overflow.

Message sums and graph readout use ordered `torch.segment_reduce` operations.
This removes the floating-point atomic variation that can change greedy
selection between symmetric graph candidates after checkpoint restoration.
Focused tests verify CPU/CUDA numerical agreement, exact repeated CUDA
inference and restoration, chunk/checkpoint output and gradient agreement,
finite low-precision policy losses, and trainable/frozen encoder updates.

The GNN/VAE smoke case uses `fragments.recond_every=never`: its tiny VAE was
trained without aligned molecular contexts. It validates stable vocabulary IDs
and an independently owned frozen Mol2Vec encoder while PPO trains the GNN.
It does not claim trained conditional decoding. Configuration preflight rejects
episode/step reconditioning for an untrained context decoder; conditioned VAE
experiments require aligned nonzero contexts and persisted training provenance.
An actual CLI preflight using the unconditioned proposer
[rejects episode reconditioning](../artifacts/local_delivery/untrained-context-rejected.log),
while the same invocation with `fragments.recond_every=never`
[succeeds](../artifacts/local_delivery/context-never-accepted.log). This verifies
the provenance gate through the shared command, as well as through unit tests.

The final regression run on 2026-10-14 used the synchronized lockfile and actual
CUDA/Ray access: **459 tests passed**, with 28 warnings, in **32.59 seconds**. The command was
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q`.
The complete local [pytest log](../artifacts/local_delivery/pytest.log) is retained.
New implementation and test files pass Ruff. An
[audit of all 46 changed Python files](../artifacts/local_delivery/ruff-audit.json)
against revision `4146926` found no introduced lint findings; 90 pre-existing
findings are retained.
`uv lock --check --offline` and `git diff --check` also passed. The final lint
cleanup only changed a return annotation and an explicit public re-export marker.
