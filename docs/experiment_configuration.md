# Experiment configuration

The shared entry point is `.venv/bin/python -m moldr.cli`. Its commands are
`train`, `evaluate`, `sample`, `profile`, and `sweep`. New runs default to
E2E, PPO, fixed Mol2Vec, the fixed GuacaMol fragment table, actor selection,
QED, improvement rewards, and local logging. The RLlib ModelV2 API stack is
preserved. The policy hidden dimension is 256; the default PPO settings retain
batch 4000, minibatch 128, 30 update epochs, learning rate 5e-5, and gamma 0.99.
Run from the repository root so the supplied asset paths resolve.

Configuration precedence is built-in defaults, a YAML file, explicit CLI flags,
then ordered `--set section.key=value` overrides. YAML values are typed: unknown
keys, misspellings, invalid types, and unsupported combinations raise an error
before constructing workers. `--set` values use YAML syntax, including `true`,
`false`, `null`, numbers, and lists. `--dry-run` resolves configuration and checks
assets without starting training.

```bash
.venv/bin/python -m moldr.cli train --dry-run
.venv/bin/python -m moldr.cli train --config configs/short.yaml --dry-run \
  --set training.seed=2 --set selection.context=classic
.venv/bin/python -m moldr.cli train --config configs/short.yaml \
  --set output.run_name=mol2vec_seed0
```

| Group | Relevant settings |
| --- | --- |
| `environment` | `mode` (`e2e` or `legacy`), `base_smiles`, `atom_limit`, `max_episode_steps`, `reward`, `score_threshold`, `final_weight`, `max_attach`, `terminate_on_score_decrease` |
| `fragments` | `source` (`fixed` or `vae`), `path` or inline `smiles`, `proposer_dir`, `pool_size`, `recond_every` |
| `chemistry` | `map_storage`, `shared_preparation`, `bond_scan`, `candidate_representation`, `dedup_key`, `mol_transfer` |
| `scoring` | registered `objective`, `input_mode` (`smiles` or `mol`), `cache_size` |
| `representation` | `name` (`mol2vec` or `mpnn`), `trainable`, `model_path`, `obs_dim`, `layers`, `hidden`, `output`, graph capacities, atom chunking, checkpointing, cache sizes, `reuse_embeddings` |
| `selection` | `mode` (`actor` or `actor_value`), `context` (`rich` or `classic`), `candidate_value`, `auxiliary_loss_coeff`, `alpha`, `temperature` |
| `training` | seed, iterations, worker/CPU/GPU counts, PPO settings, Torch thread count |
| `evaluation` | `selector`, episodes, exploration, random baseline |
| `profiling` | warmup/measured iterations, optional measured time budget, trace, event capacity |
| `output` | directory, run name, checkpoint, checkpoint frequency, logger, project name |

`environment.max_episode_steps` counts decisions. Its derived default is twice
the atom limit for E2E and once the atom limit for legacy. A one-atom merge can
leave the molecule unchanged, so the decision limit also bounds those episodes.
Atom termination uses `atom_count > atom_limit`. Time-limit truncation retains
the current state and bootstraps its value. True termination pays the configured
completion bonus. The common sampler seeds the environment and selects from
the observation's live action mask; it returns the environment's actual final
molecule for terminal, truncated, and zero-candidate episodes.

The chemistry baseline uses integer atom-map properties, shared pair
preparation, nonzero bond scanning, mapped Kekulé intermediates, Kekulé
deduplication, and SMILES round trips. Integer maps retain the map on atom zero.
Shared preparation retains insertion order and leaves inputs unchanged. Nonzero
scanning reduces Python loops; adjacency matrix operations remain dense. The
mapless, aromatic-deduplication, and direct-transfer variants can change candidate
counts, ordering, representatives, and trajectories. Direct transfer requires
`candidate_representation=mapless`.

Scoring defaults to SMILES input with caching disabled. Named deterministic
objectives can cache pure computations. The service records requests,
computations, and hits separately. Mol input uses the registered Mol evaluator;
QED has a native Mol path. Fixed Mol2Vec reuses the unchanged base embedding and
the selected candidate embedding by default; the reuse switch supports a
comparison with repeated embedding work.

Rich context includes phase, current score, elapsed/remaining decisions, and a
masked aggregate of live option embeddings. Classic context uses base and phase.
`selection.mode=actor_value` requires `selection.candidate_value=true`. The
attachment logits are `actor_logits + alpha * stopgrad(U) / temperature`, then
the action mask is applied. The same distribution supplies sampling and PPO
log probabilities. U is regressed only for selected phase-B actions against PPO
`value_targets`, using Huber loss with default coefficient 0.5. Auxiliary-only
training enables the head with actor mode and alpha zero. Training recomputes U
from the training observations so additional value forwards cannot overwrite its
teacher pairing.

MPNN observations carry signed 32-bit integer atom/bond records, counts, and
fragment IDs for the base and completed candidates. This halves graph storage
relative to 64-bit integer records; Torch converts live indexing records to
64-bit internally. Completed-candidate buffers have axis
`max_attach`. Fragment options carry stable catalog IDs; a VAE uses its vocabulary
IDs. The policy owns the Torch encoder and only encodes live graphs. Three layers,
hidden/output 128, residual updates, LayerNorm, sum message aggregation, and a
projected mean-plus-sum readout are the supplied MPNN settings. Set
`representation.trainable=true` for joint PPO training; frozen encoders are a
separate ablation.

The factory derives conservative capacities from the initial molecule, atom
limit, and largest catalog fragment:

```text
A = max(initial_atom_count, atom_limit) + largest_fragment_atom_count
E = A * (A - 1) // 2
```

`max_atoms` and `max_bonds` overrides must cover those bounds. Capacity overflow
raises an error without dropping atoms or bonds. The conservative bond bound
can increase observation memory. Atom processing defaults to 4096 per chunk;
`gradient_checkpointing=true` is available for trainable encoders. Structural
caching remains valid while an encoder trains. Persistent embedding caching
requires a frozen encoder. A VAE's conditioning remains its independently
loaded frozen Mol2Vec even when the policy uses MPNN.

Fixed fragments are the standard configuration. VAE checkpoints trained
without molecular context must set `fragments.recond_every=never`; episode or
step conditioning requires a proposer explicitly marked as context-trained.
The factory rejects conditioning an untrained context input. A GNN policy's
output dimension is independent of the VAE's frozen Mol2Vec context dimension.

```bash
.venv/bin/python -m moldr.cli train --dry-run \
  --set fragments.source=vae --set fragments.proposer_dir=models/my_proposer \
  --set fragments.recond_every=never
```

The short comparison is described by [configs/short.yaml](../configs/short.yaml).
It uses batch 1024, minibatch 32, three PPO update epochs, atom limit 20,
attachment cap 128, score threshold 1.1, eight workers, one learner GPU, three
warmup iterations, and ten measured iterations. Every fragment is presented.
The score threshold keeps QED from ending episodes before the common atom or
decision limits. YAML overlays in `configs/ablations/` select individual
switches; standalone `--config` overlays start from built-in defaults. Sweep
jobs layer their overlay on the supplied short configuration.

| Overlay | Change from the common short configuration |
| --- | --- |
| `map_string.yaml` | String atom-map properties |
| `unshared_preparation.yaml` | Recompute preparation per candidate |
| `python_bond_scan.yaml` | Python matrix scanning |
| `mapless.yaml` | Mapless intermediate candidates |
| `aromatic_dedup.yaml` | Aromatic canonical deduplication keys |
| `direct_mapless.yaml` | Direct Mol transfer with required mapless intermediates |
| `mol_scoring.yaml` | Mol evaluator input |
| `score_cache.yaml` | Scoring LRU capacity 4096 |
| `no_embedding_reuse.yaml` | Recompute Mol2Vec state/candidate embeddings |
| `classic_context.yaml` | Base-and-phase policy context |
| `actor_value.yaml` | U loss 0.5, alpha 1, temperature 1 |
| `auxiliary_only.yaml` | U loss 0.5 with zero contribution to actor logits |
| `mpnn_frozen.yaml` | Three-layer MPNN, frozen encoder |
| `mpnn_trainable.yaml` | Three-layer MPNN, jointly trained encoder |

The supplied sweep manifests contain explicit named jobs. `short.yaml` lists 54
jobs: seeds 0/1/2 for the five major configurations and ten other ablations, then
baseline workers 1/2/4 for each seed. The worker-8 baseline already appears in
the major configurations. `longwall.yaml` lists 24 major/scaling jobs with a
3600-second measured budget and an iteration ceiling of 100000. Warmup is outside
that budget; checks happen after complete PPO iterations, so the last iteration
can extend elapsed time beyond the budget. `trace.yaml` lists five separate
seed-0 jobs with two workers, batch 128, one warmup, one measured iteration, and
an additional traced iteration. Minibatch 32, three PPO epochs, and the full
fragment catalog remain the same. Batch 128 reduces nominal minibatch updates
from 96 to 12 to limit exported Torch event volume; the short speed sweep keeps
batch 1024.

```bash
.venv/bin/python -m moldr.cli sweep --config configs/short.yaml \
  --sweep configs/sweeps/short.yaml --dry-run
.venv/bin/python -m moldr.cli sweep --config configs/short.yaml \
  --sweep configs/sweeps/short.yaml
.venv/bin/python -m moldr.cli sweep --config configs/short.yaml \
  --sweep configs/sweeps/longwall.yaml
.venv/bin/python -m moldr.cli sweep --config configs/short.yaml \
  --sweep configs/sweeps/trace.yaml
```

Jobs execute serially on the single learner GPU. Planning records each resolved
configuration difference in `sweep.json`. Sweeps validate every listed job
before training starts and track planned/running/complete/failed states. The
major configurations appear first in the short manifest to support prioritized
execution. Runtime sweeps never expand an implicit cross product. The longwall
manifest allocates 24 measured GPU-hours plus warmup, setup, and checkpoint time.
Training commands requesting CUDA require a visible working device; CPU dry
runs still validate the assets.

The comparison report groups completed seeds while retaining worker counts.
`comparison.json` preserves each trial; `grouped_comparison.json` records means,
sample standard deviations, contributing counts, and planned counts per group.
`index.html` shows completed/total jobs so interim reports remain clear.
Standalone PNG/SVG figures show throughput and training-quality proxies.
Rolling sampled uniqueness uses bounded environment histories that include
warmup. It is distinct from the post-training fingerprint diversity metric.
RLlib smoothed sample and learning timers appear as separate table columns.

Speed measurements run with tracing disabled. A trace-enabled profile command
records an additional iteration after speed measurement and emits merged
Chrome/Perfetto JSON, analysis JSON, and the existing HTML viewer. Scopes include
SMILES processing, candidate generation, objective requests/computations,
candidate values, and GNN encoding. A trace event-capacity overflow fails the
run rather than presenting an incomplete timing breakdown.
The supplied trace sweep also changes worker count from eight to two and batch
size from 1024 to 128. Its traced iteration durations and scope fractions describe
that configuration; they are not measurements of the batch-1024 speed jobs.

Each new training directory saves `resolved_config.yaml`, `manifest.json`,
`fragment_catalog.json`, copied fragment/model/proposer assets, local logs,
`measurements.json`, `summary.json`, checkpoints, and `last_checkpoint.json`.
The manifest includes code revision, working-tree status, Python/library
versions, seed, source hashes, and configuration/catalog/asset hashes. Measurement
records include molecular-update and valid-molecule rates, scoring counters,
episode-completion scores, rolling sampled-uniqueness proxies, worker/learner
timers, and peak allocated CUDA memory. The valid-molecule rate counts valid
selected updates. The retained `valid_molecules` counter counts valid selected
intermediate updates. `completed_valid_molecules` counts completed episodes and
`completed_valid_molecules_per_second` divides measured episode completions by
measured wall time. Each terminal, truncated, or dead-end episode contributes
its actual validated final base molecule. Reported GPU allocation concerns this
process's Torch allocator.

Evaluation and sampling directories use the same provenance writer: they save
resolved configuration, versions, code/asset hashes, catalog, and copied assets.
Checkpoint-based evaluation additionally writes `evaluation_checkpoint.json`
with the resolved checkpoint path and hashes of its files. `molecules.json`
stores each actual final molecule, elapsed episode time, score counters, and
lookahead preview/attachment counts. Direct Python
`run_evaluation(config, checkpoint=...)` applies the same saved-model protection
as the CLI, and leaves the caller's configuration unchanged.

A new checkpoint restores from its saved configuration and copied assets. The
loader verifies configuration, catalog, and asset hashes, rejects representation
or model-shape changes and vocabulary/model-asset overrides, and rewrites asset
paths to the saved copies. Source hashes and versions record provenance.
Changes to the original CSV or Mol2Vec file do not change a restored run's inputs.
Passing a run directory resolves `last_checkpoint.json`; a checkpoint directory
also works. The following sampling command is a separate process after training:

```bash
.venv/bin/python -m moldr.cli sample \
  --checkpoint outputs/experiments/mol2vec_seed0 --episodes 12 \
  --set training.num_workers=0 --set training.num_gpus=0
.venv/bin/python -m moldr.cli evaluate \
  --checkpoint outputs/experiments/mol2vec_seed0 --episodes 12 \
  --selector value_lookahead --set training.num_workers=0
```

`value_lookahead` evaluates each attachment as
`r + gamma * (1 - terminated) * V(next_state)`. True terminal states have zero
bootstrap; time-limit truncation keeps bootstrap. It requires E2E, a fixed
fragment table, every fragment presented, and a registered deterministic
objective. It preserves molecular state, phase, and RNG state during preview.
For K attachment candidates, previews plus the selected real transition make
K+1 score requests. The evaluation report includes this cost; caches can reduce
computations while preserving request counts. Actor selection still chooses
the fragment in phase A.

Compare both evaluation selectors on the same saved checkpoint with the same
starting seeds and episode budgets:

```bash
.venv/bin/python -m scripts.compare_evaluation \
  --checkpoint outputs/experiments/mol2vec_seed0 --episodes 3 --seeds 0 1 2 \
  --output artifacts/experiments/selector_comparison
```

This comparison restores saved configuration/assets through the shared factory
and runs CPU evaluation with zero rollout workers. It records objective quality,
validity, diversity, molecular-update rates, actual requests/computations/hits,
and verified K+1 score-request deltas for every lookahead attachment. Per-episode
rollout timing includes reset, selection, chemistry, and scoring; full run wall
time additionally includes startup, restoration, and output. `comparison.json`,
`comparison.png`, and `comparison.svg` retain the result. Each CLI run uses
`starting_seed + episode_index`, so starting-seed ranges can overlap. Deterministic
actor evaluation of a fixed table can repeat the same trajectory across seeds;
the report records those repetitions in its uniqueness/diversity metrics.

Evaluate the five major trained configurations with a common stochastic episode
budget after the speed sweep completes:

```bash
.venv/bin/python -m scripts.evaluate_checkpoints \
  --sweep-directory artifacts/experiments_short/YOUR_SWEEP \
  --output artifacts/experiments_posthoc --episodes 3 --evaluation-seed 0 \
  --num-gpus 1 --plan
.venv/bin/python -m scripts.evaluate_checkpoints \
  --sweep-directory artifacts/experiments_short/YOUR_SWEEP \
  --output artifacts/experiments_posthoc --episodes 3 --evaluation-seed 0 \
  --num-gpus 1
```

Planning starts no Ray processes and performs no evaluation. Execution waits
for the speed sweep and all fifteen major checkpoints to complete. It restores
the three training seeds for actor, actor-value, auxiliary-only, frozen MPNN,
and trainable MPNN sequentially with zero rollout workers. Every checkpoint uses
evaluation seeds 0/1/2 and `explore=true` for the shown budget. The sampler seeds
Torch stochastic selection explicitly per episode, so model-initialization
draws do not alter the episode seed. No additional learning occurs. Each run
stores score, validity, uniqueness, scoring counters, and radius-2 2048-bit
Morgan/Tanimoto diversity. Grouped means and sample standard deviations describe
per-checkpoint metrics across training seeds, with the episode budget and
contributing counts retained in JSON and PNG/SVG artifacts.

Explicit `--mode legacy` selects the old greedy candidate scorer. Its derived
defaults retain sparse completion rewards and score-decrease termination.
Old Python configuration functions remain accessible through `moldr.config`
and `moldr.legacy`. Old E2E checkpoints retain environment registration
`moldr_e2e` and model `moldr_pointer`; new contracts use `moldr_e2e_v2` and
`moldr_pointer_v2`. Old checkpoints use their existing loaders and saved
`env_config.pkl`. New restoration requires the experiment manifest and never
implicitly converts an old model.

These short sweeps assess implementation, throughput, and initial quality.
Measured results and real GPU validation must be read from produced run
artifacts; the supplied YAML is an executable experiment specification.
See [experiment results](experiment_results.md) for the explicitly partial
measurement snapshot and remaining comparisons.
