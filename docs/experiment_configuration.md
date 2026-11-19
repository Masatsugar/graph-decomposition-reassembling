# Experiment configuration

> **Note (Experimental):** End-to-End mode (`environment.mode=e2e`), candidate-value selection (`selection.mode=actor_value`), MPNN encoders (`representation.name=mpnn`), and fragment VAE proposers (`fragments.source=vae`) are **experimental** features. Pass `--mode legacy` for the original greedy candidate scorer.

The shared entry point is `.venv/bin/python -m moldr.cli`. Its commands are
`train`, `evaluate`, `sample`, `profile`, and `sweep`. New runs default to
E2E (experimental), PPO, fixed Mol2Vec (`radius=1`), the fixed GuacaMol
fragment table, actor selection, QED, improvement rewards, and local logging.
The RLlib ModelV2 API stack is preserved. The policy hidden dimension is 256;
the default PPO settings retain batch 4000, minibatch 128, 30 update epochs,
learning rate 5e-5, and gamma 0.99. Run from the repository root so the
supplied asset paths resolve.

Configuration precedence is built-in defaults, a YAML file, explicit CLI flags,
then ordered `--set section.key=value` overrides. YAML values are typed: unknown
keys, misspellings, invalid types, and unsupported combinations raise an error
before constructing workers. `--set` values use YAML syntax, including `true`,
`false`, `null`, numbers, and lists. `--dry-run` resolves configuration and checks
assets without starting training.

```bash
.venv/bin/python -m moldr.cli train --dry-run
.venv/bin/python -m moldr.cli train --mode legacy --dry-run --set training.seed=2
.venv/bin/python -m moldr.cli train --mode e2e --iterations 10 --set output.run_name=my_e2e_run
```

| Group | Relevant settings |
| --- | --- |
| `environment` | `mode` (`e2e` or `legacy`), `base_smiles`, `atom_limit`, `max_episode_steps`, `reward`, `score_threshold`, `final_weight`, `max_attach`, `terminate_on_score_decrease` |
| `fragments` | `source` (`fixed` or `vae`), `path` or inline `smiles`, `proposer_dir`, `pool_size`, `recond_every` |
| `chemistry` | `backend`, `map_storage`, `shared_preparation`, `bond_scan`, `candidate_representation`, `dedup_key`, `mol_transfer` |
| `scoring` | registered `objective`, `input_mode` (`smiles` or `mol`), `cache_size` |
| `representation` | `name` (`mol2vec` or `mpnn`), `trainable`, `model_path`, `obs_dim`, `radius`, `layers`, `hidden`, `output`, graph capacities, atom chunking, checkpointing, cache sizes, `reuse_embeddings` |
| `selection` | `mode` (`actor` or `actor_value`), `context` (`rich` or `classic`), `candidate_value`, `auxiliary_loss_coeff`, `alpha`, `temperature` |
| `training` | seed, iterations, worker/CPU/GPU counts, PPO settings, `object_store_memory_mib`, Torch thread count |
| `evaluation` | `selector`, episodes, exploration, random baseline |
| `profiling` | warmup/measured iterations, optional measured time budget, `trace`, `max_events` |
| `output` | directory, run name, checkpoint, checkpoint frequency, logger, project name |

## Environment and chemistry defaults

`environment.max_episode_steps` counts decisions. Its derived default is twice
the atom limit for E2E and once the atom limit for legacy. A one-atom merge can
leave the molecule unchanged, so the decision limit also bounds those episodes.
Atom termination uses `atom_count > atom_limit`. Time-limit truncation retains
the current state and bootstraps its value. True termination pays the configured
completion bonus. The common sampler seeds the environment and selects from
the observation's live action mask; it returns the environment's actual final
molecule for terminal, truncated, and zero-candidate episodes.

By default, candidate generation uses `map_storage=int`,
`shared_preparation=true`, `bond_scan=nonzero`,
`candidate_representation=mapless`, `dedup_key=aromatic`, and
`mol_transfer=direct`. Candidates are sanitized before serialization and
deduplicated by canonical aromatic SMILES so Kekulé resonance forms do not create
duplicate actions and valid fused-ring products are retained. Alternative flags
such as `mapped` / `kekule` / `smiles` remain configurable. Direct transfer
(`mol_transfer=direct`) requires `candidate_representation=mapless`.

### Experimental C++ candidate generation

The default backend is `python`. To try `--set chemistry.backend=cpp` on
Linux, build the extension in the checkout with the same Python interpreter
that runs the experiment. Use a C++17 compiler and the conda-forge headers
from `librdkit-dev=2025.03.6` and `libboost-headers=1.85.0`. Set
`SDK_PREFIX` to the directory containing their `include/` tree. The build
links against the project's installed RDKit wheel.

```sh
.venv/bin/python scripts/build_native_reassembly.py \
  --rdkit-include "$SDK_PREFIX/include/rdkit" --boost-include "$SDK_PREFIX/include"
.venv/bin/python -m pytest -q tests/test_native_reassembly.py
.venv/bin/python -m scripts.benchmark_native_reassembly
```

Both legacy and E2E runs accept the setting. Each default-path call sends one
base molecule and one fragment to C++ and returns the sorted candidates and
generation statistics. Special bond types and chemistry ablations that select
a different construction or transfer path use the existing Python implementation.

To measure the effect on PPO, run the paired benchmark from the same checkout.
It trains fresh policies with both backends for each seed, reverses their order
for the second seed, and records elapsed time for sampling, learner updates and
the full training iteration. The defaults are eight rollout workers, 1,024
environment steps per iteration, four SGD epochs, and four measured iterations
after one warm-up. The GuacaMol fragment pool is sampled from the full catalog
with 128 options per block-selection decision. Startup times and per-iteration
records are included in the JSON output.

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.benchmark_ppo_reassembly --num-gpus 0 \
  --output outputs/native-reassembly/ppo.json
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.benchmark_ppo_reassembly --num-gpus 1 \
  --output outputs/native-reassembly/ppo-gpu.json
```

## Scoring and representation defaults

Scoring defaults to `objective=QED`, `input_mode=smiles`, and `cache_size=4096`.
Named deterministic objectives cache pure evaluations by canonical key, and
built-in objectives normalize SMILES and RDKit `Mol` inputs identically. The
service records `requests`, `computations`, and `hits` separately.
`input_mode=mol` uses the registered Mol evaluator (QED and built-in
objectives have native sanitized `Mol` paths). Fixed Mol2Vec reuses the
unchanged base embedding and selected candidate embedding by default
(`reuse_embeddings=true`).

Representation defaults to `name=mol2vec` and `radius=1` (matching the bundled
`models/model_300dim.pkl` vocabulary). When loading saved runs whose policy or
VAE uses Mol2Vec, checkpoints predating explicit `representation.radius` tracking
are rejected instead of silently changing policy observations.

## Experimental E2E selection, MPNN, and VAE options

Rich context includes phase, current score, elapsed/remaining decisions, and a
masked aggregate of live option embeddings. Classic context uses base and phase.
`selection.mode=actor_value` requires `selection.candidate_value=true`. The
attachment logits are `actor_logits + alpha * stopgrad(U) / temperature`, then
the action mask is applied. The same distribution supplies sampling and PPO
log probabilities. `U` is regressed only for selected phase-B actions against PPO
`value_targets`, using Huber loss with default coefficient 0.5. Auxiliary-only
training enables the head with `selection.mode=actor`, `candidate_value=true`,
and `alpha=0`. Training recomputes `U` from the training observations so
additional value forwards cannot overwrite its teacher pairing.

MPNN observations (`representation.name=mpnn`) carry signed 32-bit integer
atom/bond records, counts, and fragment IDs for the base and completed
candidates. Torch converts live indexing records to 64-bit internally.
Completed-candidate buffers have axis `max_attach`. Fragment options carry
stable catalog IDs (or vocabulary IDs for VAE). The policy owns the Torch
encoder and only encodes live graphs. Default MPNN settings use three layers,
hidden/output 128, residual updates, LayerNorm, sum message aggregation, and a
projected mean-plus-sum readout. Set `representation.trainable=true` for joint
PPO training or `false` for a frozen encoder.

The factory derives conservative graph capacities from the initial molecule,
atom limit, and largest catalog fragment:

```text
A = max(initial_atom_count, atom_limit) + largest_fragment_atom_count
E = A * (A - 1) // 2
```

`max_atoms` and `max_bonds` overrides must cover those bounds. Capacity overflow
raises an error without dropping atoms or bonds. Atom processing defaults to
4096 per chunk; `gradient_checkpointing=true` is available for trainable
encoders. Structural caching remains valid while an encoder trains; persistent
embedding caching requires a frozen encoder. A VAE's conditioning remains its
independently loaded frozen Mol2Vec even when the policy uses MPNN.

Fixed fragments are the standard configuration. VAE checkpoints trained
without molecular context must set `fragments.recond_every=never`; episode or
step conditioning requires a proposer explicitly marked as context-trained.
The factory rejects conditioning an untrained context input.

```bash
.venv/bin/python -m moldr.cli train --dry-run \
  --set fragments.source=vae --set fragments.proposer_dir=models/my_proposer \
  --set fragments.recond_every=never
```

## Sweeps and profiling

`moldr.cli sweep` executes an explicit YAML manifest serially (no implicit
Cartesian product expansion). A sweep YAML file defines `command` (`train` or
`profile`) and `jobs`, where each job specifies a unique `name` plus optional
`config` and `set` override list:

```yaml
command: train
jobs:
  - name: baseline_seed0
    set:
      - training.seed=0
      - training.iterations=10
```

Run a dry-run validation before executing a sweep:

```bash
.venv/bin/python -m moldr.cli sweep --sweep path/to/sweep.yaml --dry-run
.venv/bin/python -m moldr.cli sweep --sweep path/to/sweep.yaml
```

`profile` runs `profiling.warmup_iterations` warmup iterations followed by
`profiling.measured_iterations` (or until `profiling.time_budget_seconds`) with
tracing disabled. When `profiling.trace=true` (or `--trace`), an additional
traced iteration runs afterward and writes merged Chrome/Perfetto JSON
(`trace.json`), trace breakdown (`trace_analysis.json`), and a standalone
interactive viewer (`timeline.html`) into the run directory. A trace
`max_events` overflow raises an error rather than emitting an incomplete trace.

## Provenance and checkpoint restoration

Each new training directory saves `resolved_config.yaml`, `manifest.json`,
`fragment_catalog.json`, copied fragment/model/proposer assets, local logs,
`measurements.json`, `summary.json`, checkpoints, and `last_checkpoint.json`.
The manifest includes code revision, working-tree status, Python/library
versions, seed, source hashes, and configuration/catalog/asset hashes.
Measurement records include molecular-update and valid-molecule rates, scoring
counters, episode-completion scores, rolling sampled-uniqueness proxies,
worker/learner timers, and peak allocated CUDA memory. `valid_molecules` counts
valid selected intermediate updates, and `completed_valid_molecules` /
`completed_valid_molecules_per_second` count completed episodes across terminal,
truncated, and dead-end outcomes.

Evaluation and sampling directories use the same provenance writer.
Checkpoint-based evaluation additionally writes `evaluation_checkpoint.json`
with the resolved checkpoint path and hashes of its files. A new checkpoint
restores from its saved configuration and copied assets, verifying hashes and
rejecting representation/model-shape or asset overrides.

```bash
.venv/bin/python -m moldr.cli sample \
  --checkpoint outputs/experiments/YOUR_RUN --episodes 12 \
  --set training.num_workers=0 --set training.num_gpus=0
.venv/bin/python -m moldr.cli evaluate \
  --checkpoint outputs/experiments/YOUR_RUN --episodes 12 \
  --selector value_lookahead --set training.num_workers=0
```

`value_lookahead` evaluates each attachment as
`r + gamma * (1 - terminated) * V(next_state)`. True terminal states have zero
bootstrap; time-limit truncation keeps bootstrap. It requires E2E, a fixed
fragment table, every fragment presented, and a registered deterministic
objective. It preserves molecular state, phase, and RNG state during preview.
For `K` attachment candidates, previews plus the selected real transition make
`K + 1` score requests.

Explicit `--mode legacy` selects the original greedy candidate scorer. Its
derived defaults retain sparse completion rewards and score-decrease
termination. Old Python configuration functions remain accessible through
`moldr.config` and `moldr.legacy`. Old E2E checkpoints retain environment
registration `moldr_e2e` and model `moldr_pointer`; new contracts use
`moldr_e2e_v2` and `moldr_pointer_v2`. Old checkpoints use their existing
loaders and saved `env_config.pkl`. New restoration requires the experiment
manifest and never implicitly converts an old model.
