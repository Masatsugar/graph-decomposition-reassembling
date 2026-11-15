# Python-Only MOLDR Tutorial

Language: **English** | [日本語](tutorial_ja.md)

This walkthrough takes you from running a small smoke test to configuring a molecular score aligned with your research goal and sampling molecules from a trained policy.
No Jupyter installation is required. The CLI entry point is [examples/tutorial.py](../examples/tutorial.py), and the customizable objective lives in [examples/tutorial_objective.py](../examples/tutorial_objective.py).

This tutorial uses the shared factory / CLI layer and defaults to the original **legacy** environment.
Make sure you run the matching source revision as this tutorial.
Complete the Installation steps in the README first, then run the commands below from the repository root.

## 1. Run One Episode Without Training

```bash
uv run python examples/tutorial.py demo
```

This generates a molecule on CPU using the bundled 13-fragment table and the fixed Mol2Vec model.
It does not start Ray or run PPO training. Because building blocks are chosen uniformly at random, these outputs do not reflect trained-model quality.

In `legacy` mode, each step follows: **pick a building block → score every reassembly candidate in the environment → keep the highest-scoring candidate**. During training, PPO also chooses only the building block. At most 8 steps corresponds to at most 8 reassemblies.
The demo prints the SMILES, score, and reward after each step, followed by the output directory path.
Check the saved `trajectory.csv` and `molecules.json` to see why the episode ended.
By default, outputs are written to a new run directory under `outputs/tutorial/`.

To try the **experimental** E2E mode instead, pass `--mode e2e` to `demo` or `train`.
E2E splits each reassembly into two agent decisions (**Phase A: pick a building block → Phase B: pick a reassembly candidate**), so 8 steps typically corresponds to 4 reassemblies.

| File | Contents |
| --- | --- |
| `objective_preview.csv` | Sample molecules and their scores checked before running |
| `trajectory.csv` | Resulting molecule, reward, and termination flags at each decision step |
| `molecules.json` | Generated episode outputs |
| `tutorial_settings.json` | Tutorial options recorded for the run |
| `resolved_config.yaml` | Fully resolved experiment configuration |
| `manifest.json`, `fragment_catalog.json`, `assets/` | Saved configuration and assets required for restoration |

You can inspect all flags with `--help`:

```bash
uv run python examples/tutorial.py --help
uv run python examples/tutorial.py train --help
uv run python examples/tutorial.py sample --help
```

When running from outside the repository root, invoke `.venv/bin/python` and `examples/tutorial.py` using absolute paths. Relative paths passed to `--fragments`, `--checkpoint`, and `--output` are resolved from your current working directory.

## 2. Adapt the Molecular Score to Your Objective

First, choose the **score `S(m)`** used to evaluate each molecule `m`.
In this tutorial, larger scores indicate better molecules.
The default custom objective computes a weighted average of QED and proximity to a target heavy-atom count:

```text
size_score(m) = exp(-0.5 * ((heavy_atoms(m) - target_atoms) / size_width)^2)
S(m) = qed_weight * QED(m) + (1 - qed_weight) * size_score(m)
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--target-atoms` | `8` | Target heavy-atom count (non-hydrogen atoms) |
| `--size-width` | `4` | Tolerance around the target atom count (smaller is stricter) |
| `--qed-weight` | `0.5` | Weight on QED in `[0, 1]`; `1 - qed_weight` is assigned to `size_score` |

For example, to prioritize molecules around 10 heavy atoms:

```bash
uv run python examples/tutorial.py demo \
  --objective custom --target-atoms 10 --size-width 2 --qed-weight 0.3
```

Inspect the printed sample scores and `objective_preview.csv` to verify that the molecules you prefer receive higher scores.
This example score is deterministic and finite within `[0, 1]`. Combining QED with atom count is purely illustrative; choosing metrics and weights appropriate for your research problem requires domain judgment.

To optimize pure QED instead, select the built-in objective:

```bash
uv run python examples/tutorial.py demo --objective QED
```

To plug in your own metric, edit the scoring expression in `tutorial_objective.py`.
Register a function mapping SMILES to float with `moldr.scoring.register_objective` and pass that registered name to `scoring.objective`.
This example also registers `mol_function` to score RDKit `Mol` objects directly.
Keep return values finite (never `NaN` or infinite) and oriented so higher values are better.
Because the same molecule always yields the same value, it passes `deterministic=True`.
If your custom metric uses a different numeric range, also update `environment.score_threshold=1.1` in `tutorial.py` to match your desired termination threshold, keeping in mind that score magnitude directly affects RL reward scaling.

## 3. Distinguish Molecular Score from RL Reward

The **reward `r`** passed to PPO is computed by the environment from the molecular score.
Registering a score function is independent from choosing how transitions turn scores into rewards:

| Situation | `--reward improvement` | `--reward sparse` |
| --- | --- | --- |
| Adopt highest-scoring reassembly candidate and continue | `S(new) - S(previous)` | `0` |
| Reassemble and satisfy an environment termination condition | Above + `final_weight * S(new)` | `final_weight * S(new)` |
| No valid reassembly candidates remain (terminated) | `final_weight * S(previous)` | `final_weight * S(previous)` |
| Step limit reached only (`truncated`) | No completion bonus added | No completion bonus added |

For example, when `S(previous)=0.4`, `S(new)=0.6`, and `final_weight=1`, the step improvement reward is `0.2`.
If that transition triggers true termination, the reward becomes `0.2 + 0.6 = 0.8`; if it is only truncated by the step limit, the reward remains `0.2`.
Differences between `score` and `reward` in saved trajectories reflect this rule.

```bash
uv run python examples/tutorial.py demo --reward improvement --final-weight 1
uv run python examples/tutorial.py demo --reward sparse --final-weight 0.5
```

In `legacy` mode, the defaults are `sparse` reward and termination when score decreases.
`terminated` indicates termination due to score decrease, score threshold, atom-limit exceeded, or zero candidates, whereas `truncated` indicates hitting `max_episode_steps`.
The tutorial sets `score_threshold=1.1` so `[0, 1]` scores never trigger early threshold termination.
Atom termination checks `atom_count > atom_limit`, so a final molecule may slightly exceed `atom_limit`.
With `sparse` rewards, episodes that only end by truncation receive zero total reward. Use `--reward improvement` if you want intermediate score changes rewarded at each step.
Combining `sparse` with `final_weight=0` makes rewards zero everywhere, including terminal steps—keep `final_weight > 0` when training with `sparse`.

With `--mode e2e` (**experimental**), the default reward is `improvement` and score-decrease termination is disabled. Phase A returns `0` whenever valid candidates exist, and Phase B applies the table above upon reassembly.

## 4. Run a Short CPU Training Job

Pass the scoring settings verified in the demo to `train`:

```bash
uv run python examples/tutorial.py train --iterations 1 \
  --objective custom --target-atoms 10 --size-width 2 --qed-weight 0.3
```

This runs a lightweight CPU configuration: 0 GPUs, 0 rollout workers, batch size 64, minibatch 32, 1 SGD epoch, and policy hidden width 32.
Even with `num_workers=0`, training still starts local Ray.
A single iteration is only a smoke check of training, checkpoint saving, and restoration—it is not sufficient for evaluating molecular optimization quality.

Note the run directory printed at completion, e.g. `outputs/tutorial/train_..._seed7`.
In addition to configuration and assets, training writes:

| File | Contents |
| --- | --- |
| `measurements.json` | Per-iteration losses, timings, and metrics |
| `summary.json` | Training step summary and weight updates |
| `checkpoints/iteration_000001/` | Trained policy checkpoint after iteration 1 |
| `last_checkpoint.json` | Path to the most recent checkpoint |
| `logs/` | Training logs |

Your custom objective function and captured parameters are saved into `assets/objective.pkl`.
Because values like target atom count and weights are captured in the scoring closures, editing `tutorial_objective.py` later will not silently alter how an already-saved run scores molecules.
Keep RDKit and matching dependencies available in the Python environment used for sampling.

## 5. Restore and Sample the Saved Model in a Separate Process

From a fresh terminal, point `sample` at the run directory printed by `train`:

```bash
uv run python examples/tutorial.py sample \
  --checkpoint outputs/tutorial/train_ACTUAL_TIMESTAMP_seed7 --episodes 3
```

Replace `train_ACTUAL_TIMESTAMP_seed7` with the actual directory name.
`sample` reloads the saved configuration, fragment table, Mol2Vec model, custom objective, and policy weights automatically.
You do not need to re-register the custom objective or reconstruct the policy network manually, and sampling evaluates with the saved environment configuration.

Sampling writes `molecules.json` and `molecules.csv` (SMILES, score, reward, termination flags, etc.) alongside `evaluation.json` (mean score, validity rate, diversity, and related metrics).
Sampling 3 episodes verifies restoration wiring; for research comparisons, match episode count, seeds, and search bounds against a random baseline.

Preserve or move the **entire run directory**, not just the `checkpoints/` subdirectory.
Because restoration reads self-contained copies under `assets/`, moved runs continue working even if original input paths change.
You cannot swap the fragment table or observation dimension after training.
`sample` also restores `environment.mode` automatically: runs trained without `--mode e2e` restore as `legacy`, while `train --mode e2e` runs restore as `e2e` and cannot be mixed.
Older checkpoints created without a shared-CLI manifest require their matching legacy loaders.

## 6. Substitute Your Own Molecules and Building Blocks

The default 13 building blocks were mined from the toy sample in [quickstart.ipynb](../examples/quickstart.ipynb).
`--fragments` expects a **building-block table for reassembly**, not raw target molecules to decompose automatically. To mine building blocks from your own molecules first, follow the Python snippet in [README Decomposition](../README.md#decomposition), or inspect mined structures visually in the notebook.

For example, create `my_fragments.smi`:

```text
CC
CO
CN
C1CCCCC1
```

Then run:

```bash
uv run python examples/tutorial.py demo \
  --fragments my_fragments.smi --base-smiles CCO --atom-limit 16 --max-steps 12
```

If using CSV, provide a leading index column followed by the `smiles` column:

```csv
,smiles
0,CC
1,CO
2,CN
```

Verify loading and scoring on a small molecule set before scaling up.
The current reassembly implementation does not handle explicit atomic or bond stereochemistry, and the environments reject inputs containing it.
Once satisfied with the demo, run `train` with your updated settings and pass the resulting run directory to `sample --checkpoint`.

For complete CLI and YAML configuration options, see [experiment_configuration.md](experiment_configuration.md).
