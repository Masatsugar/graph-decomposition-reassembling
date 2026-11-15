# Molecular Graph Generation by Decomposition and Reassembling
This repository is for [MOLDR: Molecular Graph Generation by Decomposition and Reassembling](https://pubs.acs.org/doi/10.1021/acsomega.3c01078). 

## Installation

### For Latest Version
Use Python 3.11–3.12.

```shell
pip install --upgrade uv
git clone https://github.com/Masatsugar/graph-decomposition-reassembling.git
cd graph-decomposition-reassembling
uv sync --python 3.12 --locked --extra dev
```

The lock file pins compatible Ray, NumPy, SciPy and PyArrow versions. SciPy
1.11.4 retains the import GuacaMol requires. The uv configuration excludes
GuacaMol's obsolete `rdkit-pypi` dependency and installs the maintained `rdkit`
distribution instead. No site-packages edits or manual uninstalls are needed.
Use a recent uv supporting
[`exclude-dependencies`](https://docs.astral.sh/uv/reference/settings/#exclude-dependencies).

On Linux and Windows, the uv configuration selects PyTorch's CUDA 12.6 build
from the [official PyTorch index](https://download.pytorch.org/whl/cu126).
GPU training requires a compatible NVIDIA driver; CPU training also works
with this build.


## Usage
Set python path: `export PYTHONPATH=.`.

Enable [Ray token authentication](https://docs.ray.io/en/latest/ray-security/token-auth.html)
in your shell before starting Python, `uv run`, or Jupyter:

```shell
export RAY_AUTH_MODE=token
```

Ray reads this setting when it is imported. Restart an existing Python process
or notebook kernel after setting it. Keep Ray management interfaces within a
trusted network; token authentication does not encrypt network traffic.

### Start here: a small working example

For a **Python-only walkthrough**, start with
the tutorial ([English](docs/tutorial.md) | [日本語](docs/tutorial_ja.md)). It
explains how to define a molecular score,
turn that score into PPO rewards, train a small CPU policy, and restore the
saved policy and custom objective in a new process:

```shell
uv run python examples/tutorial.py demo
uv run python examples/tutorial.py train --iterations 1 \
  --objective custom --target-atoms 10 --size-width 2 --qed-weight 0.3
uv run python examples/tutorial.py sample --checkpoint PATH_PRINTED_BY_TRAIN --episodes 3
```

Replace `PATH_PRINTED_BY_TRAIN` with the run directory printed by training;
the script also prints a complete sampling command. Edit
[`tutorial_objective.py`](examples/tutorial_objective.py) to adapt the score
to your research objective. Sampling restores its saved definition and
parameters without registering the function again. No Jupyter dependency is
needed for this route.

Open the quickstart notebook ([English](examples/quickstart.ipynb) | [日本語](examples/quickstart_ja.ipynb))
for a guided walkthrough: eight sample molecules → fast-gspan fragments → reassembly candidates →
one legacy episode → saved CSV results. The sample SMILES are included in
the notebook; no external dataset download is needed. The default cells use
CPU and the supplied Mol2Vec model with the original **legacy** environment:
the agent selects a building block, and the environment chooses the
highest-scoring reassembly candidate. The demo chooses building blocks at
random, using sparse rewards and termination on score decrease. The linked
Python tutorial uses the same defaults. A separate, opt-in section runs a small
CPU PPO training job in legacy mode and samples from its saved checkpoint.
Neither demonstration establishes molecular optimization quality.

After installation, run this from the repository root:

```shell
uv run --locked --with 'jupyterlab>=4,<5' --with 'ipykernel>=6,<7' \
  python -m jupyter lab examples/quickstart.ipynb
```

For the Japanese notebook, open `examples/quickstart_ja.ipynb` instead.
Select the Python kernel from that environment and run the cells in order.
The notebook requires this development version's shared `moldr.factory` and
`moldr.cli` modules. It explains where to substitute your own molecules and
objective, and how to restore checkpoints with their saved configuration.

### Training and sampling

The shared CLI (`python -m moldr.cli`) supports both `--mode legacy` (the
published greedy candidate-scoring environment) and `--mode e2e`
(**experimental** End-to-End two-phase policy selection, used as the CLI
default). New runs default to E2E + PPO + fixed Mol2Vec + a fixed fragment table
+ actor selection, with QED, improvement rewards, and local logging. Training,
evaluation, sampling, profiling, and explicit sweeps share a typed YAML
configuration and the existing RLlib ModelV2 API stack.

```shell
uv run python -m moldr.cli train --mode legacy --iterations 10 --num_workers 8 --num_gpus 1
uv run python -m moldr.cli train --mode e2e --iterations 10 --num_workers 8 --num_gpus 1
uv run python -m moldr.cli sample --checkpoint outputs/experiments/YOUR_RUN --episodes 12
```

Use the run directory printed by training for `--checkpoint`. See
[experiment configuration](docs/experiment_configuration.md) for defaults,
overrides, sweeps, candidate values, MPNN, and restoration.
`--mode legacy` selects the original greedy candidate scorer. The old Python
factories in `moldr.config` and `moldr.legacy` retain their legacy behavior,
and old E2E registration names remain available for existing checkpoints.
The bundled Mol2Vec model now uses its training radius of 1, recorded as
`representation.radius`. Runs saved before radius tracking must be restored
with their original source revision or retrained; the loader rejects them
instead of changing the observations supplied to a trained policy. New runs
also sanitize candidates before serialization and deduplicate aromatic
structures, so candidate counts and trajectories can differ from earlier runs.

Ray 2.56 no longer includes logger classes embedded in older checkpoints. If
restoration fails with `No module named 'ray.tune.logger.unified'`, use the
original environment to copy and migrate your own trusted run, then restore
the copy with the updated environment:

```shell
/path/to/original-environment/bin/python scripts/migrate_ray_checkpoint.py \
  outputs/experiments/OLD_RUN outputs/experiments/MIGRATED_RUN
uv run python -m moldr.cli sample --checkpoint outputs/experiments/MIGRATED_RUN
```

The destination must not exist. Migration removes only saved logger
configuration and adjusts copied checkpoint paths; it preserves the original
run, model weights, and optimizer state. It does not migrate older observation
formats or make untrusted pickle files safe to load.

### Decomposition
In the decomposition step, molecules are decomposed into subgraphs using
[fast-gspan](https://github.com/Masatsugar/fast-gspan), backed by gBolt C++.
Choose the "raw" decomposition method for greater versatility, or "jt" (junction tree) to take cliques into account.


```python
from pathlib import Path
from moldr.core.decompose import MolsMining, DefaultConfig
from moldr.chemutils import get_mol

test_smiles = [
    "CC1CCC2=CC=CC=C2O1",
    "CC",
    "COC",
    "c1ccccc1",
    "CC1C(=O)NC(=O)S1",
    "CO",
]
mols = [get_mol(s) for s in test_smiles]
minsup = max(1, int(len(mols) * 0.1))
config = DefaultConfig(
    data_path=Path("zinc_jt.data"),
    support=minsup,
    lower=2,
    upper=7,
    method="jt",
    backend="fast-gspan",
    num_threads=4,
)
runner = MolsMining(config)
gspan = runner.decompose(mols)
```

`fast-gspan` is installed by `uv sync` and is the default backend, including
in the quickstart notebook. Existing `MolsMining`, molecule reconstruction,
pickle saving, and fragment CSV calls continue to work. `support` counts
distinct input graphs; the report's comma-separated `support_ids` lists each
supporting graph once (the Python backend listed individual embeddings).
`num_threads=0` uses all cores; `timeout` sets a mining timeout in seconds,
and `show_progress=True` displays progress.

Use `backend="python"` to select the original bundled implementation.
The fast backend accepts undirected graphs. With fast-gspan 0.1.x, requests
below its effective 1% support floor raise an error; increase `support` or
select the Python backend for those requests. If your platform requires a
source build, run `uv run python -m fast_gspan build` with CMake and a C++
compiler installed.

If you want to see the subgraphs in detail, see
[the decomposition notebook](examples/decomposition.ipynb).

### Reassembling
- Training 

Building blocks come from the decomposition step. In the shared CLI, the
**experimental** E2E environment (`--mode e2e`, default) lets PPO select a
building block and then an attachment. The original published environment lets
the agent select a building block and greedily retains the highest-scoring
candidate; this behavior is available through `--mode legacy` and the existing
Python factories (`moldr.config` / `moldr.legacy`). The shared CLI uses PPO.

We already mined the building blocks from gucamol or zinc dataset. The default is guacamol with minsup 10,000.
Set your own dataset to generate building blocks if you want to use other datasets.

```shell
uv run python -m moldr.train --epochs 10 --num_workers 8 --num_gpus 1
```

For a short run without an external experiment tracker:

```shell
uv run python -m moldr.train --objective QED --epochs 1 --num_workers 2 \
  --num_gpus 1 --logger local --train-batch-size 128 \
  --minibatch-size 64 --num-epochs 1 --length 12
```

Generated molecules are sampled through the trained agent and the run's saved
configuration and assets:

```shell
uv run python -m moldr.cli sample --checkpoint outputs/experiments/YOUR_RUN
```

Registered objectives are selected with `--objective`; QED is the default.
Custom Python evaluators can be registered through
`moldr.scoring.register_objective`. Existing `run_moldr.py` and the legacy
configuration functions remain compatibility entry points for earlier runs.


## Faster Reassembly and Experimental Extensions

The sections below describe the chemistry speedup and experimental E2E, MPNN,
and generative extensions along with their Python APIs. The original legacy
and experimental E2E environments share the corrected merge implementation;
the CLI uses versioned observations/models while these older API examples keep
their original contracts.

> **Note (Experimental):** The End-to-End (`moldr.e2e`), MPNN (`moldr.gnn`), and
> generative building-block (`moldr.generative`) features below are experimental
> extensions. Use `--mode legacy` (or the legacy Python factories) for the
> original greedy MOLDR workflow.

### 1. Faster reassembly (transparent)

The merge/reassembly step no longer round-trips molecules through networkx and
SMILES on every candidate; it uses direct numpy/RWMol operations
(`moldr.chemutils.mol_to_adj` / `mol_from_adj`). The generated molecule sets are
identical for the tested neutral, non-stereochemical fixtures
(see `tests/test_reassemble_fast.py`). Bug fixes additionally retain charges
and isotopes, use actual bonds in bridged rings, and always return a list from
`merge_edge`. These corrected chemistry cases intentionally differ from the
old results. Both environments now impose an episode step limit and report
time limits as truncations for PPO bootstrapping.

### 2. End-to-end reassembly (Experimental)

In the original `MolEnvValueMax`, the agent only chose *which* building block to
attach; the environment then enumerated every merge, scored them all, and
greedily kept `argmax(score)`. `moldr/e2e` makes that attachment choice *learned*
instead, as a two-phase MDP (select building block → select attachment) driven by
a single action-masked **pointer policy**, while keeping RLlib PPO. Only the
chosen candidate is scored, reducing objective evaluations per attachment.

```python
from moldr.e2e.config import get_e2e_config
from guacamol.standard_benchmarks import logP_benchmark

config = get_e2e_config(
    logP_benchmark(8.0),
    building_blocks_smiles,  # same fixed table as before
    model_path="models/model_300dim.pkl",
    num_workers=8,
    num_gpus=1,
)
algo = config.build()
for _ in range(epochs):
    algo.train()
```

### 3. Generative building blocks (Experimental)

`moldr/generative` replaces the fixed mined table with a learned **fragment VAE
proposer**. A vocabulary-pointer decoder guarantees chemical validity by
construction (it can only emit pre-validated fragments), and `N` frozen latent
anchors keep the action space a fixed `Discrete(N)` so PPO is unchanged. The
pool can be re-decoded conditioned on the current molecule (`recond_every` ∈
`{"never", "episode", "step"}`).

For the shared CLI, an unconditional VAE must use
`--set fragments.recond_every=never`. Episode or step reconditioning requires
training with molecular contexts and a saved context-trained proposer.

```python
from moldr.generative.producer import build_proposer_from_smiles

# Train a proposer from the existing building-block CSVs (or full molecules
# with decompose=True). For real runs raise epochs/hidden_dim on a GPU.
proposer = build_proposer_from_smiles(
    building_blocks_smiles,
    n_blocks=1000,
    device="cuda",
    hidden_dim=256,
    batch_size=256,
    epochs=50,
)
# Keep rollout workers on CPU: save the CUDA-trained model and pass its spec.
proposer.save("tmp/proposer")
proposer_spec = {"dir": "tmp/proposer"}
```

Use it in either environment:

```python
# (a) generative blocks + original reassembly (ablation vs the gSpan table)
from moldr.config import get_generative_config_v1

config = get_generative_config_v1(
    objective, proposer_spec, model_path="models/model_300dim.pkl"
)

# (b) generative blocks + end-to-end reassembly (combines 2 and 3)
from moldr.e2e.config import get_e2e_config

config = get_e2e_config(
    objective, model_path="models/model_300dim.pkl", proposer=proposer_spec
)
```

Run the unit and integration test suite with:

```shell
uv run python -m pytest -q
```

Fragment proposals select entries from the existing vocabulary; unseen
fragments are not created. Without per-fragment `contexts`, VAE training is
unconditional and requires `fragments.recond_every=never`. Specified
tetrahedral or bond stereochemistry is currently unsupported by the adjacency
merge implementation: environments reject it, and fragment vocabularies filter
it out rather than silently discarding it.

### Profiling and Chrome/Perfetto traces (Experimental)

Capture Chrome/Perfetto timelines and trace breakdowns via the shared CLI:

```shell
uv run python -m moldr.cli profile --trace --num_workers 2 --train-batch-size 128
```

This writes `trace.json`, `trace_analysis.json`, and an offline interactive
`timeline.html` viewer into the run output directory. Each `trace.json` also
opens in Perfetto or Chrome Tracing (`chrome://tracing`). The collector records
CPU chemistry/scoring/embedding scopes and Torch/CUDA profiler events on a
shared single-host clock.


### For Paper Reproducibility

To reproduce the exact results from our paper, please use the specific version tagged for the publication:

```shell
git clone https://github.com/Masatsugar/graph-decomposition-reassembling.git
cd graph-decomposition-reassembling
git checkout v1.0.0-paper
conda env create --file env.yaml
conda activate moldr
pip install -e mi-collections/
```

**Note**: The v1.0.0-paper tag contains the exact code and dependencies used in the published paper to ensure results.


### Citation

```bib
@article{Yamada2023MOLDR,
  author={Yamada, Masatsugu and Sugiyama, Mahito},
  title={Molecular Graph Generation by Decomposition and Reassembling},
  journal={ACS Omega},
  year={2023},
  month={Jun},
  day={06},
  publisher={American Chemical Society},
  volume={8},
  number={22},
  pages={19575-19586},
  doi={10.1021/acsomega.3c01078},
  url={https://doi.org/10.1021/acsomega.3c01078}
}
```
