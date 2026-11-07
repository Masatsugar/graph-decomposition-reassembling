# Molecular Graph Generation by Decomposition and Reassembling
This repository is for [MOLDR: Molecular Graph Generation by Decomposition and Reassembling](https://pubs.acs.org/doi/10.1021/acsomega.3c01078). 

## Installation

### For Latest Version
Use Python 3.10.13–3.12 (the GPU validation uses Python 3.12).

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


## Usage
Set python path: `export PYTHONPATH=.`.

### Start here: a small working example

Open [the quickstart notebook](examples/quickstart.ipynb) for a guided Japanese
walkthrough: eight sample molecules → gSpan fragments → reassembly candidates →
one end-to-end episode → saved CSV results. The sample SMILES are included in
the notebook; no external dataset download is needed. The default cells use
CPU and the supplied Mol2Vec model, with valid actions chosen at random to
check the pipeline. A separate, opt-in section runs a small CPU PPO training
job and samples from its saved checkpoint. Neither demonstration establishes
molecular optimization quality.

After installation, run this from the repository root:

```shell
uv run --locked --with 'jupyterlab>=4,<5' --with 'ipykernel>=6,<7' \
  python -m jupyter lab examples/quickstart.ipynb
```

Select the Python kernel from that environment and run the cells in order.
The notebook requires this development version's shared `moldr.factory` and
`moldr.cli` modules; older checkouts with only the original E2E API are not
sufficient. It explains where to substitute your own molecules and objective,
and how legacy, initial E2E, and E2E v2 checkpoints differ.

### Training and sampling

New training runs use **E2E + PPO + fixed Mol2Vec + a fixed fragment table +
actor selection**, with QED, improvement rewards, and local logging. The policy
selects both the building block and its completed attachment candidate. Training,
evaluation, sampling, profiling, and explicit sweeps share a typed YAML
configuration and the existing RLlib ModelV2 API stack.

```shell
uv run python -m moldr.cli train --iterations 10 --num_workers 8 --num_gpus 1
uv run python -m moldr.cli train --config configs/short.yaml --dry-run
uv run python -m moldr.cli sample --checkpoint outputs/experiments/YOUR_RUN --episodes 12
```

Use the run directory printed by training for `--checkpoint`. See
[experiment configuration](docs/experiment_configuration.md) for defaults,
overrides, ablation files, sweeps, candidate values, MPNN, and restoration.
`--mode legacy` selects the original greedy candidate scorer. The old Python
factories in `moldr.config` and `moldr.legacy` retain their legacy behavior,
and old E2E registration names remain available for existing checkpoints.
The [GPU validation](docs/experiment_validation.md) and
[short comparison results](docs/experiment_results.md) record the measured
behavior and its limits.
The [five-configuration, 500-iteration comparison](docs/experiment_learning500.md)
reports training time, checkpoint evaluations, and quality/diversity limits
for training seed 0, with [plots and extracted data](docs/learning500/index.html).

### Decomposition
In the decomposition step, molecules are decomposed into subgraphs using gSpan.  
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
minsup = int(len(mols) * 0.1)
config = DefaultConfig(
    data_path=Path("zinc_jt.data"), support=minsup, lower=2, upper=7, method="jt"
)
runner = MolsMining(config)
gspan = runner.decompose(mols)
```

If you want to see the subgraphs in detail, see
[the decomposition notebook](examples/decomposition.ipynb).

### Reassembling
- Training 

Building blocks come from the decomposition step. The standard E2E environment
lets PPO select a building block and then an attachment. The original research
environment lets the agent select a building block and greedily retains the
highest-scoring candidate; this behavior is available through `--mode legacy`
and the existing Python factories. The shared CLI uses PPO.

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


## Faster, End-to-End, and Generative Extensions

The sections below describe the chemistry, E2E, and generative extensions and
their existing Python APIs. E2E is the shared CLI's standard environment.
The original and new environments share the corrected merge implementation;
the CLI uses versioned observations/models while these older API examples keep
their original contracts.

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

### 2. End-to-end reassembly (no greedy heuristic)

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
    building_blocks_smiles,            # same fixed table as before
    model_path="models/model_300dim.pkl",
    num_workers=8, num_gpus=1,
)
algo = config.build()
for _ in range(epochs):
    algo.train()
```

### 3. Generative building blocks (instead of a fixed gSpan table)

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
    building_blocks_smiles, n_blocks=1000, device="cuda", hidden_dim=256,
    batch_size=256, epochs=50,
)
# Keep rollout workers on CPU: save the CUDA-trained model and pass its spec.
proposer.save("tmp/proposer")
proposer_spec = {"dir": "tmp/proposer"}
```

Use it in either environment:

```python
# (a) generative blocks + original reassembly (ablation vs the gSpan table)
from moldr.config import get_generative_config_v1
config = get_generative_config_v1(objective, proposer_spec, model_path="models/model_300dim.pkl")

# (b) generative blocks + end-to-end reassembly (combines 2 and 3)
from moldr.e2e.config import get_e2e_config
config = get_e2e_config(objective, model_path="models/model_300dim.pkl", proposer=proposer_spec)
```

All three extensions are covered by unit and real PPO tests. CUDA tests run
when a GPU is visible. Reproduce the full GPU validation with:

```shell
uv run python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m scripts.validate_training --output tmp/gpu_validation.json
```

See [the measured GPU validation](docs/gpu_validation.md) for settings,
timings and limits. Model computation benefits from CUDA, while these short
PPO runs remain dominated by CPU molecule enumeration. Fragment proposals
select entries from the existing vocabulary; unseen fragments are not created.
Without per-fragment `contexts`, VAE training is unconditional. Specified
tetrahedral or bond stereochemistry is currently unsupported by the adjacency
merge implementation: environments reject it, and fragment vocabularies filter
it out rather than silently discarding it.

### CPU/GPU timelines and distributed rollout profiling

Capture real Chrome/Perfetto timelines and compare 1, 2, 4 and 8 CPU rollout
workers with the same global batch:

```shell
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m scripts.profile_training
```

Open `artifacts/profiling/2026-10-13/index.html` for the comparison and offline
interactive timelines; each `trace.json` also opens in Perfetto or Chrome
Tracing. The collector records CPU chemistry/scoring/embedding and CUDA kernels
on a shared single-host clock. Throughput trials run without tracing overhead.
See [the profiling report](docs/performance_profiling.md) for measured scaling,
worker balance, clock limits and reproduction settings.


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
