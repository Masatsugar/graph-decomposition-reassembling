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

If you want to see the subgraphs in detail, see `examples/decomponsition.ipynb`.

### Reassembling
- Training 

The agent takes an action from building blocks obtained from a decomposition step. 
In default, PPO with RLlib is used for training the agent, but you can also use other algorithms such as DQN, A2C, or IMPALA by changing the `algo` parameter in `train.py`.
For detailes, see rllib [documentation](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html).

We already mined the building blocks from gucamol or zinc dataset. The default is guacamol with minsup 10,000.
Set your own dataset to generate building blocks if you want to use other datasets.

```shell
uv run python moldr/train.py --epochs 10 --num_workers 8 --num_gpus 1
```

For a short run without an external experiment tracker:

```shell
uv run python -m moldr.train --objective QED --epochs 1 --num_workers 2 \
  --num_gpus 1 --logger tensorboard --train-batch-size 128 \
  --minibatch-size 64 --num-epochs 1 --length 12
```

Generated molecules are sampled through the trained agent. Select and rewrite the model path that you want to use.

```shell
uv run python run_moldr.py 
```

If you want to use the custom score function to optimize the generated molecules, 
edit `moldr/objective.py` directly according to guacamol API, and add it into `sc_list` in `train.py`.


## Faster, End-to-End, and Generative Extensions

Three optional extensions improve the speed and expressiveness of the method.
The original and new environments share the corrected merge implementation.

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
