# Molecular Graph Generation by Decomposition and Reassembling
This repository is for [MOLDR: Molecular Graph Generation by Decomposition and Reassembling](https://pubs.acs.org/doi/10.1021/acsomega.3c01078). 

## Installation

### For Latest Version
We assume Python 3.10 or later.

```shell
pip install uv
git clone https://github.com/Masatsugar/graph-decomposition-reassembling.git
cd graph-decomposition-reassembling
uv sync
uv pip install -e .
```

After installing it, please modify `.venv/lib/<your-python-version>/site-packages/guacamol/utils/chemistry.py` file in the guacamol library.

```diff
-from scipy import histogram
+from numpy import histogram
```

This is a temporary fix for compatibility with the guacamol library. 
Related to the [issue#33](https://github.com/BenevolentAI/guacamol/issues/33) in `BenevolentAI/guacamol` repository.

In addition, remove either `rdkit` or `rdkit-pypi` due to the dependency problem. We test with `rdkit-pypi==2022.9.5`.

```shell
uv pip uninstall rdkit
```


## Usage
Set python path: `export PYTHONPATH=.`.

### Decomposition
In the decomposition step, molecules are decomposed into subgraphs using gSpan.  
Choose the "raw" decomposition method for greater versatility, or "jt" (junction tree) to take cliques into account.


```python
from pathlib import Path
from moldr.decompose import MolsMining, DefaultConfig
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

Generated molecules are sampled through the trained agent. Select and rewrite the model path that you want to use.

```shell
uv run python run_moldr.py 
```

If you want to use the custom score function to optimize the generated molecules, 
edit `moldr/objective.py` directly according to guacamol API, and add it into `sc_list` in `train.py`.


## Faster, End-to-End, and Generative Extensions

Three optional extensions improve the speed and expressiveness of the method.
They are additive: the original pipeline above is unchanged.

### 1. Faster reassembly (transparent)

The merge/reassembly step no longer round-trips molecules through networkx and
SMILES on every candidate; it uses direct numpy/RWMol operations
(`moldr.chemutils.mol_to_adj` / `mol_from_adj`). The generated molecule sets are
byte-identical to before (see `tests/test_reassemble_fast.py`), so nothing in
your workflow changes — it is simply ~2x faster per step, and `merge_edge` now
always returns a list (never `None`).

### 2. End-to-end reassembly (no greedy heuristic)

In the original `MolEnvValueMax`, the agent only chose *which* building block to
attach; the environment then enumerated every merge, scored them all, and
greedily kept `argmax(score)`. `moldr/e2e` makes that attachment choice *learned*
instead, as a two-phase MDP (select building block → select attachment) driven by
a single action-masked **pointer policy**, while keeping RLlib PPO. Only the
chosen candidate is scored, so expensive objectives are evaluated far fewer times
(~33x fewer scoring calls on a matched rollout).

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
proposer = build_proposer_from_smiles(building_blocks_smiles, n_blocks=1000)
```

Use it in either environment:

```python
# (a) generative blocks + original reassembly (ablation vs the gSpan table)
from moldr.config import get_generative_config_v1
config = get_generative_config_v1(objective, proposer, model_path="models/model_300dim.pkl")

# (b) generative blocks + end-to-end reassembly (combines 2 and 3)
from moldr.e2e.config import get_e2e_config
config = get_e2e_config(objective, model_path="models/model_300dim.pkl", proposer=proposer)
```

All three extensions are covered by unit tests in `tests/` (CPU-only; a real
RLlib PPO train step is smoke-tested for the end-to-end and generative paths).


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
