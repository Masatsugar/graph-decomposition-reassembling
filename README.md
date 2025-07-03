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
