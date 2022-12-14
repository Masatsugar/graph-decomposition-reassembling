# Molecular Graph Generation by Decomposition and Reassembling

MOLDR: Molecular Graph Generation by Decomposition and Reassembling

## Installation

```shell
git clone https://github.com/Masatsugar/graph-decomposition-reassembling.git
conda install create --file env.yaml
python setup.py install mi-collections
```

## Usage
Set path `export PYTHONPATH=.`.

### Decomposition
In decomposition step, molecules are decomposed into subgraphs by gSpan. You can select raw or junction tree data.

```python
from pathlib import Path
from mi_collections.moldr.decompose import MolsMining, DefaultConfig
from mi_collections.chemutils import get_mol

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

### Training 
The agent takes an action from building blocks obtained from a decomposition step. The agent is trained by PPO with RLlib.

```shell
python train.py --epochs 100 --num_workers 128 --num_gpus 1
```

Generated molecules are sampled through the trained agent. Select and rewrite the model that you want to use.

```shell
python run_moldr.py 
```