# mi_collections
This package includes features of Generating molecules and computing descriptors.

## Requirements
```shell
conda create -c rdkit -n my-rdkit-env rdkit==2020.09.1
```

## Mol2Vec
Mol2vec is proposed by S. Jaeger et al. Official repositroy of Mol2Vec is here: [Link](https://github.com/samoturk/mol2vec). 

How to use mol2vec:

```
import rdkit.Chem as Chem
from mi_collections import Mol2Vec

mol2vec = Mol2Vec(n_radius=2)
mols = [Chem.MolFromSmiles(s) for s in ['CC', 'CCC', 'CCO']]
vec = mol2vec.fit_transform(mols)
```

### Preprocessing by Word2Vec
When you want to train the model, you need to convert sdf, smi and etc. into a corpus file.

```
mol2vec.preprocess('data/test.sdf', 'output_corpus', threshold=3)
```

where, `threshold` means that the substructures occured within the number of threshold is regarded as a same indentifier UNK.


### Train model
Set the hyperparameters: vector_size, window, min_count to train Word2Vec model.

- save the model

```
mol2vec.train('output_corpus', 'mymodel.pkl', epochs=100)
```

- not save the model

```
model = mol2vec.train_model('output_corpus', epochs=100)
```

### Load model

```
mol2vec.load_model('mymodel.pkl')
```

- OR

```
mol2vec = Mol2Vec(model_path='mymodel.pkl')
```

## Descriptors
Descriptors are used for converting molecules into graph features for computing the state in reinforcement learning. 

- mol2vec

- Graph Kernel
  - Weisfeller-Lehman Graph Kernel (WL kernel)
  - Wasserstein WL Kernel (WWL kernel)

- ECFP

- Graph Neural Network
  - Graph Convolutinal Network (GCN)