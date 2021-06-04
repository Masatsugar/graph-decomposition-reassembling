# Molecular Graph Generation by Decomposition and Reassembling

This is the implementation of our paper. 


## Installation
Install RDKit using Anaconda or Miniconda environment. 

```
conda create -c rdkit -n my-rdkit-env rdkit==2020.09.1
```
After creating the environment, please install below:
```
pip install networkx scipy tqdm
```

## How to run
MOLDR takes two steps to generate molecular graphs: 

### 1. Decomposition step  
SMILES are firstly converted into the *raw graph* or *junction tree* dataset for gSpan graph mining algorithm, and stored in the folder `data/graph/gspan.data`. 
Mined molecules are saved into the `'results/mining/`. 
 
- preprocessing + gSpan

```
python decomposition.py --gspan --jt --preprocess \
  --data data/zinc/all.txt --minsup 1000 --length 7
```

- only applying gSpan

```
python decomposition.py --gspan --jt \
 --data data/zinc/all.txt --jt --minsup 1000 --length 7 \
 --gspan_data data/graph/jt_graph.data
```

If `--jt` argument is set, tree decomposition is applied to molecules.

The results of decomposition to molecules are already stored in `data/results/` folder beforehand.


### 2. Reassembling step

- PlopP

```
python reassemble.py --data data/results/zinc_jt_gspan_s1000_l7.csv --prop PlogP \
 --rollout 1000 --max_wt 1200
```

- QED

```
python reassemble.py --data data/results/zinc_jt_gspan_s1000_l7.csv --prop QED \
 --rollout 10000 --max_wt 350
```


All the generated molecules along the training process are stored in the `results/generate` folder.
Generated molecules are stored as SMILES.


### Visualization examples
You can check each components of our method by jupyter notebooks. See notebooks folder.
Generated molecules optimized with penalized log P and QED are stored in the `data/results/paper`.
Note that if the exploration steps increase, the better molecules can be found.

- Additional Installation

```
pip install jupyter matplotlib
``` 