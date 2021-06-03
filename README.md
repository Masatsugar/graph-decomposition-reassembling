# Molecular Graph Generation by Decomposition and Reassembling

This is the implementation of our paper. 


## Installation
Install RDKit using Anaconda environment. 

```
conda create -c rdkit -n my-rdkit-env rdkit==2020.09.1
```
After creating the environment, please install:
```
pip install networkx scipy tqdm
```


## How to run
MOLDR takes two steps to generate molecular graphs: 

### 1. Decomposition step  
SMILES are converted into the graph or junction tree dataset for gSpan algorithm, and stored in the folder `data/graph/gspan.data`. 
Mined molecules are saved to the `'results/mining/name.csv`. 

- preprocessing + gSpan

```
python decomposition.py \
  --data data/zinc/all.txt --method jt --minsup 1000 --length 7
```

- only applying gSpan
```
python decomposition.py \
  --data data/zinc/all.txt --method jt --minsup 1000 --length 7 \
   --is_preprocess False --gspan_data data/graph/jt_graph.data
```

The argument of method has either `raw` or `jt`. 
The raw is only graph, jt is applied a molecule to tree decomposition.
The calculated samples beforehand are stored in `data/results/tmp` folder.


### 2. Reassembling step

```
python reassemble.py --data 'zinc_jt_gspan_s1000_l7.csv' --target QED --rollout 100
```

All the generated molecules along the training process are stored in the `results/generate` folder.
Generated molecules are stored as SMILES.

