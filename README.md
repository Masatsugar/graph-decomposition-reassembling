# Molecular Graph Generation by Decomposition and Reassembling

This is the implementation of our paper. 


## Installation
Install RDKit using Anaconda environment. 

```
conda create -c rdkit -n my-rdkit-env rdkit==2020.09.1
```


## How to run
MOLDR takes two steps to generate molecular graphs: 


- Decomposition step  
SMILES are converted into the graph or junction tree dataset for gSpan algorithm.
Mined molecules are saved to the `'results/mining/name.data'`. There are applied samples in `data/results/tmp` folder.

```
python decomposition.py --data 'data/zinc/all.txt' --method raw --support 1000 
```


- Reassembling step

```
python reassemble.py --data 'zinc_jt_gspan_s1000_l7.csv' --target QED --rollout 100
```

All the generated molecules along the training process are stored in the `results/generate` folder.
Generated molecules are stored as SMILES.

