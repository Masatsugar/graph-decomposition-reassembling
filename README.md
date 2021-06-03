# Molecular Graph Generation by Decomposition and Reassembling

graph-decomposition-reassembling

## Requirements



## Run

- Decomposition step
```
python decomposition.py --data 'zinc_jt_gspan_s1000_l7.csv' --target QED --rollout 100

```

- Reassembling step
```
python reassemble.py --data 'zinc_jt_gspan_s1000_l7.csv' --target QED --rollout 100
```