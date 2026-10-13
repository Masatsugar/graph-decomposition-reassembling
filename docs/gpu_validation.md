# GPU validation — 2026-10-13 (Asia/Tokyo)

Validated locally on an NVIDIA GeForce RTX 4060 Ti, 16 GB, driver 570.211.01,
Python 3.12.7 and CUDA runtime 12.6. Exact library versions and measurements
are in [gpu_validation.json](gpu_validation.json). The final code retains
formal charges during merges; the vocabulary includes 17 charged entries.

## What passed

- Final committed-code test run: 161 passed, no skips (49.05 seconds), including
  a real CUDA VAE regression, local/remote-worker RLlib PPO updates, NumPy block
  input, and rejection of an empty block table.
- Actual CLI: QED PPO with two CPU rollout workers and one GPU trained and
  saved a checkpoint using `--logger tensorboard`. The saved CUDA policy was
  restored through `PPO.from_checkpoint` and the actual Sampler completed three
  molecule-generation episodes.
- Real GuacaMol building-block CSV: 1,709 rows, 1,679 valid canonical fragments.
  All five measured PPO runs changed model weights with finite loss, entropy,
  KL and weights. GPU policy parameters were on `cuda:0`.
- Fixed-table CPU and CUDA PPO losses matched at rtol=1e-4, atol=1e-6 with
  identical seeds/settings. This checks these runs, not all future trajectories.
- Pointer output and gradient equivalence are covered by regression tests;
  CUDA fp16 forward/backward and masked distributions remained finite.
- A CUDA-trained VAE proposed valid blocks and reloaded on CPU with matching
  logits. PPO checkpoint restore preserved every saved weight exactly.

## Computation and memory

Pointer minibatch: 128 molecules, 1,679 option slots, embedding dimension 300,
hidden width 256, float32, three warmup passes, ten measured forward/backward
passes. CUDA timings synchronize before/after measurement. CPU uses one thread.
The original column executes the previous concatenated scorer with the same
weights and inputs; it is a comparison of the model computation.

| Pointer forward + backward | Original | Fixed |
| --- | ---: | ---: |
| CPU, milliseconds | 2043.3 | 1385.2 |
| GPU, milliseconds | 29.15 | 21.55 |
| GPU peak allocated memory, MiB | 1344.4 | 1107.6 |

The fixed GPU model is 1.35× faster and uses
17.6% less allocated memory than the original scorer.
These are PyTorch allocation peaks, not total driver memory or whole-job peaks.

VAE: 50 epochs, hidden width 256, latent dimension 16, batch size 256, beta 0.01,
seed 23, no contexts. Vocabulary construction is excluded from timing.

| VAE measurement | CPU, one thread | GPU |
| --- | ---: | ---: |
| Training seconds | 2.722 | 0.316 |
| Fragment examples/second | 30839 | 265416 |
| Reconstruction cross entropy at posterior mean | 5.924 | 5.997 |

GPU VAE training is 8.61× faster than this one-thread CPU run.
Reconstruction cross entropy decreases from 7.427 to 5.997; this is
neither a held-out score nor the full ELBO.

## Whole PPO run

Shared settings: seed 31, train batch 512, minibatch 128, three optimization
epochs, two CPU workers, length 20, threshold 1.1. E2E uses pool/attachment caps
32, hidden width 128, and time limit 40 actions. Generative runs load a CPU
proposer with 64 anchors from the saved GPU-trained VAE. CPU and CUDA E2E runs
have three training iterations each. Legacy variants have one GPU iteration.
Reported wall times exclude algorithm construction and worker startup. Ray
internal timers are smoothed averages; wall times are measured per call.

| E2E iteration, mean | CPU learner | GPU learner |
| --- | ---: | ---: |
| Wall time, seconds | 3.211 | 3.134 |
| Ray learner timer, milliseconds | 126.0 | 56.8 |
| Ray sampling timer, milliseconds | 3088.3 | 3084.5 |

CUDA accelerates learning, but whole iterations improve only
2.4% here. CPU sampling/reassembly is the dominant cost.
The final fixed-table CUDA run spends approximately
97.9% of its Ray training-step timer in sampling.
GPU utilization alone therefore cannot establish whole-job efficiency. Increasing
CPU rollout throughput and using larger useful learner batches are the next
performance experiments; neither was established by these checks.

Other final-tree CUDA paths passed: generative E2E (3.031s/iteration),
original reassembly (7.788s), and generative original reassembly
(6.614s). Their action spaces/transition budgets differ,
so these timings should not be treated as a controlled comparison of methods.

## Molecular optimization is not established

After three fixed-table E2E PPO iterations, 12 seeded evaluation episodes gave
mean final QED 0.492 (trained deterministic policy), versus
0.537 (uniform valid actions). The short trained run did not outperform
random; this sample is too small to establish convergence or a quality advantage.

The VAE's training-vocabulary reconstruction accuracy at posterior mean is only
0.596%. Its 17-dimensional descriptors collide (for example `CCCC` and
`CC(C)C`), so some classes cannot be distinguished with identical contexts.
The decoder selects existing vocabulary fragments, and this run used zero
contexts; it does not demonstrate unseen-fragment generation or trained
conditional molecular search. Longer training, richer fragment encoding,
meaningful parent-molecule contexts and held-out molecular benchmarks remain
necessary to assess those research claims.

Specified tetrahedral and bond stereochemistry is explicitly unsupported by
this adjacency-based merger. Environments reject such inputs and generated
vocabularies filter them. Charge, isotope and explicit hydrogen retention are
tested in both node and edge merges. This validation does not cover arbitrary
scoring functions, long-run convergence, multiple GPUs or other machines.

## Reproduce

Run from the repository root with a visible GPU:

```sh
uv sync --python 3.12 --locked --extra dev
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m scripts.validate_training --output tmp/gpu_validation.json
uv run python -m moldr.train --objective QED --epochs 1 --num_workers 2 \
  --num_gpus 1 --logger tensorboard --train-batch-size 128 \
  --minibatch-size 64 --num-epochs 1 --length 12
```

In this managed environment, CUDA and Ray sockets required sandbox escalation.
No tracking service was contacted, and no cloud GPU was provisioned. Local
checkpoints are under `tmp/` and `outputs/`, excluded from Git.
