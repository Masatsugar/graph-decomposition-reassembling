# Experiment results — 2026-10-14 (Asia/Tokyo)

All **54 of 54 short speed-comparison jobs completed successfully**: 15 major
configuration jobs, 30 additional ablation jobs, and nine worker-scaling jobs.
Every configuration has seeds 0, 1, and 2. All five separate GPU trace cases,
all fifteen post-training checkpoint evaluations, and the CPU actor/lookahead
comparison have also completed successfully.
Recorded trials are available in
the [local comparison report](../artifacts/experiments_short/sweep_2026-10-13_23-28-09-727175/index.html)
and [sweep state](../artifacts/experiments_short/sweep_2026-10-13_23-28-09-727175/sweep.json).
The separate [real GPU validation](experiment_validation.md) verifies learning,
gradients, compatibility, and restoration mechanics.

Common conditions are batch 1024, minibatch 32, three PPO update epochs, policy
hidden width 256, atom limit 20, attachment cap 128, QED threshold 1.1, eight
rollout workers, and one learner GPU. Every one of the 1709 fragments is
presented. Three warmup iterations precede ten measured iterations. MPNN uses
three layers with hidden/output width 128. The frozen MPNN comparison disables
persistent embedding caching, matching the explicit encoder comparison settings.
Speed measurements have no tracing overhead.

The machine has an Intel Core i9-14900 CPU with 32 logical CPUs and CPU affinity
covering all 32, plus an NVIDIA RTX 4060 Ti with 16 GiB VRAM. Torch, OpenMP,
OpenBLAS, and MKL each use one thread. Ray uses the configured rollout-worker
count; the driver's default CPU resource allocation is
`max(2, num_workers + 2)`. These CPU and thread settings matter for interpreting
the sampling times as well as reproducing the learner-GPU comparison.

Mol2Vec uses the existing pretrained model asset. Each MPNN encoder starts from
its training seed's random initialization. The frozen MPNN case freezes that
initial encoder without loading a pretrained GNN; the trainable case updates it
with PPO. Their short-run scores therefore compare different initialization and
training histories, rather than matched pretrained encoders.

Values below are means ± sample standard deviations across completed training
seeds. A single-seed result has no estimated deviation.

| Configuration | Seeds complete/planned | Valid selected updates/s | Valid final episode outputs/s | Mean final training QED | Peak allocated CUDA MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mol2Vec actor | 3/3 | 66.91 ± 1.14 | 21.08 ± 0.46 | 0.7395 ± 0.0062 | 8255.2 ± 0.3 |
| Mol2Vec actor-value | 3/3 | 58.38 ± 0.12 | 18.99 ± 0.11 | 0.7373 ± 0.0056 | 8257.0 ± 0.3 |
| Mol2Vec auxiliary-only | 3/3 | 61.46 ± 0.51 | 19.98 ± 0.31 | 0.7360 ± 0.0026 | 8257.2 ± 0.3 |
| Frozen MPNN actor | 3/3 | 13.04 ± 0.02 | 4.24 ± 0.05 | 0.6206 ± 0.0012 | 8498.7 ± 0.9 |
| Trainable MPNN actor | 3/3 | 12.69 ± 0.03 | 4.10 ± 0.14 | 0.6266 ± 0.0044 | 9197.9 ± 11.7 |

The actor-value and auxiliary-only cases add learner work in this short test.
Mean RLlib smoothed learning time rises from 0.96 seconds for the actor to
1.86 seconds for actor-value and 1.67 seconds for auxiliary-only. Their training
QED means are close to the baseline at this budget; these values do not establish
long-run learning quality or an independent candidate-value advantage.

Frozen MPNN is slower in the complete fixed-catalog setting measured here.
Its smoothed sampling/learning times average 26.18/12.68 seconds, compared with
6.50/0.96 seconds for Mol2Vec. The jointly trained encoder averages
26.24/13.71 seconds and uses about 699 MiB more peak allocated CUDA memory than
the frozen encoder. Its final training QED mean is 0.6266 versus 0.6206 for the
frozen encoder at this short budget. These timings are RLlib rolling values,
shown separately; they are not exclusive profiler
durations or a complexity claim about graph encoding.

The ten remaining eight-worker ablations also have all three seeds completed.
Each row changes the named setting from the Mol2Vec actor baseline, except
direct Mol transfer, which also requires mapless candidates. Score computations
are counts across the ten measured iterations, averaged over seeds.

| Ablation | Valid selected updates/s | Valid final episode outputs/s | Mean final training QED | Score computations |
| --- | ---: | ---: | ---: | ---: |
| String atom maps | 65.68 ± 0.15 | 20.69 ± 0.15 | 0.7395 ± 0.0062 | 6726 ± 10 |
| Unshared pair preparation | 65.26 ± 0.58 | 20.55 ± 0.25 | 0.7395 ± 0.0062 | 6726 ± 10 |
| Python bond scan | 67.17 ± 0.66 | 21.16 ± 0.14 | 0.7395 ± 0.0062 | 6726 ± 10 |
| Mapless candidates | 66.72 ± 0.39 | 21.00 ± 0.13 | 0.7401 ± 0.0036 | 6724 ± 15 |
| Aromatic deduplication | 66.55 ± 0.47 | 21.04 ± 0.22 | 0.7421 ± 0.0050 | 6731 ± 16 |
| Direct Mol transfer + mapless | 67.38 ± 0.31 | 21.21 ± 0.37 | 0.7401 ± 0.0054 | 6725 ± 27 |
| Mol-input scoring | 67.08 ± 0.44 | 21.13 ± 0.11 | 0.7395 ± 0.0062 | 6726 ± 10 |
| Score cache | 67.31 ± 0.19 | 21.20 ± 0.12 | 0.7395 ± 0.0062 | 4850 ± 7 |
| No embedding reuse | 64.74 ± 0.41 | 20.39 ± 0.19 | 0.7395 ± 0.0062 | 6726 ± 10 |
| Classic context | 69.49 ± 0.24 | 21.98 ± 0.36 | 0.7255 ± 0.0047 | 6732 ± 35 |

The chemistry and Mol-input scoring switches change end-to-end throughput by
less than 2.5% relative to the baseline mean in this short comparison. The Python
scan is slightly faster here, so these totals do not demonstrate a speed gain
from nonzero scanning in the full training workload. Mapless representation,
aromatic deduplication, and direct transfer can change candidate ordering and
trajectories; their score differences are not isolated scoring-speed effects.

The score cache serves 1877 ± 14 hits out of 6726 ± 10 requests, reducing actual
QED computations by 27.9%. Its selected-update rate is only 0.6% higher than the
baseline mean. Disabling embedding reuse lowers that rate by 3.3%. Classic
context is 3.8% faster but has a lower training QED mean by 0.0140 at this budget.
Aggregate training QED and score-request counts match the baseline for string
maps, unshared preparation, Python scan, Mol scoring, caching, and no embedding
reuse. These end-to-end observations need the separate trace breakdown for cost
attribution and do not establish long-run quality.

The baseline worker comparison retains batch 1024 and all other common
settings. Sampling and learning columns below show mean RLlib smoothed times,
not exclusive profiler durations; the other metrics show three-seed mean ±
sample standard deviation.

| Rollout workers | Valid selected updates/s | Valid final episode outputs/s | Mean final training QED | Smoothed sampling seconds | Smoothed learning seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24.07 ± 0.17 | 7.58 ± 0.12 | 0.7516 ± 0.0048 | 19.54 | 0.93 |
| 2 | 38.23 ± 0.45 | 12.09 ± 0.12 | 0.7451 ± 0.0061 | 11.86 | 0.95 |
| 4 | 54.00 ± 0.53 | 17.24 ± 0.19 | 0.7438 ± 0.0052 | 8.00 | 0.96 |
| 8 | 66.91 ± 1.14 | 21.08 ± 0.46 | 0.7395 ± 0.0062 | 6.50 | 0.96 |

Eight workers produce 2.78 times the selected-update rate of one worker, with
sublinear scaling. Smoothed sampling time falls from 19.54 to 6.50 seconds while
learning stays near one second, consistent with sampling dominating this
Mol2Vec workload. Peak allocated CUDA memory stays near 8255 MiB across worker
counts. Worker count also changes rollout scheduling and stochastic trajectories;
the QED differences are not evidence of a quality gain from CPU parallelism.

The retained `valid_molecules` counter counts validated selected intermediate
updates and therefore matches update count in these environments. Final output
rate counts completed episodes, including terminal, truncated, and dead-end
endings, divided by measured wall time. Every final base passes the environment's
chemistry validation gate. Counts include repeated molecules. The report's
rolling sampled uniqueness uses bounded histories that include warmup; the
following checkpoint evaluation measures fingerprint diversity separately.

The separate trace sweep uses batch **128**, minibatch 32, three PPO epochs,
two workers, and the full 1709-fragment catalog. One warmup and one untraced
measured iteration precede one traced iteration. The smaller batch reduces
nominal minibatch updates from 96 to 12 and limits exported Torch event volume;
the speed sweep above retains batch **1024**. Trace iteration duration and scope
fractions must be interpreted within the trace configuration, rather than
compared directly with the eight-worker, batch-1024 speed measurements.

All five trace jobs completed with **zero dropped events**. The following table
uses fresh measured driver spans and the union of traced CUDA activity; GPU
percent is a fraction of that iteration, not whole-device utilization. Each row
is one seed-0 traced iteration with the smaller trace configuration.

| Trace configuration | Iteration seconds | Driver sampling/wait seconds | Learner seconds | Traced GPU busy seconds | Traced GPU busy fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Mol2Vec actor](../artifacts/experiments_trace/sweep_2026-10-14_02-07-52-856936/trace_mol2vec_actor_seed0_workers2/timeline.html) | 3.867 | 3.684 | 0.175 | 0.141 | 3.65% |
| [Mol2Vec actor-value](../artifacts/experiments_trace/sweep_2026-10-14_02-07-52-856936/trace_actor_value_seed0_workers2/timeline.html) | 4.062 | 3.768 | 0.285 | 0.251 | 6.19% |
| [Mol2Vec auxiliary-only](../artifacts/experiments_trace/sweep_2026-10-14_02-07-52-856936/trace_auxiliary_only_seed0_workers2/timeline.html) | 3.929 | 3.654 | 0.266 | 0.231 | 5.87% |
| [Frozen MPNN actor](../artifacts/experiments_trace/sweep_2026-10-14_02-07-52-856936/trace_mpnn_frozen_seed0_workers2/timeline.html) | 13.186 | 10.431 | 2.613 | 0.513 | 3.89% |
| [Trainable MPNN actor](../artifacts/experiments_trace/sweep_2026-10-14_02-07-52-856936/trace_mpnn_trainable_seed0_workers2/timeline.html) | 13.433 | 10.325 | 3.097 | 0.731 | 5.44% |

The worker breakdown exposes SMILES processing, candidate-value computation,
and graph encoding. Values below are **mean exclusive milliseconds per traced
worker fragment** across the two concurrent CPU rollout workers. They must not
be summed across workers and treated as iteration wall time. A dash means no
worker span for that category; auxiliary-only U training occurs in the learner.

| Trace configuration | SMILES processing ms | Candidate U ms | GNN ms |
| --- | ---: | ---: | ---: |
| Mol2Vec actor | 168.3 | — | — |
| Mol2Vec actor-value | 172.1 | 109.6 | — |
| Mol2Vec auxiliary-only | 174.2 | — | — |
| Frozen MPNN actor | 226.4 | — | 6920.3 |
| Trainable MPNN actor | 230.9 | — | 6798.4 |

Graph encoding dominates the measured MPNN worker fragments in this full-table
setting. The driver also has a 2.59–2.68-second gap after the final traced worker
fragment in these captures. That gap can include worker postprocessing,
compression, Ray serialization/transport, and driver aggregation; it is not a
direct network-time measurement. Configuration names in the trace table link
their HTML timelines; each run directory also retains merged Chrome/Perfetto
JSON and `trace_analysis.json`, including the
[actor breakdown](../artifacts/experiments_trace/sweep_2026-10-14_02-07-52-856936/trace_mol2vec_actor_seed0_workers2/trace_analysis.json).

The two GNN timelines use compact display: only raw `cpu_op`, `cuda_runtime`,
and `cuda_driver` duration spans are omitted. Logical scopes, all device events,
counters, and the original time envelope remain. Frozen/trainable GNN HTML
sizes decrease from 598.7/691.4 MiB to 220.5/243.7 MiB; the original HTML remains
as `timeline_full.html`. Each compact page shows omitted counts and links the
complete JSON. SHA256 checks confirm that every source JSON and analysis is
unchanged, and GPU busy time exactly matches the original full viewer. This is
a display filter, separate from the recorder's zero dropped events. The
[display audit](../artifacts/experiments_trace/sweep_2026-10-14_02-07-52-856936/compact_delivery.json)
records retained scopes, counts, sizes, and before/after hashes for all five jobs.

Matched post-training sampling restored the final checkpoints of the five major
configurations at all three training seeds. Every checkpoint used evaluation
seeds 0/1/2, `explore=true`, zero rollout workers, and one GPU. No further learning
occurred. This is **three episodes per checkpoint, fifteen checkpoints, 45 final
molecules**. The table averages each checkpoint's three-molecule metrics over
the three training seeds; deviations are sample SD across those training seeds.

| Configuration | Mean evaluation QED | Morgan/Tanimoto diversity |
| --- | ---: | ---: |
| Mol2Vec actor | 0.8283 ± 0.0311 | 0.8604 ± 0.0108 |
| Mol2Vec actor-value | 0.8131 ± 0.0428 | 0.8589 ± 0.0225 |
| Mol2Vec auxiliary-only | 0.7765 ± 0.0521 | 0.8768 ± 0.0191 |
| Frozen MPNN actor | 0.5803 ± 0.0904 | 0.8672 ± 0.0173 |
| Trainable MPNN actor | 0.6800 ± 0.1182 | 0.8781 ± 0.0080 |

All 45 final molecules were valid. Each checkpoint's three molecules were
distinct, giving per-checkpoint uniqueness 1.0; this does not assert uniqueness
across the nine molecules from different training seeds. Diversity is one minus
mean pairwise Tanimoto similarity of radius-2, 2048-bit Morgan fingerprints,
computed within each checkpoint's three episodes. The small budget provides
three molecule pairs per checkpoint and does not establish a quality or diversity
advantage. Evaluation uses final policies, whereas the training table averages
completed episode scores over the measured learning iterations. The
[post-training JSON](../artifacts/experiments_posthoc/comparison.json) and
[PNG](../artifacts/experiments_posthoc/quality_diversity.png)/[SVG](../artifacts/experiments_posthoc/quality_diversity.svg)
retain the metrics, checkpoint identities, and training-seed deviations.

The selector comparison uses identical weights from the full-catalog short-sweep
Mol2Vec actor seed-0 checkpoint at iteration 13. It runs on CPU with one Torch
thread, no rollout workers, and `explore=false`. Starting seeds 0/1/2 each request
three episodes, giving nine episodes per selector. Actual episode seeds are
`starting_seed + episode_index`, so the ranges overlap.

| CPU selector | Mean QED | Valid final molecules/s | Rollout seconds, nine episodes | Score requests/computations, nine episodes |
| --- | ---: | ---: | ---: | ---: |
| Actor | 0.7412 | 13.31 | 0.6762 | 36 / 36 |
| Value lookahead | 0.9060 | 3.54 | 2.5456 | 1008 / 1008 |

Lookahead changes the final molecule and raises QED on this particular trajectory,
while rollout time is 3.76 times greater and score requests are 28 times greater.
All 27 lookahead attachment transitions verified their actual **K+1** request
delta: the trajectory presents K=1, 30, and 77 candidates. Totals are 972 preview
requests, 27 normal selected-attachment requests, and nine reset requests. Caching
was disabled, so every request computes QED. Rollout timing includes reset,
selection, chemical transitions, and scoring, while model/Ray startup and output
are recorded separately: total full-run wall times were 17.24 seconds for actor
and 17.09 seconds for lookahead across the three runs each. Startup variation
therefore obscures the much shorter episode work in end-to-end process time.

All final molecules were valid. Each deterministic selector returned one repeated
molecule across its nine episodes: aggregate uniqueness is 1/9 and fingerprint
diversity is zero. These repetitions demonstrate the selected trajectory and cost,
not independent seed evidence of a general lookahead quality advantage. The
[comparison JSON](../artifacts/experiments_evaluation/comparison.json) and
[PNG](../artifacts/experiments_evaluation/comparison.png)/[SVG](../artifacts/experiments_evaluation/comparison.svg)
include episode-level counters, elapsed times, repeated outputs, and checkpoint
provenance. Every evaluation run also saves its resolved configuration, manifest,
asset snapshots, and checkpoint file hashes.

The 3600-second-per-job longwall sweep remains an explicit future experiment
specification. This delivery measures implementation, throughput, and initial
quality at the stated short budgets; it does not assess long-run convergence.

Provenance note: while this sweep was running, checkpoint validation, handling
of undefined explained variance in JSON, legacy hidden width, custom-objective
serialization, Matplotlib dependency/version records, and final output-path
printing were strengthened. The existing driver retains its imported modules;
later workers read the newer files. Manifest `source_sha256` records disk contents
when saved and is not a complete snapshot of already loaded modules. The
built-in QED, new E2E, Mol2Vec/GNN, chemistry, and default training computation
branches used by this sweep were unchanged. Reproduction should use the final
committed implementation and each resolved configuration, with this provenance
limit kept explicit. Recorded revisions also differ: earlier manifests use
`4146926`, while runs saved after the VAE provenance fix use `241447a`. Both may
include other uncommitted implementation files identified by their source hashes;
the jobs do not all record the same commit revision.
The later trace captures record core revision `6bee2a0`; their compact viewer
support was added in `47ac124`. Both post-training sampling and the CPU selector
comparison record evaluation revision `47ac124`. Those evaluations did not
modify the trained checkpoint weights.
