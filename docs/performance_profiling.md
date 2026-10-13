# Distributed rollout profiling — 2026-10-13 (Asia/Tokyo)

Actual Chrome/Perfetto traces now cover the driver, CPU Ray rollout workers,
RDKit node/edge merging and candidate validation, QED scoring, Mol2Vec,
policy inference, batch transfer, PPO updates and CUDA kernels. The complete
local artifact set is in `artifacts/profiling/2026-10-13/`; open `index.html`
for scaling plots and four interactive timelines. `profiles.zip` packages the
entry point and its relative links. Each `timeline.html` is self-contained.

The paper reports 40 Xeon CPU cores, 256 GB memory and a 32 GB Tesla V100
([Experiments, primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10249382/)).
The current environment has an Intel Core i9-14900 (32 logical CPUs) and
one 16 GB RTX 4060 Ti. This test varies local CPU rollout parallelism; it
neither reproduces the paper's full settings nor measures multiple machines.

## Uninstrumented scaling

Each configuration has one warmup iteration followed by three measured
iterations, profiler disabled. Global train batch is exactly 1,024 environment
steps, minibatch 128, three optimization epochs, one CUDA learner, and one
single-threaded CPU environment per worker. Fragment lengths are
1,024 / 512 / 256 / 128 for 1 / 2 / 4 / 8 workers, with truncated episodes.
Every measured iteration asserts sampled and trained step counts equal 1,024.
CUDA is synchronized around each wall-time measurement.

| Mode | Workers | Mean ± SD, seconds | Mean steps/s | Speedup vs 1 worker |
| --- | ---: | ---: | ---: | ---: |
| legacy | 1 | 30.896 ± 0.275 | 33.1 | 1.00× |
| legacy | 2 | 15.798 ± 0.074 | 64.8 | 1.96× |
| legacy | 4 | 9.209 ± 0.323 | 111.3 | 3.35× |
| legacy | 8 | 5.633 ± 0.207 | 182.0 | 5.49× |
| e2e | 1 | 11.689 ± 0.159 | 87.6 | 1.00× |
| e2e | 2 | 6.333 ± 0.202 | 161.9 | 1.85× |
| e2e | 4 | 3.585 ± 0.297 | 287.5 | 3.26× |
| e2e | 8 | 2.330 ± 0.038 | 439.7 | 5.02× |

The original reassembly path scales from 30.896 to 5.633 seconds (5.49×),
and end-to-end from 11.689 to 2.330 seconds (5.02×). This supports the value of
parallel CPU molecular generation in this environment. It does not establish
large-cluster scaling or a benefit from adding learner GPUs.

Worker counts change sampled trajectories even with a fixed seed. Both modes
use the same environment-action batch size but have different actions, neural
models and transition costs: the legacy path merges and scores candidates in
one action, whereas E2E uses two actions and a sampled building-block pool.
Compare worker scaling **within a mode**, not the absolute mode timings as a
controlled molecular-generation quality or method-speed benchmark.

## Actual profiled iterations

Profiles are captured separately after a warmup with the same workload
settings; their costs and trajectories differ from the three uninstrumented
trials. Full traces include PyTorch CPU operators and CUDA kernel/copy events.
Nested logical spans also report exclusive `self_us`, so CPU category totals
avoid adding both a parent and its children. Summed worker CPU wall work can
exceed one iteration's elapsed time because workers run concurrently.

| Mode | Workers | Profiled iteration, s | Driver sampling/wait | Optimization, ms | GPU kernel/copy union, ms | Worker completion spread, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 2 | 16.018 | 99.14% | 120.1 | 16.23 | 940.5 |
| legacy | 8 | 5.817 | 97.66% | 115.8 | 16.26 | 1422.9 |
| e2e | 2 | 6.262 | 97.98% | 113.8 | 20.69 | 61.3 |
| e2e | 8 | 2.403 | 94.12% | 119.5 | 20.87 | 435.1 |

At eight workers, original reassembly's recorded sampler time breaks down
approximately into 49% chemistry and 40% QED scoring, with 4% embedding.
E2E is about 62% chemistry and 23% embedding; scoring is about 3%. These are
shares of summed exclusive **worker** scope time, not driver wall-time shares.

The original eight-worker fragments finish over a 1.423-second range, versus
0.435 seconds for E2E. Synchronous PPO waits for the latest result, so differing
fragment costs leave some workers idle before learning starts. The tail from
the last traced fragment to the driver sampling return is only about 8 ms
(original) or 25 ms (E2E) here. That tail includes worker postprocessing,
serialization, transfer and aggregation; it does not isolate network latency.

GPU kernels and copies occupy short intervals after the CPU sampling barrier.
The union excludes overlapping kernels/copies and excludes GPU annotation
ranges; it is activity inside this trace, **not** hardware utilization or SM
occupancy. In the eight-worker traces it is about 16 ms (original) and 21 ms
(E2E), while optimization includes host-side launches and PPO bookkeeping too.

## Trace integrity and how to read it

All four final traces have the expected 2/8 worker fragment scopes, exactly
1,024 traced sampler steps, visible CUDA kernels, finite learner statistics,
and zero dropped events. The actual Ray actor dispatcher caches methods, so
simply patching `RolloutWorker.sample` misses that outer call. The recorder
instead instruments the actual `SyncSampler.next` invoked inside it. This
covers each sampling fragment and excludes later batch compression and Ray
serialization. Its scope is labelled explicitly throughout the analyzer/UI.

Driver `sync_weights` is a publish/dispatch interval; Ray applies weights
asynchronously. `Policy.set_weights` on each worker records the real apply
scope without forcing a new blocking barrier.

Kineto timestamps are converted using its `baseTimeNanoseconds`. Driver and
workers use a host-wall-clock-anchored monotonic clock, and CPU/GPU lanes and
flow IDs remain distinct. This clock alignment is valid on this host. A
multi-node extension requires separately measured clock offsets.

Open an offline timeline, drag to pan, use the zoom controls or mouse wheel,
and filter worker/category. Turn on details for chemistry and PyTorch operators;
select an event for its duration, exclusive time and arguments. To inspect the
full tracing ecosystem, load a `trace.json` into
[Perfetto](https://ui.perfetto.dev/) or `chrome://tracing`; these are the
[Ray-supported trace viewers](https://docs.ray.io/en/latest/ray-observability/getting-started.html#task-timeline).

PyTorch profiling follows the
[official CPU/CUDA Chrome trace export](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality).
Shape, memory and Python-stack tracing are disabled to keep overhead lower.
Profiler overhead is excluded from the scaling comparison; the displayed
profile/unprofiled ratio includes trajectory differences and is not a pure
measurement of profiler overhead.

## Reproduce

```sh
uv sync --python 3.12 --locked --extra dev
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m scripts.profile_training
```

For one smaller trace:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m scripts.profile_training --modes legacy --workers 2 \
  --trace-workers 2 --train-batch-size 512 --output-dir artifacts/profiling/small
```

The collector generates JSON/HTML/SVG, `scaling.json` and `index.html`. To
rebuild the overview entry point from an existing result:

```sh
uv run python -c "from scripts.profile_viewer import build_index; build_index('artifacts/profiling/small/scaling.json', 'artifacts/profiling/small/index.html')"
```

Exact settings and all timing samples are preserved in
[profiling_scaling.json](profiling_scaling.json). Large trace artifacts are
local outputs excluded from Git; the collector, analyzer, renderer and their
regressions are repository code. No cloud cluster is provisioned.

## Validation

24 profiling regressions passed: nested/exclusive timing, bounded recording,
Ray cached-method dispatch, clock conversion, separate GPU lanes/flow IDs,
fragment accounting, overlapping GPU activity, annotation exclusion and
portable artifact links. The existing GPU/Ray training suite also passed.
The offline index and an eight-worker timeline were opened in headless Firefox
for visual inspection. Official Perfetto trace_processor loaded all four final
JSON files with zero parser errors and 8,288–8,290 CPU/GPU flow links each.
Flow IDs use collision-free integers compatible with its JSON importer.
The ZIP archive passed a CRC check.
