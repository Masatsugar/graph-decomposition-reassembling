# Reassembly performance and compatibility

The reassembly hot path now constructs RDKit molecules directly from NumPy
adjacency matrices, reuses the combined molecule within each node/edge
candidate enumeration, and builds `MolGraph.graph` only when requested.
The existing `merge_node`, `merge_edge`, and `sanitize_molgraph` entry points
remain available. `MolGraph.graph` remains writable, and pickled objects with
the original public `graph` attribute retain their saved graph when loaded.

This change is limited to chemistry and candidate construction. The existing
legacy environment still scores every candidate and chooses its highest score.
Its observations, actions, reward calculation, training configuration, and
checkpoint architecture are unchanged. End-to-end policies, VAE proposers,
and the shared experiment CLI are separate changes.

## Measured comparison

The [recorded CPU microbenchmark](reassembly_benchmark.json) compares the
unmodified main revision
[`301fd9a`](https://github.com/Masatsugar/graph-decomposition-reassembling/commit/301fd9aca3342ee449199d7666175a076ca29210)
with the optimized implementation under the same interpreter and dependencies.
Each implementation runs in its own process. The script extracts the original
source directly from Git; it does not simulate the original with the new helpers.

On twelve neutral, non-stereochemical molecule pairs, the sum of per-pair
median times fell from **0.510 seconds to 0.0693 seconds (7.36×)**. Each pair had
two warmups and seven measured repetitions. Raw node candidates, raw edge
candidates, and final candidates matched the original main exactly, including
ordering, for all twelve pairs.

The measurement includes enumeration, candidate graph construction,
parsing/serialization, and final sanitization. It excludes imports and initial
input-SMILES parsing. It measures CPU chemistry work, **not end-to-end PPO
training speed**. The small fixed fixture set does not establish a universal
speedup or chemical equivalence for every input. All timing samples, candidate
counts, dependency versions, and both implementations' source hashes are in
the JSON report. Recorded versions: Python 3.12.7, RDKit 2025.03.6, NumPy 1.26.4,
NetworkX 3.7, and SciPy 1.11.4. BLAS/OpenMP thread limits were one.

Reproduce from the repository root in an environment with the project dependencies:

```bash
python scripts/benchmark_reassembly.py --warmup 2 --repeats 7 \
  --output tmp/reassembly_benchmark.json
```

The pinned baseline commit must exist locally; fetch full Git history if using
a shallow clone. A candidate or ordering mismatch makes the script exit with
an error. Timing differences do not fail tests.

## Correctness boundaries

The optimization includes chemistry corrections needed to avoid carrying
known data-loss bugs into the fast implementation:

- Atom templates retain formal charges, isotopes, and explicit hydrogen counts.
  Final sanitization removes temporary atom maps without rebuilding plain atoms.
- Ring-edge attachment uses actual bonds. The original junction-tree clique
  ordering could select non-bonded atoms in bridged rings.
- Specified tetrahedral/bond stereochemistry is rejected explicitly because
  adjacency-based merging cannot preserve it. Existing stereochemistry is
  retained by standalone `sanitize_molgraph`.
- `merge_edge` always returns a list, including when no candidates exist.

These corrections intentionally differ from the original behavior on affected
inputs. The exact-equivalence claim above applies only to the twelve documented
neutral fixtures. Fixed charged, isotopic, hydrogen-bearing, bridged-ring, and
stereo cases have separate regression tests. A frozen reference implementation
with the chemistry corrections independently checks the additional optimizations;
it is distinct from the original-main fixtures.

## Validation

```bash
python -m pytest -q tests/test_reassemble_fast.py tests/test_reassemble_compatibility.py
```

The focused run passed **92 tests**, including original-main fixture parity,
corrected chemistry, input immutability, graph assignment and pickle compatibility,
and one real legacy-environment step using the repository's Mol2Vec model.
The [original-main fixture](../tests/fixtures/reassembly_main_301fd9a.json) records
its baseline commit, source hashes, environment, and exact candidate lists. To
regenerate it from the pinned main implementation, pass
`--write-fixtures tests/fixtures/reassembly_main_301fd9a.json` to the benchmark.

The pre-existing `tests/test_env.py` imports a missing `get_default_config` name
from `moldr.config`, so the original repository's full test collection is already
broken. That factory issue is outside this chemistry change. The focused smoke
test constructs `MolEnvValueMax` directly and does not start Ray.
