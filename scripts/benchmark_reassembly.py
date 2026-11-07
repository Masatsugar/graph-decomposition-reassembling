"""Compare the chemistry hot path with the unmodified, pinned main revision.

Run from the repository root. Both implementations run in separate processes
under the same Python environment. Imports and initial input-SMILES parsing are
excluded from timing. Node/edge enumeration, candidate graph construction,
candidate parsing/serialization and final sanitization are included.
No Ray, policy inference, reward calculation or training is measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE_COMMIT = "301fd9aca3342ee449199d7666175a076ca29210"
SOURCE_PATHS = (
    "moldr/__init__.py",
    "moldr/chemutils.py",
    "moldr/core/__init__.py",
    "moldr/core/reassemble.py",
    "moldr/core/molgraph.py",
    "moldr/core/moltree.py",
)
PAIRS = [
    ["C", "CCO"],
    ["CCO", "C#N"],
    ["CC(=O)O", "CO"],
    ["c1ccccc1", "C1CCCCC1"],
    ["c1ccncc1", "C1CCOCC1"],
    ["N1CCCCC1", "C1CCOCC1"],
    ["CCO", "c1ccccc1"],
    ["CC1CCC2=CC=CC=C2O1", "c1ccccc1"],
    ["O=C(O)c1ccccc1", "CO"],
    ["CN1CCN(C)CC1", "CC"],
    ["C1=CCc2ccccc2C1", "C1CCCCC1"],
    ["CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "CCO"],
]
WORKER = r'''
import contextlib
import json
import platform
import sys
import time
import networkx
import numpy
import rdkit
import scipy
from rdkit import RDLogger
from moldr.chemutils import get_mol, get_smiles
from moldr.core.molgraph import sanitize_molgraph
from moldr.core.reassemble import merge_node, merge_edge

RDLogger.DisableLog("rdApp.*")
settings = json.load(sys.stdin)

def pipeline(base, fragment):
    nodes = merge_node(base, fragment)
    edges = merge_edge(base, fragment)
    final = sanitize_molgraph(nodes + edges)
    return {
        "node": [candidate.smiles for candidate in nodes],
        "edge": [candidate.smiles for candidate in edges],
        "final": [get_smiles(mol) for mol in final],
    }

cases = []
with contextlib.redirect_stdout(sys.stderr):
    for base, fragment in settings["pairs"]:
        molecules = get_mol(base), get_mol(fragment)
        result = pipeline(*molecules)
        for _ in range(settings["warmup"]):
            pipeline(*molecules)
        samples = []
        for _ in range(settings["repeats"]):
            start = time.perf_counter()
            repeated = pipeline(*molecules)
            samples.append(time.perf_counter() - start)
            if repeated != result:
                raise AssertionError("Candidate order changed between repetitions")
        cases.append({"base": base, "fragment": fragment,
                      "candidates": result, "seconds": samples})
print(json.dumps({"environment": {
    "python": platform.python_version(), "platform": platform.platform(),
    "rdkit": rdkit.__version__, "numpy": numpy.__version__,
    "networkx": networkx.__version__, "scipy": scipy.__version__,
}, "cases": cases}))
'''


def run_worker(directory, settings):
    environment = dict(os.environ, PYTHONPATH=str(directory))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    result = subprocess.run(
        [sys.executable, "-c", WORKER], cwd=directory, env=environment,
        input=json.dumps(settings), text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def digest_sources(directory):
    return {
        relative: hashlib.sha256((directory / relative).read_bytes()).hexdigest()
        for relative in SOURCE_PATHS
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("tmp/reassembly_benchmark.json"))
    parser.add_argument("--write-fixtures", type=Path,
                        help="Write expected candidates generated only by the pinned main")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    settings = {"pairs": PAIRS, "repeats": args.repeats, "warmup": args.warmup}
    with tempfile.TemporaryDirectory(prefix="moldr-original-main-") as temporary:
        baseline_root = Path(temporary)
        for relative in SOURCE_PATHS:
            result = subprocess.run(
                ["git", "show", f"{BASELINE_COMMIT}:{relative}"], cwd=root,
                capture_output=True, check=True,
            )
            path = baseline_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(result.stdout)
        baseline_hashes = digest_sources(baseline_root)
        baseline = run_worker(baseline_root, settings)
    optimized = run_worker(root, settings)
    if baseline["environment"] != optimized["environment"]:
        raise AssertionError("Baseline and optimized dependency versions differ")
    comparisons = []
    for before, after in zip(baseline["cases"], optimized["cases"], strict=True):
        before_median = statistics.median(before["seconds"])
        after_median = statistics.median(after["seconds"])
        comparisons.append({
            "base": before["base"], "fragment": before["fragment"],
            "exact_candidate_order": before["candidates"] == after["candidates"],
            "candidate_counts": {key: len(value) for key, value in before["candidates"].items()},
            "baseline_seconds": before["seconds"], "optimized_seconds": after["seconds"],
            "baseline_median_seconds": before_median,
            "optimized_median_seconds": after_median,
            "speedup": before_median / after_median,
        })
    baseline_total = sum(case["baseline_median_seconds"] for case in comparisons)
    optimized_total = sum(case["optimized_median_seconds"] for case in comparisons)
    report = {
        "baseline_commit": BASELINE_COMMIT,
        "baseline_source_sha256": baseline_hashes,
        "optimized_source_sha256": digest_sources(root),
        "environment": baseline["environment"],
        "scope": "CPU node/edge enumeration, candidate serialization, and sanitization; no PPO or scoring",
        "settings": {"repeats": args.repeats, "warmup": args.warmup, "threads": 1},
        "all_candidates_and_order_equal": all(case["exact_candidate_order"] for case in comparisons),
        "baseline_sum_of_medians_seconds": baseline_total,
        "optimized_sum_of_medians_seconds": optimized_total,
        "speedup_from_sum_of_medians": baseline_total / optimized_total,
        "cases": comparisons,
    }
    write_json(args.output, report)
    if args.write_fixtures:
        write_json(args.write_fixtures, {
            "baseline_commit": BASELINE_COMMIT,
            "source_sha256": baseline_hashes,
            "environment": baseline["environment"],
            "cases": [{key: value for key, value in case.items() if key != "seconds"}
                      for case in baseline["cases"]],
        })
    print(json.dumps({key: report[key] for key in (
        "all_candidates_and_order_equal", "baseline_sum_of_medians_seconds",
        "optimized_sum_of_medians_seconds", "speedup_from_sum_of_medians",
    )}, indent=2))
    print(f"Report: {args.output}")
    if not report["all_candidates_and_order_equal"]:
        raise SystemExit("Candidate mismatch; inspect results before claiming a transparent speedup")


if __name__ == "__main__":
    main()
