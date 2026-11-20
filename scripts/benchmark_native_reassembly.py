"""Compare Python and C++ candidate generation on fixed-seed fragment growth.

Run from a checkout with the native extension and dev dependencies installed:
    python -m scripts.benchmark_native_reassembly
"""

import argparse
import csv
import json
import platform
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase

from moldr.chemistry import CandidateGenerator
from moldr.chemutils import get_mol
from tests.test_local_reassembly import molecular_state

BLOCKS = Path(__file__).resolve().parents[1] / "data" / "building_blocks"
CORPORA = {
    "zinc": BLOCKS / "zinc/s1000_l2.csv",
    "guacamol": BLOCKS / "guacamol/guacamol_s1000_l2.csv",
}


@dataclass
class Case:
    corpus: str
    base: Chem.Mol
    fragment: Chem.Mol
    stats: dict


def checked_cases(python, cpp, rng):
    """Compare both APIs before timing and use Python outputs for trajectory growth."""
    cases = []
    for corpus, path in CORPORA.items():
        with path.open() as f:
            rows = csv.reader(f)
            next(rows)
            fragments = [mol for row in rows if (mol := get_mol(row[1])) is not None]
        for episode in range(6):
            base = get_mol("C")
            for step in range(40):
                fragment = rng.choice(fragments)
                expected = python.generate(base, fragment)
                actual = cpp.generate(base, fragment)
                where = f"{corpus} episode {episode}, step {step}"
                if python.stats != cpp.stats:
                    raise AssertionError(f"{where}: {python.stats} != {cpp.stats}")
                if [c.smiles for c in expected] != [c.smiles for c in actual]:
                    raise AssertionError(f"{where}: ordered candidate SMILES differ")
                for left, right in zip(expected, actual, strict=True):
                    if molecular_state(left.mol) != molecular_state(right.mol):
                        raise AssertionError(
                            f"{where}: Mol state differs for {left.smiles}"
                        )
                cases.append(Case(corpus, base, fragment, python.stats.copy()))
                if not expected:
                    break
                base = rng.choice(expected).mol
                if base.GetNumAtoms() >= 65:
                    break
    return cases


def compare(seed, repeats):
    rng = random.Random(seed)
    backends = {
        "python": CandidateGenerator(),
        "cpp": CandidateGenerator(backend="cpp"),
    }
    cases = checked_cases(backends["python"], backends["cpp"], rng)
    samples = {name: [[] for _ in cases] for name in backends}
    indices = list(range(len(cases)))
    names = list(backends)
    for _ in range(repeats):
        rng.shuffle(indices)
        for i in indices:
            case = cases[i]
            rng.shuffle(names)
            for name in names:
                generate = backends[name].generate
                start = time.perf_counter()
                candidates = generate(case.base, case.fragment)
                elapsed = time.perf_counter() - start
                del candidates
                samples[name][i].append(elapsed)

    report = {
        "seed": seed,
        "repeats": repeats,
        "rdkit": rdBase.rdkitVersion,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "scope": "CandidateGenerator.generate, including Python/native transfer. "
        "One untimed validation pass, then randomized execution order. "
        "Seconds are sums of per-pair medians.",
        "ordered_smiles_mol_states_and_stats_equal": True,
    }
    for corpus in [*CORPORA, "all"]:
        selected = [
            i for i, c in enumerate(cases) if corpus == "all" or c.corpus == corpus
        ]
        seconds = {
            name: sum(statistics.median(values[i]) for i in selected)
            for name, values in samples.items()
        }
        report[corpus] = {
            "pairs": len(selected),
            "generated": sum(cases[i].stats["generated"] for i in selected),
            "valid": sum(cases[i].stats["valid"] for i in selected),
            "seconds": seconds,
            "speedup": seconds["python"] / seconds["cpp"],
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    RDLogger.DisableLog("rdApp.*")
    report_json = json.dumps(compare(args.seed, args.repeats), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n")
    print(report_json)
