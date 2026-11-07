"""Reproduce P1 equality and P2 chemistry ablations without training or tracing.

Run from the repository root:
    .venv/bin/python -m scripts.compare_chemistry

The reference imports only RDKit and numpy, independent of optimized helpers.
Wall timings are CPU microbenchmarks, not PPO throughput measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from itertools import product
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger, rdBase

from moldr.chemistry import CandidateGenerator
from moldr.chemutils import get_mol
from moldr.core.reassemble import merge_edge, merge_node
from moldr.experiment import ChemistryConfig, ExperimentConfig
from moldr.scoring import ScoringService
from tests.reference_reassembly import candidates as reference_candidates

FIXTURES = [
    ("atom_zero", "C", "CCO"),
    ("multiple_bonds", "CCO", "C#N"),
    ("carbonyl", "CC(=O)O", "CO"),
    ("aromatic_saturated_ring", "c1ccccc1", "C1CCCCC1"),
    ("heteroaromatic_ring", "c1ccncc1", "C1CCOCC1"),
    ("formal_charge", "C[NH3+]", "CC"),
    ("isotope", "[13CH3]CO", "CC"),
    ("aromatic_explicit_hydrogen", "c1cc[nH]c1", "CC"),
    ("explicit_hydrogen", "[NH2]C", "CO"),
    ("bridged_rings", "C1CC2CCC1C2", "C1CCC2CC1C2"),
    ("ring_isotope", "[13CH2]1CCCCC1", "C1CCCCC1"),
    ("ring_formal_charge", "C1CC[NH2+]CC1", "C1CCCCC1"),
]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def timed(function, repeats):
    function()  # one untimed warmup, including Python/RDKit setup
    elapsed = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        elapsed.append(time.perf_counter() - start)
    return result, {"seconds": elapsed, "median_seconds": statistics.median(elapsed)}


def canonical_set(candidates):
    return {Chem.MolToSmiles(candidate.mol) for candidate in candidates}


def result_record(result, generator, baseline):
    sequence = [candidate.smiles for candidate in result]
    molecules = canonical_set(result)
    baseline_molecules = canonical_set(baseline)
    return {
        "stats": generator.stats,
        "smiles": sequence,
        "order_hash": digest(sequence),
        "molecular_set_hash": digest(sorted(molecules)),
        "same_order_as_default": sequence
        == [candidate.smiles for candidate in baseline],
        "added_molecules": sorted(molecules - baseline_molecules),
        "removed_molecules": sorted(baseline_molecules - molecules),
    }


def benchmark_scoring():
    from guacamol.standard_benchmarks import logP_benchmark, qed_benchmark

    from moldr.objective import qed_sa

    cases = []
    for name, benchmark in [
        ("QED", qed_benchmark()),
        ("logp8", logP_benchmark(8.0)),
        ("logp-1", logP_benchmark(-1.0)),
        ("QED_SA", qed_sa()),
    ]:
        for smiles in ["CCO", "C[NH3+]", "[13CH3]O", "c1cc[nH]c1", "C1CC2CCC1C2"]:
            previous = benchmark.objective.score(smiles)
            current = ScoringService(name).score(smiles)
            direct = ScoringService(name, input_mode="mol").score(
                Chem.MolFromSmiles(smiles)
            )
            if previous != current or previous != direct:
                raise AssertionError(f"Registry scoring differs for {name}: {smiles}")
            cases.append(
                {
                    "objective": name,
                    "smiles": smiles,
                    "reference": previous,
                    "smiles_score": current,
                    "mol_score": direct,
                }
            )
    return {"all_exact": True, "comparisons": cases}


def compare(repeats=3):
    RDLogger.DisableLog("rdApp.*")
    defaults = CandidateGenerator.DEFAULTS
    variants = {
        "default_p1": {},
        "string_maps": {"map_storage": "string"},
        "per_candidate_preparation": {"shared_preparation": False},
        "python_bond_scan": {"bond_scan": "python"},
        "mapless_intermediate": {"candidate_representation": "mapless"},
        "aromatic_dedup": {"dedup_key": "aromatic"},
        "direct_mol": {"candidate_representation": "mapless", "mol_transfer": "direct"},
    }
    fixtures = []
    molecular_inputs = []
    baselines = []
    oracle_results = []
    for name, base, fragment in FIXTURES:
        inputs = get_mol(base), get_mol(fragment)
        molecular_inputs.append(inputs)
        oracle, oracle_time = timed(
            lambda inputs=inputs: reference_candidates(*inputs), repeats
        )
        nodes, edges, final = oracle
        oracle_results.append(oracle)
        generator = CandidateGenerator()
        baseline = generator.generate(*inputs)
        baselines.append(baseline)
        raw_node = [candidate.smiles for candidate in merge_node(*inputs)]
        raw_edge = [candidate.smiles for candidate in merge_edge(*inputs)]
        if (
            nodes != raw_node
            or edges != raw_edge
            or final != [candidate.smiles for candidate in baseline]
        ):
            raise AssertionError(
                f"Default raw/final SMILES or order differs from oracle: {name}"
            )
        entry = {
            "name": name,
            "base": base,
            "fragment": fragment,
            "reference": {
                "raw_node_count": len(nodes),
                "raw_edge_count": len(edges),
                "raw_node_order_hash": digest(nodes),
                "raw_edge_order_hash": digest(edges),
                "final_count": len(final),
                "final_order_hash": digest(final),
                **oracle_time,
            },
            "variants": {},
        }
        for variant, differences in variants.items():
            generator = CandidateGenerator(differences)
            result, timing = timed(
                lambda gen=generator, inputs=inputs: gen.generate(*inputs), repeats
            )
            record = result_record(result, generator, baseline)
            record.update(timing)
            record["config_diff"] = differences
            record["reference_speed_ratio"] = (
                oracle_time["median_seconds"] / timing["median_seconds"]
            )
            entry["variants"][variant] = record
        fixtures.append(entry)
    matrix = []
    keys = [
        "map_storage",
        "shared_preparation",
        "bond_scan",
        "candidate_representation",
        "dedup_key",
        "mol_transfer",
    ]
    options = [
        ("int", "string"),
        (True, False),
        ("nonzero", "python"),
        ("mapped", "mapless"),
        ("kekule", "aromatic"),
        ("smiles", "direct"),
    ]
    for choices in product(*options):
        config = dict(zip(keys, choices))
        invalid = (
            config["mol_transfer"] == "direct"
            and config["candidate_representation"] != "mapless"
        )
        try:
            ExperimentConfig(chemistry=ChemistryConfig(**config)).validate()
            generator = CandidateGenerator(config)
        except ValueError as error:
            if not invalid:
                raise
            matrix.append(
                {"config": config, "status": "rejected_expected", "reason": str(error)}
            )
            continue
        if invalid:
            raise AssertionError("Mapped direct transfer was not rejected")
        rows = []
        for index, (inputs, baseline) in enumerate(zip(molecular_inputs, baselines)):
            start = time.perf_counter()
            result = generator.generate(*inputs)
            elapsed = time.perf_counter() - start
            record = result_record(result, generator, baseline)
            # Check reproducibility of both representative choice and order.
            repeated = generator.generate(*inputs)
            if [candidate.smiles for candidate in repeated] != [
                candidate.smiles for candidate in result
            ]:
                raise AssertionError(f"Nondeterministic candidate order: {config}")
            if (
                config["candidate_representation"] == "mapped"
                and config["dedup_key"] == "kekule"
            ):
                nodes, edges, final = oracle_results[index]
                common = {
                    key: config[key]
                    for key in ("map_storage", "shared_preparation", "bond_scan")
                }
                raw_node = [
                    candidate.smiles for candidate in merge_node(*inputs, **common)
                ]
                raw_edge = [
                    candidate.smiles for candidate in merge_edge(*inputs, **common)
                ]
                if (
                    nodes != raw_node
                    or edges != raw_edge
                    or final != [candidate.smiles for candidate in result]
                ):
                    raise AssertionError(
                        f"P1 differs from independent reference: {config}"
                    )
                record["raw_order_matches_reference"] = True
            record["fixture"] = FIXTURES[index][0]
            record["seconds"] = elapsed
            rows.append(record)
        matrix.append({"config": config, "status": "passed", "fixtures": rows})
    invalid_inputs = []
    for base, fragment in [
        (None, "CC"),
        ("", "CC"),
        ("C[C@H](O)F", "CC"),
        ("CC", "C/C=C/C"),
    ]:
        inputs = get_mol(base) if base is not None else None, get_mol(fragment)
        for name, function in [
            ("frozen_reference", reference_candidates),
            ("default_p1", CandidateGenerator().generate),
        ]:
            try:
                function(*inputs)
            except ValueError as error:
                invalid_inputs.append(
                    {
                        "base": base,
                        "fragment": fragment,
                        "implementation": name,
                        "status": "rejected",
                        "reason": str(error),
                    }
                )
            else:
                raise AssertionError(f"Invalid input accepted by {name}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "schema_version": 1,
        "metadata": {
            "revision": revision,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "repeats": repeats,
            "warmup_calls": 1,
            "timing": "CPU wall microbenchmark, tracing disabled",
            "determinism": "Seed-independent exhaustive deterministic enumeration",
            "reference_sha256": hashlib.sha256(
                Path("tests/reference_reassembly.py").read_bytes()
            ).hexdigest(),
            "source_sha256": {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in (
                    "moldr/chemistry.py",
                    "moldr/chemutils.py",
                    "moldr/core/reassemble.py",
                    "moldr/scoring.py",
                    "scripts/compare_chemistry.py",
                )
            },
            "nonzero_scan": "Dense adjacency remains O(V^2); numpy nonzero reduces Python visits.",
        },
        "default_chemistry": defaults,
        "summary": {
            "default_and_all_p1_raw_smiles_order_exact": True,
            "matrix_valid_configs": sum(row["status"] == "passed" for row in matrix),
            "matrix_invalid_configs_rejected": sum(
                row["status"] == "rejected_expected" for row in matrix
            ),
            "fixtures": len(fixtures),
            "p2_candidate_set_changes_permitted_and_recorded": True,
        },
        "fixtures": fixtures,
        "configuration_matrix": matrix,
        "invalid_inputs": invalid_inputs,
        "legacy_scoring_parity": benchmark_scoring(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments_validation/chemistry.json"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    report = compare(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
