"""Compare actor and state-value lookahead on one saved E2E checkpoint.

The rollout timer covers reset, action selection, chemical transitions and
objective evaluation. Configuration/model loading, Ray startup, and output
writing are included separately in each run's wall time. Both selectors use
the same starting seeds and episode budget, with CPU evaluation by default.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from moldr.cli import run_evaluation
from moldr.factory import resolve_config
from moldr.provenance import checkpoint_path, load_saved_config, write_json
from moldr.sampling import EpisodeResult, Sampler


def episode_metrics(episode: EpisodeResult, seconds: float) -> dict:
    """Validate and retain actual request deltas, including every preview."""
    previews = selected_attachment_requests = 0
    attachment_steps = 0
    previous_requests = None
    details = []
    for info in episode.infos:
        requests = info.get("scoring", {}).get("requests")
        if info.get("phase") == "select_attach":
            attachment_steps += 1
        choice = info.get("lookahead")
        if choice is not None:
            count = len(choice["returns"])
            if choice["score_requests"] != count:
                raise RuntimeError(
                    "A lookahead must request one score per presented candidate"
                )
            if requests is None or previous_requests is None:
                raise RuntimeError(
                    "Lookahead validation requires transition scoring counters"
                )
            actual = requests - previous_requests
            if actual != count + 1:
                raise RuntimeError(
                    f"Observed {actual} score requests for {count} previews; expected K+1"
                )
            previews += count
            selected_attachment_requests += 1
            details.append({"candidates": count, "actual_score_requests": actual})
        if requests is not None:
            previous_requests = requests
    if not np.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("Episode timing must be positive and finite")
    return {
        "smiles": episode.smiles,
        "score": episode.score,
        "reward": episode.total_reward,
        "steps": episode.steps,
        "terminated": episode.terminated,
        "truncated": episode.truncated,
        "seconds": seconds,
        "attachment_steps": attachment_steps,
        "score_requests": episode.scoring.get("requests", 0),
        "score_computations": episode.scoring.get("computations", 0),
        "score_hits": episode.scoring.get("hits", 0),
        "lookahead_previews": previews,
        "lookahead_selected_attachment_requests": selected_attachment_requests,
        "lookahead_transitions": details,
    }


@dataclass
class EpisodeCapture:
    records: list[dict] = field(default_factory=list)

    def timed(self, original: Callable):
        def episode(sampler, *args, **kwargs):
            start = time.perf_counter()
            result = original(sampler, *args, **kwargs)
            record = episode_metrics(result, time.perf_counter() - start)
            record["seed"] = kwargs.get("seed")
            self.records.append(record)
            return result

        return episode


def aggregate(selector: str, runs: list[dict]) -> dict:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    records = [
        record
        for run in runs
        if run["selector"] == selector
        for record in run["episodes"]
    ]
    if not records:
        raise ValueError("Cannot aggregate an empty evaluation")
    seconds = sum(record["seconds"] for record in records)
    mols = [Chem.MolFromSmiles(record["smiles"]) for record in records]
    valid = [mol for mol in mols if mol is not None]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(mol) for mol in valid]
    similarities = [
        DataStructs.TanimotoSimilarity(left, right)
        for index, left in enumerate(fingerprints)
        for right in fingerprints[:index]
    ]
    requests = sum(record["score_requests"] for record in records)
    computations = sum(record["score_computations"] for record in records)
    updates = sum(record["attachment_steps"] for record in records)
    return {
        "selector": selector,
        "episodes": len(records),
        "mean_score": float(np.mean([record["score"] for record in records])),
        "score_std": float(np.std([record["score"] for record in records])),
        "validity": len(valid) / len(records),
        "unique_fraction": len({Chem.MolToSmiles(mol) for mol in valid}) / len(records),
        "diversity": 1 - float(np.mean(similarities)) if similarities else None,
        "rollout_seconds": seconds,
        "full_run_wall_seconds": sum(
            run["wall_seconds"] for run in runs if run["selector"] == selector
        ),
        "attachment_steps": updates,
        "molecule_updates_per_second": updates / seconds,
        "valid_molecules_per_second": len(valid) / seconds,
        "score_requests": requests,
        "score_computations": computations,
        "score_hits": sum(record["score_hits"] for record in records),
        "score_requests_per_second": requests / seconds,
        "score_computations_per_second": computations / seconds,
        "lookahead_previews": sum(record["lookahead_previews"] for record in records),
        "lookahead_selected_attachment_requests": sum(
            record["lookahead_selected_attachment_requests"] for record in records
        ),
    }


def plot_comparison(report, directory):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/moldr_evaluation_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = report["summary"]
    labels = [row["selector"] for row in rows]
    colors = ["#315d89", "#24826f"]
    figure, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0, 0].bar(
        labels,
        [row["mean_score"] for row in rows],
        yerr=[row["score_std"] for row in rows],
        capsize=4,
        color=colors,
    )
    axes[0, 0].set_ylabel("Mean objective score")
    axes[0, 1].bar(labels, [row["unique_fraction"] for row in rows], color=colors)
    axes[0, 1].set_ylabel("Unique final molecule fraction")
    axes[0, 1].set_ylim(0, 1)
    axes[1, 0].bar(
        labels, [row["score_requests"] / row["episodes"] for row in rows], color=colors
    )
    axes[1, 0].set_ylabel("Score requests / final molecule")
    axes[1, 1].bar(
        labels, [row["rollout_seconds"] / row["episodes"] for row in rows], color=colors
    )
    axes[1, 1].set_ylabel("Rollout seconds / final molecule")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle("Same checkpoint, starting seeds and episode budgets")
    figure.tight_layout()
    figure.savefig(directory / "comparison.png", dpi=180)
    figure.savefig(directory / "comparison.svg")
    plt.close(figure)


def compare_checkpoint(checkpoint, output, *, episodes=3, seeds=(0, 1, 2)):
    if (
        episodes < 1
        or not seeds
        or any(seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "episodes must be positive and seeds must be distinct nonnegative integers"
        )
    checkpoint = Path(checkpoint).resolve()
    directory = Path(output).resolve()
    saved, saved_run = load_saved_config(checkpoint)
    if saved.environment.mode != "e2e":
        raise ValueError("The comparison requires a new E2E checkpoint")
    # Validate the complete lookahead restriction before starting either run.
    preflight = copy.deepcopy(saved)
    preflight.evaluation.selector = "value_lookahead"
    preflight.evaluation.random = False
    resolve_config(preflight)
    if (directory / "comparison.json").exists():
        raise ValueError(f"Refusing to overwrite an existing comparison: {directory}")
    configs = []
    for seed in seeds:
        for selector in ("actor", "value_lookahead"):
            config = copy.deepcopy(saved)
            config.training.seed = int(seed)
            config.training.num_workers = 0
            config.training.num_gpus = 0.0
            config.training.num_cpus = 2
            config.evaluation.selector = selector
            config.evaluation.episodes = int(episodes)
            config.evaluation.random = False
            config.evaluation.explore = False
            config.output.directory = str(directory / "runs")
            config.output.run_name = f"{selector}_seed{seed}"
            if (directory / "runs" / config.output.run_name).exists():
                raise ValueError(
                    f"Evaluation directory already exists: {config.output.run_name}"
                )
            configs.append(config)
    directory.mkdir(parents=True, exist_ok=True)
    runs = []
    for config in configs:
        capture = EpisodeCapture()
        start = time.perf_counter()
        # The existing CLI owns restoration, masks, local config/output, and
        # quality/diversity calculation. Capture only its actual episode work.
        with patch.object(Sampler, "episode", capture.timed(Sampler.episode)):
            result_dir = run_evaluation(config, checkpoint=str(checkpoint))
        wall_seconds = time.perf_counter() - start
        evaluation = json.loads((result_dir / "evaluation.json").read_text())
        actual = {
            key: sum(record[field] for record in capture.records)
            for key, field in (
                ("requests", "score_requests"),
                ("computations", "score_computations"),
                ("hits", "score_hits"),
            )
        }
        if evaluation["scoring"] != actual:
            raise RuntimeError(
                "Captured episode score counters do not match the CLI report"
            )
        if len(capture.records) != episodes:
            raise RuntimeError("The CLI did not complete the requested episode budget")
        run = {
            "selector": config.evaluation.selector,
            "starting_seed": config.training.seed,
            "episode_seeds": [record["seed"] for record in capture.records],
            "directory": str(result_dir.relative_to(directory)),
            "wall_seconds": wall_seconds,
            "evaluation": evaluation,
            "episodes": capture.records,
        }
        runs.append(run)
        write_json(directory / "runs.json", runs)
    report = {
        "checkpoint": checkpoint_path(checkpoint),
        "saved_run": str(saved_run),
        "device": "cpu",
        "starting_seeds": list(seeds),
        "episodes_per_starting_seed": episodes,
        "explore": False,
        "same_checkpoint": True,
        "timing_scope": "Rollout seconds include reset/actions/transitions/scoring; full wall time includes startup/restore/output",
        "seed_scope": "Each CLI run uses starting_seed + episode_index; seed ranges may overlap",
        "runs": runs,
        "summary": [
            aggregate(selector, runs) for selector in ("actor", "value_lookahead")
        ],
    }
    write_json(directory / "comparison.json", report)
    plot_comparison(report, directory)
    print(
        json.dumps({"output": str(directory), "summary": report["summary"]}), flush=True
    )
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodes per selector and starting seed",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    default = Path("artifacts/experiments") / (
        "evaluation_"
        + datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args(argv)
    return compare_checkpoint(
        args.checkpoint, args.output, episodes=args.episodes, seeds=tuple(args.seeds)
    )


if __name__ == "__main__":
    main()
