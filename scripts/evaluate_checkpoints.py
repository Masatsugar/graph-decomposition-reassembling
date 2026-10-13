"""Evaluate the five major sweep configurations with matched sampling budgets.

Run this serial post-training comparison after the speed sweep finishes.
It performs no training, uses the saved checkpoint configuration/assets, and
measures Morgan/Tanimoto diversity independently of training uniqueness proxies.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from moldr.cli import run_evaluation
from moldr.provenance import checkpoint_path, load_saved_config, write_json
from scripts.experiment_report import seed_statistics

MAIN_CONFIGURATIONS = (
    "mol2vec_actor",
    "actor_value",
    "auxiliary_only",
    "mpnn_frozen",
    "mpnn_trainable",
)
METRICS = (
    "mean_score",
    "validity",
    "unique_fraction",
    "fingerprint_diversity",
    "seconds",
    "episodes_per_second",
    "score_requests",
    "score_computations",
    "score_hits",
)


def make_plan(sweep_directory, *, episodes=3, evaluation_seed=0, num_gpus=1.0):
    if episodes <= 0 or evaluation_seed < 0 or num_gpus not in (0, 1):
        raise ValueError(
            "Episodes must be positive; evaluation seed nonnegative; num_gpus zero or one"
        )
    directory = Path(sweep_directory).resolve()
    sweep = json.loads((directory / "sweep.json").read_text())
    by_name = {job["name"]: job for job in sweep["jobs"]}
    jobs = []
    for configuration in MAIN_CONFIGURATIONS:
        for training_seed in (0, 1, 2):
            name = f"{configuration}_seed{training_seed}_workers8"
            if name not in by_name:
                raise ValueError(
                    f"The major-configuration checkpoint is absent from this sweep: {name}"
                )
            source = by_name[name]
            source_run = Path(source.get("directory", directory / name)).resolve()
            ready = (
                source["status"] == "complete"
                and (source_run / "last_checkpoint.json").is_file()
            )
            checkpoint = checkpoint_path(source_run) if ready else None
            if checkpoint is not None:
                ready = (Path(checkpoint) / "algorithm_state.pkl").is_file()
            jobs.append(
                {
                    "name": name,
                    "configuration": configuration,
                    "training_seed": training_seed,
                    "source_run": str(source_run),
                    "checkpoint": checkpoint,
                    "source_status": source["status"],
                    "ready": ready,
                }
            )
    return {
        "sweep_directory": str(directory),
        "completed_sweep_jobs": sum(
            job["status"] == "complete" for job in sweep["jobs"]
        ),
        "total_sweep_jobs": len(sweep["jobs"]),
        "sweep_complete": all(job["status"] == "complete" for job in sweep["jobs"]),
        "serial": True,
        "training": False,
        "num_workers": 0,
        "num_gpus": float(num_gpus),
        "explore": True,
        "episodes_per_checkpoint": episodes,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + episodes)),
        "ready_checkpoints": sum(job["ready"] for job in jobs),
        "total_checkpoints": len(jobs),
        "jobs": jobs,
    }


def summarize(rows):
    return [
        {
            "configuration": configuration,
            "n": len(trials),
            "total_episodes": sum(row["episodes"] for row in trials),
            "metrics": {
                metric: seed_statistics(row.get(metric) for row in trials)
                for metric in METRICS
            },
        }
        for configuration in MAIN_CONFIGURATIONS
        if (trials := [row for row in rows if row["configuration"] == configuration])
    ]


def plot_groups(groups, directory):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/moldr_posthoc_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9, 8))
    positions = np.arange(len(groups))
    for axis, metric, label, color in (
        (axes[0], "mean_score", "Mean final objective score", "#315d89"),
        (axes[1], "fingerprint_diversity", "Morgan/Tanimoto diversity", "#24826f"),
    ):
        live = [
            index
            for index, group in enumerate(groups)
            if group["metrics"][metric]["mean"] is not None
        ]
        axis.bar(
            live,
            [groups[index]["metrics"][metric]["mean"] for index in live],
            yerr=[groups[index]["metrics"][metric]["sample_sd"] or 0 for index in live],
            capsize=4,
            color=color,
        )
        axis.set_xticks(
            positions,
            [
                f"{group['configuration']}\nn={group['n']} checkpoints"
                for group in groups
            ],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[1].set_ylim(0, 1)
    figure.suptitle(
        "Matched evaluation seeds/budgets: means and sample SD across training seeds"
    )
    figure.tight_layout()
    figure.savefig(directory / "quality_diversity.png", dpi=180)
    figure.savefig(directory / "quality_diversity.svg")
    plt.close(figure)


def evaluate_sweep(
    sweep_directory,
    output,
    *,
    episodes=3,
    evaluation_seed=0,
    num_gpus=1.0,
    plan_only=False,
):
    plan = make_plan(
        sweep_directory,
        episodes=episodes,
        evaluation_seed=evaluation_seed,
        num_gpus=num_gpus,
    )
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "evaluation_plan.json", plan)
    if plan_only:
        print(
            json.dumps(
                {
                    "output": str(directory),
                    "ready_checkpoints": plan["ready_checkpoints"],
                    "total_checkpoints": plan["total_checkpoints"],
                    "sweep_complete": plan["sweep_complete"],
                    "evaluation_seeds": plan["evaluation_seeds"],
                    "training": False,
                }
            ),
            flush=True,
        )
        return directory
    if not plan["sweep_complete"]:
        raise ValueError(
            "Post-training evaluation waits for the speed sweep to complete"
        )
    if plan["ready_checkpoints"] != plan["total_checkpoints"]:
        raise ValueError(
            "All fifteen major checkpoints must be complete before this evaluation"
        )
    if (directory / "comparison.json").exists() or (directory / "runs.json").exists():
        raise ValueError(
            f"Refusing to overwrite an existing post-training evaluation: {directory}"
        )
    for job in plan["jobs"]:
        if (directory / "runs" / job["name"]).exists():
            raise ValueError(f"Evaluation run directory already exists: {job['name']}")
    rows = []
    for job in plan["jobs"]:
        config, saved_run = load_saved_config(
            job["checkpoint"],
            [
                "training.num_workers=0",
                f"training.num_gpus={num_gpus}",
                "training.num_cpus=2",
                f"training.seed={evaluation_seed}",
                f"evaluation.episodes={episodes}",
                "evaluation.explore=true",
                "evaluation.random=false",
                "evaluation.selector=actor",
            ],
        )
        source_manifest = json.loads((saved_run / "manifest.json").read_text())
        if source_manifest["seed"] != job["training_seed"]:
            raise ValueError(
                f"Checkpoint training seed differs from the explicit sweep job: {job['name']}"
            )
        config.output.directory = str(directory / "runs")
        config.output.run_name = job["name"]
        config.output.checkpoint = job["checkpoint"]
        result_dir = run_evaluation(config, checkpoint=job["checkpoint"])
        evaluation = json.loads((result_dir / "evaluation.json").read_text())
        molecules = json.loads((result_dir / "molecules.json").read_text())
        if [row["seed"] for row in molecules] != plan["evaluation_seeds"]:
            raise RuntimeError("Evaluation did not use the shared episode seed budget")
        scoring = evaluation["scoring"] or {}
        rows.append(
            {
                "name": job["name"],
                "configuration": job["configuration"],
                "training_seed": job["training_seed"],
                "checkpoint": job["checkpoint"],
                "directory": str(result_dir.relative_to(directory)),
                "episodes": evaluation["episodes"],
                "mean_score": evaluation["mean_score"],
                "validity": evaluation["validity"],
                "unique_fraction": evaluation["unique_fraction"],
                "fingerprint_diversity": evaluation["diversity"],
                "seconds": evaluation["seconds"],
                "episodes_per_second": evaluation["episodes_per_second"],
                "score_requests": scoring.get("requests", 0),
                "score_computations": scoring.get("computations", 0),
                "score_hits": scoring.get("hits", 0),
            }
        )
        write_json(directory / "runs.json", rows)
    report = {
        "completed_checkpoints": len(rows),
        "total_checkpoints": len(plan["jobs"]),
        "evaluation_seeds": plan["evaluation_seeds"],
        "episodes_per_checkpoint": episodes,
        "num_gpus": num_gpus,
        "explore": True,
        "training": False,
        "diversity_definition": "One minus mean pairwise Tanimoto similarity of radius-2, 2048-bit Morgan fingerprints",
        "statistics_scope": "Mean and sample SD of per-checkpoint metrics across training seeds0/1/2",
        "groups": summarize(rows),
        "runs": rows,
    }
    write_json(directory / "comparison.json", report)
    plot_groups(report["groups"], directory)
    print(
        json.dumps({"output": str(directory), "groups": report["groups"]}), flush=True
    )
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=float, choices=(0, 1), default=1)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Save readiness/budget plan without Ray, GPU, or evaluation",
    )
    args = parser.parse_args(argv)
    return evaluate_sweep(
        args.sweep_directory,
        args.output,
        episodes=args.episodes,
        evaluation_seed=args.evaluation_seed,
        num_gpus=args.num_gpus,
        plan_only=args.plan,
    )


if __name__ == "__main__":
    main()
