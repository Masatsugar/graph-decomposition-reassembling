"""Evaluate the five major sweep configurations with matched sampling budgets.

Run this serial post-training comparison after the speed sweep finishes.
It performs no training, uses the saved checkpoint configuration/assets, and
measures Morgan/Tanimoto diversity independently of training uniqueness proxies.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from moldr.cli import run_evaluation
from moldr.provenance import (
    checkpoint_path,
    load_saved_config,
    sha256_file,
    write_json,
)
from moldr.sampling import Sampler
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


def make_plan(
    sweep_directory,
    *,
    episodes=3,
    evaluation_seed=0,
    num_gpus=1.0,
    training_seeds=(0, 1, 2),
    checkpoint_iteration=None,
    explore=True,
    diagnostics=False,
):
    if episodes <= 0 or evaluation_seed < 0 or num_gpus not in (0, 1):
        raise ValueError(
            "Episodes must be positive; evaluation seed nonnegative; num_gpus zero or one"
        )
    try:
        training_seeds = tuple(training_seeds)
    except TypeError as error:
        raise ValueError(
            "Training seeds must be distinct nonnegative integers"
        ) from error
    if (
        not training_seeds
        or any(type(seed) is not int or seed < 0 for seed in training_seeds)
        or len(set(training_seeds)) != len(training_seeds)
    ):
        raise ValueError("Training seeds must be distinct nonnegative integers")
    if checkpoint_iteration is not None and (
        type(checkpoint_iteration) is not int or checkpoint_iteration <= 0
    ):
        raise ValueError("Checkpoint iteration must be a positive integer")
    directory = Path(sweep_directory).resolve()
    sweep = json.loads((directory / "sweep.json").read_text())
    by_name = {job["name"]: job for job in sweep["jobs"]}
    jobs = []
    for configuration in MAIN_CONFIGURATIONS:
        for training_seed in training_seeds:
            name = f"{configuration}_seed{training_seed}_workers8"
            if name not in by_name:
                raise ValueError(
                    f"The major-configuration checkpoint is absent from this sweep: {name}"
                )
            source = by_name[name]
            source_run = Path(source.get("directory", directory / name)).resolve()
            if checkpoint_iteration is None:
                ready = (
                    source["status"] == "complete"
                    and (source_run / "last_checkpoint.json").is_file()
                )
                checkpoint = checkpoint_path(source_run) if ready else None
            else:
                checkpoint = str(
                    source_run / "checkpoints" / f"iteration_{checkpoint_iteration:06d}"
                )
                ready = source["status"] == "complete"
            if checkpoint is not None:
                ready = ready and (Path(checkpoint) / "algorithm_state.pkl").is_file()
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
        "explore": bool(explore),
        "diagnostics": bool(diagnostics),
        "training_seeds": list(training_seeds),
        "checkpoint_iteration": checkpoint_iteration,
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
            color=color,
        )
        uncertainty = [
            index
            for index in live
            if groups[index]["metrics"][metric]["sample_sd"] is not None
        ]
        if uncertainty:
            axis.errorbar(
                uncertainty,
                [groups[index]["metrics"][metric]["mean"] for index in uncertainty],
                yerr=[
                    groups[index]["metrics"][metric]["sample_sd"]
                    for index in uncertainty
                ],
                fmt="none",
                ecolor="#333333",
                capsize=4,
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
    figure.suptitle("Means across training seeds; sample SD shown only for n>=2")
    figure.tight_layout()
    figure.savefig(directory / "quality_diversity.png", dpi=180)
    figure.savefig(directory / "quality_diversity.svg")
    plt.close(figure)


def _plan_identity(plan):
    """Ignore readiness changes while protecting the selected evaluation budget."""
    defaults = {
        "training_seeds": [0, 1, 2],
        "checkpoint_iteration": None,
        "diagnostics": False,
    }
    fields = (
        "sweep_directory",
        "num_workers",
        "num_gpus",
        "explore",
        "episodes_per_checkpoint",
        "evaluation_seeds",
        *defaults,
    )
    result = {key: plan.get(key, defaults.get(key)) for key in fields}
    result["jobs"] = [
        {
            key: job[key]
            for key in (
                "name",
                "configuration",
                "training_seed",
                "source_run",
                "checkpoint",
            )
        }
        for job in plan["jobs"]
    ]
    return result


def _timing_scope(diagnostics):
    capture = (
        "Diagnostic capture and entropy bookkeeping are included. "
        if diagnostics
        else "Diagnostic capture is disabled. "
    )
    return (
        "Episode seconds include reset, policy selection, chemical transitions and scoring. "
        + capture
        + "Model/Ray startup, checkpoint restoration, output writing and offline diagnostic analysis/audits are excluded."
    )


def _read_row(job, result_dir, directory, plan):
    evaluation = json.loads((result_dir / "evaluation.json").read_text())
    molecules = json.loads((result_dir / "molecules.json").read_text())
    if (
        evaluation["episodes"] != plan["episodes_per_checkpoint"]
        or [row["seed"] for row in molecules] != plan["evaluation_seeds"]
    ):
        raise RuntimeError("Evaluation did not use the shared episode seed budget")
    scoring = evaluation["scoring"] or {}
    row = {
        "name": job["name"],
        "configuration": job["configuration"],
        "training_seed": job["training_seed"],
        "checkpoint_iteration": plan["checkpoint_iteration"],
        "explore": plan["explore"],
        "checkpoint": job["checkpoint"],
        "directory": str(result_dir.relative_to(directory)),
        "episodes": evaluation["episodes"],
        "mean_score": evaluation["mean_score"],
        "validity": evaluation["validity"],
        "unique_fraction": evaluation["unique_fraction"],
        "fingerprint_diversity": evaluation["diversity"],
        "seconds": evaluation["seconds"],
        "timing_scope": _timing_scope(plan["diagnostics"]),
        "episodes_per_second": evaluation["episodes_per_second"],
        "score_requests": scoring.get("requests", 0),
        "score_computations": scoring.get("computations", 0),
        "score_hits": scoring.get("hits", 0),
    }
    if plan["diagnostics"]:
        path = result_dir / "molecular_diagnostics.json"
        if not path.is_file():
            raise ValueError(f"Incomplete diagnostic output: {result_dir}")
        diagnostic = json.loads(path.read_text())
        episodes = diagnostic.get("episodes") if isinstance(diagnostic, dict) else None
        summary = diagnostic.get("summary") if isinstance(diagnostic, dict) else None
        if (
            not isinstance(episodes, list)
            or len(episodes) != plan["episodes_per_checkpoint"]
            or not isinstance(summary, dict)
            or type(summary.get("episode_count")) is not int
            or summary["episode_count"] != plan["episodes_per_checkpoint"]
            or any(not isinstance(episode, dict) for episode in episodes)
            or [episode.get("seed") for episode in episodes] != plan["evaluation_seeds"]
        ):
            raise ValueError(
                f"Incomplete or mismatched diagnostic episode budget: {result_dir}"
            )
        row["diagnostics"] = str(path.relative_to(directory))
    return row


def _reuse_row(job, result_dir, directory, plan):
    required = (
        "evaluation.json",
        "molecules.json",
        "manifest.json",
        "resolved_config.yaml",
        "evaluation_checkpoint.json",
    )
    if any(not (result_dir / name).is_file() for name in required):
        raise ValueError(
            f"Incomplete evaluation cannot be resumed or overwritten: {result_dir}"
        )
    # Validate the saved evaluation assets/configuration as well as its source.
    saved, _ = load_saved_config(result_dir)
    if (
        saved.training.seed != plan["evaluation_seeds"][0]
        or saved.training.num_workers != plan["num_workers"]
        or saved.training.num_gpus != plan["num_gpus"]
        or saved.evaluation.explore != plan["explore"]
        or saved.evaluation.random
        or saved.evaluation.selector != "actor"
        or saved.evaluation.episodes != plan["episodes_per_checkpoint"]
    ):
        raise ValueError(
            f"Saved evaluation settings differ from the resume plan: {result_dir}"
        )
    checkpoint = Path(job["checkpoint"])
    recorded = json.loads((result_dir / "evaluation_checkpoint.json").read_text())
    actual = {
        str(path.relative_to(checkpoint)): sha256_file(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    }
    if Path(recorded["path"]).resolve() != checkpoint or recorded["sha256"] != actual:
        raise ValueError(
            f"Checkpoint differs from the completed evaluation: {result_dir}"
        )
    return _read_row(job, result_dir, directory, plan)


def evaluate_sweep(
    sweep_directory,
    output,
    *,
    episodes=3,
    evaluation_seed=0,
    num_gpus=1.0,
    plan_only=False,
    training_seeds=(0, 1, 2),
    checkpoint_iteration=None,
    explore=True,
    diagnostics=False,
    resume=False,
):
    plan = make_plan(
        sweep_directory,
        episodes=episodes,
        evaluation_seed=evaluation_seed,
        num_gpus=num_gpus,
        training_seeds=training_seeds,
        checkpoint_iteration=checkpoint_iteration,
        explore=explore,
        diagnostics=diagnostics,
    )
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "evaluation_plan.json"
    previous = json.loads(plan_path.read_text()) if plan_path.is_file() else None
    existing_results = (
        (directory / "comparison.json").exists()
        or (directory / "runs.json").exists()
        or ((directory / "runs").is_dir() and any((directory / "runs").iterdir()))
    )
    if resume:
        if previous is None or _plan_identity(previous) != _plan_identity(plan):
            raise ValueError("Resume requires an existing matching evaluation plan")
    elif existing_results and (
        not plan_only
        or previous is None
        or _plan_identity(previous) != _plan_identity(plan)
    ):
        raise ValueError(
            f"Refusing to overwrite an existing post-training evaluation: {directory}; use --resume"
        )
    write_json(plan_path, plan)
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
            f"All {plan['total_checkpoints']} planned checkpoints must be ready before this evaluation "
            f"({plan['ready_checkpoints']} ready)"
        )
    reused = {}
    for job in plan["jobs"]:
        result_dir = directory / "runs" / job["name"]
        if result_dir.exists():
            reused[job["name"]] = _reuse_row(job, result_dir, directory, plan)
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
                f"evaluation.explore={str(bool(explore)).lower()}",
                "evaluation.random=false",
                "evaluation.selector=actor",
            ],
        )
        source_manifest = json.loads((saved_run / "manifest.json").read_text())
        if source_manifest["seed"] != job["training_seed"]:
            raise ValueError(
                f"Checkpoint training seed differs from the explicit sweep job: {job['name']}"
            )
        if job["name"] in reused:
            rows.append(reused[job["name"]])
            write_json(directory / "runs.json", rows)
            continue
        config.output.directory = str(directory / "runs")
        config.output.run_name = job["name"]
        config.output.checkpoint = job["checkpoint"]
        diagnostic = None
        if diagnostics:
            from scripts.molecular_diagnostics import MolecularDiagnostics

            diagnostic = MolecularDiagnostics()
        capture = (
            patch.object(Sampler, "episode", diagnostic.capture(Sampler.episode))
            if diagnostic is not None
            else nullcontext()
        )
        with capture:
            result_dir = run_evaluation(config, checkpoint=job["checkpoint"])
        if diagnostic is not None:
            diagnostic.write(result_dir)
        rows.append(_read_row(job, result_dir, directory, plan))
        write_json(directory / "runs.json", rows)
    report = {
        "completed_checkpoints": len(rows),
        "total_checkpoints": len(plan["jobs"]),
        "evaluation_seeds": plan["evaluation_seeds"],
        "episodes_per_checkpoint": episodes,
        "num_gpus": num_gpus,
        "explore": bool(explore),
        "diagnostics": bool(diagnostics),
        "timing_scope": _timing_scope(diagnostics),
        "training_seeds": plan["training_seeds"],
        "checkpoint_iteration": checkpoint_iteration,
        "training": False,
        "diversity_definition": "One minus mean pairwise Tanimoto similarity of radius-2, 2048-bit Morgan fingerprints",
        "statistics_scope": "Mean of per-checkpoint metrics across the selected training seeds; sample SD is undefined (null) for n=1 and shown only for n>=2",
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
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--checkpoint-iteration", type=int)
    parser.add_argument("--no-explore", dest="explore", action="store_false")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete matching evaluations; incomplete run directories are rejected",
    )
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
        training_seeds=tuple(args.training_seeds),
        checkpoint_iteration=args.checkpoint_iteration,
        explore=args.explore,
        diagnostics=args.diagnostics,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
