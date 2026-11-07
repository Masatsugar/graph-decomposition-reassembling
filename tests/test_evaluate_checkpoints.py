"""Post-training plans keep seed budgets and independent checkpoint groups."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from moldr.experiment import load_config
from moldr.provenance import sha256_file, write_json
from scripts import evaluate_checkpoints


def _sweep(tmp_path, seeds=(0, 1, 2)):
    jobs = []
    for configuration in evaluate_checkpoints.MAIN_CONFIGURATIONS:
        for seed in seeds:
            name = f"{configuration}_seed{seed}_workers8"
            run = tmp_path / name
            (run / "checkpoint").mkdir(parents=True)
            (run / "checkpoint" / "algorithm_state.pkl").write_bytes(b"checkpoint")
            (run / "last_checkpoint.json").write_text(
                json.dumps({"path": "checkpoint"})
            )
            (run / "manifest.json").write_text(json.dumps({"seed": seed}))
            jobs.append({"name": name, "status": "complete", "directory": str(run)})
    (tmp_path / "sweep.json").write_text(json.dumps({"jobs": jobs}))
    return jobs


def _fake_evaluation(monkeypatch):
    calls = []

    def restore(checkpoint, overrides=()):
        path = Path(checkpoint)
        if (path / "resolved_config.yaml").is_file():
            return load_config(path / "resolved_config.yaml", overrides), path
        run = next(
            parent for parent in path.parents if (parent / "manifest.json").is_file()
        )
        seed = json.loads((run / "manifest.json").read_text())["seed"]
        return load_config(overrides=[f"training.seed={seed}", *overrides]), run

    def evaluate(config, *, checkpoint):
        calls.append((config, checkpoint))
        output = Path(config.output.directory) / config.output.run_name
        output.mkdir(parents=True)
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(config.to_dict()))
        write_json(output / "manifest.json", {"seed": config.training.seed})
        write_json(
            output / "evaluation_checkpoint.json",
            {
                "path": checkpoint,
                "sha256": {
                    str(path.relative_to(checkpoint)): sha256_file(path)
                    for path in Path(checkpoint).rglob("*")
                    if path.is_file()
                },
            },
        )
        episodes = config.evaluation.episodes
        write_json(
            output / "molecules.json",
            [
                {"seed": config.training.seed + index, "smiles": "CC"}
                for index in range(episodes)
            ],
        )
        write_json(
            output / "evaluation.json",
            {
                "episodes": episodes,
                "mean_score": 0.5,
                "validity": 1.0,
                "unique_fraction": 1 / episodes,
                "diversity": 0.0,
                "seconds": 1.0,
                "episodes_per_second": episodes,
                "scoring": {"requests": episodes, "computations": episodes, "hits": 0},
            },
        )
        return output

    monkeypatch.setattr(evaluate_checkpoints, "load_saved_config", restore)
    monkeypatch.setattr(evaluate_checkpoints, "run_evaluation", evaluate)
    monkeypatch.setattr(evaluate_checkpoints, "plot_groups", lambda *args: None)
    return calls


def _fake_diagnostics(monkeypatch):
    original_evaluate = evaluate_checkpoints.run_evaluation
    instances = []
    original_episode = lambda self, **kwargs: None
    monkeypatch.setattr(evaluate_checkpoints.Sampler, "episode", original_episode)

    class Diagnostic:
        def __init__(self):
            self.seeds = []
            instances.append(self)

        def capture(self, original):
            def episode(sampler, *, seed):
                self.seeds.append(seed)
                return original(sampler, seed=seed)

            return episode

        def write(self, directory):
            path = Path(directory) / "molecular_diagnostics.json"
            write_json(
                path,
                {
                    "episodes": [{"seed": seed} for seed in self.seeds],
                    "summary": {"episode_count": len(self.seeds)},
                },
            )
            return path

    def evaluate(config, *, checkpoint):
        for index in range(config.evaluation.episodes):
            evaluate_checkpoints.Sampler.episode(
                object(), seed=config.training.seed + index
            )
        return original_evaluate(config, checkpoint=checkpoint)

    monkeypatch.setitem(
        sys.modules,
        "scripts.molecular_diagnostics",
        SimpleNamespace(MolecularDiagnostics=Diagnostic),
    )
    monkeypatch.setattr(evaluate_checkpoints, "run_evaluation", evaluate)
    return instances, original_episode


def test_plan_keeps_identical_seed_budget_and_blocks_during_speed_measurement(tmp_path):
    jobs = _sweep(tmp_path)
    plan = evaluate_checkpoints.make_plan(tmp_path, episodes=3, evaluation_seed=10)
    assert plan["ready_checkpoints"] == plan["total_checkpoints"] == 15
    assert plan["evaluation_seeds"] == [10, 11, 12]
    assert plan["num_workers"] == 0 and plan["num_gpus"] == 1.0
    assert plan["explore"] is True and plan["training"] is False
    jobs.append({"name": "map_string_seed0_workers8", "status": "running"})
    (tmp_path / "sweep.json").write_text(json.dumps({"jobs": jobs}))
    with pytest.raises(ValueError, match="speed sweep"):
        evaluate_checkpoints.evaluate_sweep(tmp_path, tmp_path / "posthoc")


def test_summary_uses_checkpoint_seed_variance_and_nullable_diversity():
    rows = [
        {
            "configuration": "mol2vec_actor",
            "episodes": 3,
            "mean_score": score,
            "fingerprint_diversity": None,
        }
        for score in (0.2, 0.4, 0.6)
    ]
    group = evaluate_checkpoints.summarize(rows)[0]
    assert group["n"] == 3 and group["total_episodes"] == 9
    assert group["metrics"]["mean_score"]["mean"] == pytest.approx(0.4)
    assert group["metrics"]["mean_score"]["sample_sd"] == pytest.approx(0.2)
    assert group["metrics"]["fingerprint_diversity"] == {
        "mean": None,
        "sample_sd": None,
        "n": 0,
    }


@pytest.mark.parametrize("seeds", [(), (0, 0), (-1,), (0.5,), (True,), None])
def test_plan_rejects_invalid_training_seeds(tmp_path, seeds):
    with pytest.raises(ValueError, match="Training seeds"):
        evaluate_checkpoints.make_plan(tmp_path, training_seeds=seeds)


@pytest.mark.parametrize("iteration", [0, -1, 1.5, True])
def test_plan_rejects_invalid_checkpoint_iteration(tmp_path, iteration):
    with pytest.raises(ValueError, match="Checkpoint iteration"):
        evaluate_checkpoints.make_plan(tmp_path, checkpoint_iteration=iteration)


@pytest.mark.parametrize("iteration", [100, 400, 500])
def test_plan_selects_requested_iteration_instead_of_latest_checkpoint(
    monkeypatch, tmp_path, iteration
):
    jobs = _sweep(tmp_path, seeds=(0,))
    for job in jobs:
        checkpoint = (
            Path(job["directory"]) / "checkpoints" / f"iteration_{iteration:06d}"
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "algorithm_state.pkl").write_bytes(b"earlier-checkpoint")
    plan = evaluate_checkpoints.make_plan(
        tmp_path, training_seeds=(0,), checkpoint_iteration=iteration, explore=False
    )
    assert plan["total_checkpoints"] == plan["ready_checkpoints"] == 5
    assert plan["training_seeds"] == [0] and plan["checkpoint_iteration"] == iteration
    assert plan["explore"] is False
    assert all(
        job["checkpoint"].endswith(f"checkpoints/iteration_{iteration:06d}")
        for job in plan["jobs"]
    )
    calls = _fake_evaluation(monkeypatch)
    evaluate_checkpoints.evaluate_sweep(
        tmp_path,
        tmp_path / "evaluation",
        training_seeds=(0,),
        checkpoint_iteration=iteration,
    )
    assert len(calls) == 5 and all(
        path.endswith(f"iteration_{iteration:06d}") for _, path in calls
    )
    (
        Path(jobs[0]["directory"])
        / "checkpoints"
        / f"iteration_{iteration:06d}"
        / "algorithm_state.pkl"
    ).unlink()
    with pytest.raises(ValueError, match="All 5 planned checkpoints.*4 ready"):
        evaluate_checkpoints.evaluate_sweep(
            tmp_path,
            tmp_path / "output",
            training_seeds=(0,),
            checkpoint_iteration=iteration,
        )


def test_single_seed_greedy_evaluation_preserves_shared_budget_and_source_guard(
    monkeypatch, tmp_path
):
    jobs = _sweep(tmp_path, seeds=(0,))
    calls = _fake_evaluation(monkeypatch)
    output = evaluate_checkpoints.evaluate_sweep(
        tmp_path,
        tmp_path / "output",
        training_seeds=(0,),
        explore=False,
        evaluation_seed=10,
    )
    report = json.loads((output / "comparison.json").read_text())
    assert len(calls) == report["completed_checkpoints"] == 5
    assert report["explore"] is False and report["training_seeds"] == [0]
    assert "Diagnostic capture is disabled" in report["timing_scope"]
    assert "undefined (null) for n=1" in report["statistics_scope"]
    assert report["evaluation_seeds"] == [10, 11, 12]
    assert all(
        not config.evaluation.explore and config.training.seed == 10
        for config, _ in calls
    )
    assert all(
        group["metrics"]["mean_score"]["sample_sd"] is None
        for group in report["groups"]
    )
    assert all(
        row["checkpoint_iteration"] is None and row["explore"] is False
        for row in report["runs"]
    )
    (Path(jobs[0]["directory"]) / "manifest.json").write_text(json.dumps({"seed": 2}))
    with pytest.raises(ValueError, match="Checkpoint training seed differs"):
        evaluate_checkpoints.evaluate_sweep(
            tmp_path, tmp_path / "other", training_seeds=(0,)
        )


def test_cli_forwards_checkpoint_seed_exploration_and_diagnostic_options(
    monkeypatch, tmp_path
):
    seen = {}

    def evaluate(*args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(evaluate_checkpoints, "evaluate_sweep", evaluate)
    evaluate_checkpoints.main(
        [
            "--sweep-directory",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
            "--training-seeds",
            "0",
            "--checkpoint-iteration",
            "100",
            "--no-explore",
            "--diagnostics",
            "--resume",
            "--plan",
        ]
    )
    assert seen["training_seeds"] == (0,)
    assert seen["checkpoint_iteration"] == 100 and seen["explore"] is False
    assert seen["diagnostics"] and seen["resume"] and seen["plan_only"]


def test_resume_reuses_only_complete_matching_results(monkeypatch, tmp_path):
    _sweep(tmp_path, seeds=(0,))
    calls = _fake_evaluation(monkeypatch)
    output = evaluate_checkpoints.evaluate_sweep(
        tmp_path, tmp_path / "output", training_seeds=(0,)
    )
    names = [
        f"{name}_seed0_workers8" for name in evaluate_checkpoints.MAIN_CONFIGURATIONS
    ]
    first = output / "runs" / names[0] / "evaluation.json"
    original = first.read_bytes()
    for name in names[2:]:
        shutil.rmtree(output / "runs" / name)
    (output / "comparison.json").unlink()
    calls.clear()
    evaluate_checkpoints.evaluate_sweep(
        tmp_path, output, training_seeds=(0,), resume=True
    )
    assert len(calls) == 3 and first.read_bytes() == original
    report = json.loads((output / "comparison.json").read_text())
    assert report["completed_checkpoints"] == 5
    before = (output / "evaluation_plan.json").read_bytes()
    with pytest.raises(ValueError, match="matching evaluation plan"):
        evaluate_checkpoints.evaluate_sweep(
            tmp_path, output, training_seeds=(0,), episodes=4, resume=True
        )
    assert (output / "evaluation_plan.json").read_bytes() == before
    calls.clear()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        evaluate_checkpoints.evaluate_sweep(tmp_path, output, training_seeds=(0,))
    assert not calls


def test_resume_rejects_partial_run_and_changed_checkpoint(monkeypatch, tmp_path):
    _sweep(tmp_path, seeds=(0,))
    calls = _fake_evaluation(monkeypatch)
    output = evaluate_checkpoints.evaluate_sweep(
        tmp_path, tmp_path / "output", training_seeds=(0,)
    )
    report = json.loads((output / "comparison.json").read_text())
    result_dir = output / report["runs"][0]["directory"]
    path = result_dir / "evaluation.json"
    original = path.read_bytes()
    path.unlink()
    calls.clear()
    with pytest.raises(ValueError, match="Incomplete evaluation"):
        evaluate_checkpoints.evaluate_sweep(
            tmp_path, output, training_seeds=(0,), resume=True
        )
    assert not calls and not path.exists()
    path.write_bytes(original)
    checkpoint = Path(report["runs"][0]["checkpoint"])
    (checkpoint / "algorithm_state.pkl").write_bytes(b"modified checkpoint")
    with pytest.raises(ValueError, match="Checkpoint differs"):
        evaluate_checkpoints.evaluate_sweep(
            tmp_path, output, training_seeds=(0,), resume=True
        )
    assert not calls


def test_diagnostics_capture_is_scoped_per_checkpoint(monkeypatch, tmp_path):
    _sweep(tmp_path, seeds=(0,))
    _fake_evaluation(monkeypatch)
    instances, original_episode = _fake_diagnostics(monkeypatch)
    output = evaluate_checkpoints.evaluate_sweep(
        tmp_path, tmp_path / "output", training_seeds=(0,), diagnostics=True
    )
    assert len(instances) == 5 and all(item.seeds == [0, 1, 2] for item in instances)
    assert evaluate_checkpoints.Sampler.episode is original_episode
    report = json.loads((output / "comparison.json").read_text())
    assert report["diagnostics"] is True
    assert all((output / row["diagnostics"]).is_file() for row in report["runs"])
    assert (
        "Diagnostic capture and entropy bookkeeping are included"
        in report["timing_scope"]
    )
    assert "offline diagnostic analysis/audits are excluded" in report["timing_scope"]
    assert all(row["timing_scope"] == report["timing_scope"] for row in report["runs"])


@pytest.mark.parametrize(
    "diagnostic",
    [
        {},
        {"episodes": [{"seed": 0}, {"seed": 1}], "summary": {"episode_count": 3}},
        {"episodes": [0, 1, 2], "summary": {"episode_count": 3}},
        {
            "episodes": [{"seed": 0}, {"seed": 1}, {"seed": 2}],
            "summary": {"episode_count": 2},
        },
        {
            "episodes": [{"seed": 0}, {"seed": 1}, {"seed": 2}],
            "summary": {"episode_count": True},
        },
        {
            "episodes": [{"seed": 0}, {"seed": 2}, {"seed": 1}],
            "summary": {"episode_count": 3},
        },
        [],
        "{truncated",
    ],
)
def test_resume_rejects_incomplete_or_mismatched_diagnostics(
    monkeypatch, tmp_path, diagnostic
):
    _sweep(tmp_path, seeds=(0,))
    calls = _fake_evaluation(monkeypatch)
    _fake_diagnostics(monkeypatch)
    output = evaluate_checkpoints.evaluate_sweep(
        tmp_path, tmp_path / "output", training_seeds=(0,), diagnostics=True
    )
    comparison = output / "comparison.json"
    original = comparison.read_bytes()
    row = json.loads(original)["runs"][0]
    path = output / row["diagnostics"]
    invalid = diagnostic if isinstance(diagnostic, str) else json.dumps(diagnostic)
    path.write_text(invalid)
    calls.clear()
    with pytest.raises(ValueError):
        evaluate_checkpoints.evaluate_sweep(
            tmp_path, output, training_seeds=(0,), diagnostics=True, resume=True
        )
    assert not calls
    assert path.read_text() == invalid and comparison.read_bytes() == original


@pytest.mark.parametrize("checkpoint_count", [1, 2])
def test_plot_exports_five_configurations_with_only_defined_sample_sd(
    monkeypatch, tmp_path, checkpoint_count
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    calls = []
    original = Axes.errorbar

    def errorbar(self, *args, **kwargs):
        calls.append(kwargs["yerr"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", errorbar)
    groups = evaluate_checkpoints.summarize(
        [
            {
                "configuration": configuration,
                "episodes": 3,
                "mean_score": 0.4 + index * 0.1,
                "fingerprint_diversity": 0.7 + index * 0.1,
            }
            for configuration in evaluate_checkpoints.MAIN_CONFIGURATIONS
            for index in range(checkpoint_count)
        ]
    )
    evaluate_checkpoints.plot_groups(groups, tmp_path)
    assert (tmp_path / "quality_diversity.png").stat().st_size > 1000
    assert (tmp_path / "quality_diversity.svg").stat().st_size > 1000
    if checkpoint_count == 1:
        assert calls == []
    else:
        assert len(calls) == 2
        assert all(
            len(values) == 5 and all(value > 0 for value in values) for values in calls
        )
