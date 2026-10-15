"""Small saved-result fixtures for learning summaries; no Ray/model execution."""

import json
from pathlib import Path

import pytest

from scripts import learning_report


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _record(iteration, *, episodes=2, quality_sum=1.0, seconds=2.0, updates=20):
    return {
        "iteration": iteration,
        "measured": True,
        "seconds": seconds,
        "sampled_steps": 32,
        "chemistry": {
            "updates": updates,
            "valid_molecules": updates,
            "episodes": episodes,
            "quality_sum": quality_sum,
            "requests": updates + 2,
            "computations": updates,
            "hits": 2,
        },
        "learner_stats": {
            "total_loss": -0.2,
            "policy_loss": -0.1,
            "vf_loss": 0.3,
            "entropy": 4.0,
            "kl": 0.02,
            "vf_explained_var": 0.5,
        },
        "timers": {"sample_time_ms": 1000.0, "learn_time_ms": 500.0},
        "peak_gpu_memory_mib": 1500.0 + iteration,
    }


def _sweep(tmp_path, records, *, status="complete", name="actor_value_seed0_workers8"):
    sweep = tmp_path / "sweep"
    run = sweep / name
    _write(run / "measurements.json", records)
    _write(
        sweep / "sweep.json", {"jobs": [{"name": name, "status": status, "diff": {}}]}
    )
    if status == "complete":
        _write(run / "summary.json", {"iterations": len(records)})
    return sweep, run


def _auxiliary_log(run, pairs):
    path = run / "logs" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "training_iteration": iteration,
                    "info": {
                        "learner": {
                            "default_policy": {
                                "learner_stats": {"candidate_value_loss": 999},
                                "model": {"candidate_value_loss": loss},
                            }
                        }
                    },
                }
            )
            + "\n"
            for iteration, loss in pairs
        )
    )
    return path


@pytest.fixture
def no_plots(monkeypatch):
    monkeypatch.setattr(learning_report, "_plots", lambda report, directory: [])


def test_windows_weight_quality_and_throughput_by_counts_and_time(tmp_path, no_plots):
    records = [
        _record(i, episodes=1, quality_sum=1, seconds=1, updates=20)
        for i in range(1, 4)
    ]
    records.extend(
        [
            _record(4, episodes=1, quality_sum=1, seconds=1, updates=10),
            _record(5, episodes=9, quality_sum=0, seconds=9, updates=90),
        ]
    )
    sweep, run = _sweep(tmp_path, records)
    log = _auxiliary_log(run, [(4, 0.25), (5, 0.75)])
    original_measurements = (run / "measurements.json").read_bytes()
    original_log = log.read_bytes()
    destination = learning_report.build_report(
        sweep, tmp_path / "report", expected_iterations=5, require_complete=True
    )
    report = json.loads(destination.with_name("learning_report.json").read_text())
    trial = report["runs"][0]
    whole, steady, last = (
        trial["windows"][key]
        for key in (
            "whole",
            "steady_after_first_three",
            "last_fifty",
        )
    )
    assert whole["mean_final_training_qed"] == pytest.approx(4 / 13)
    assert steady["mean_final_training_qed"] == pytest.approx(0.1)
    assert whole["molecule_updates_per_second"] == pytest.approx(160 / 13)
    assert steady["molecule_updates_per_second"] == 10
    assert steady["sampled_steps_per_second"] == pytest.approx(64 / 10)
    assert last["iterations"] == 5 and not last["full_window"]
    assert whole["gpu_peak_mib"] == 1505
    assert whole["metrics"]["candidate_value_loss"] == {
        "mean": 0.5,
        "observed_iterations": 2,
    }
    assert trial["curve"][0]["candidate_value_loss"] is None
    assert trial["curve"][3]["candidate_value_loss"] == 0.25
    assert trial["training_seed_n"] == 1 and trial["training_seed_sample_sd"] is None
    assert report["evaluation"]["state"] == "not_provided"
    assert report["evaluation"]["runs"] == []
    assert report["state"] == "complete"
    assert (run / "measurements.json").read_bytes() == original_measurements
    assert log.read_bytes() == original_log
    text = destination.read_text()
    assert "not a final-checkpoint evaluation" in text
    assert "randomly initialized, untrained encoder" in text
    assert "No completed checkpoint evaluation" in text


def test_last_fifty_uses_observed_tail_and_steady_excludes_first_three(tmp_path):
    records = [_record(i, quality_sum=float(i)) for i in range(1, 61)]
    sweep, _ = _sweep(tmp_path, records)
    run = learning_report.read_training_run(
        {"name": "actor_value_seed0_workers8", "status": "complete"}, sweep, 60
    )
    assert run["windows"]["last_fifty"]["first_iteration"] == 11
    assert run["windows"]["last_fifty"]["last_iteration"] == 60
    assert run["windows"]["last_fifty"]["full_window"]
    assert run["windows"]["last_fifty"]["mean_final_training_qed"] == pytest.approx(
        sum(range(11, 61)) / 100
    )
    assert run["windows"]["steady_after_first_three"]["iterations"] == 57
    assert run["curve"][-1]["elapsed_training_seconds"] == 120


def test_zero_episodes_missing_aux_and_undefined_variance_stay_null(tmp_path, no_plots):
    record = _record(1, episodes=0, quality_sum=0)
    record["learner_stats"]["vf_explained_var"] = float("nan")
    record["peak_gpu_memory_mib"] = None
    record["chemistry"].pop("updates")
    sweep, _ = _sweep(tmp_path, [record])
    destination = learning_report.build_report(
        sweep, tmp_path / "report", expected_iterations=1
    )
    contents = destination.with_name("learning_report.json").read_text()
    trial = json.loads(contents)["runs"][0]
    whole = trial["windows"]["whole"]
    assert whole["episodes"] == 0
    assert whole["completed_molecules_per_second"] == 0
    assert whole["mean_final_training_qed"] is None
    assert whole["molecule_updates_per_second"] is None
    assert whole["gpu_peak_mib"] is None
    assert whole["metrics"]["candidate_value_loss"] == {
        "mean": None,
        "observed_iterations": 0,
    }
    assert whole["metrics"]["vf_explained_var"]["mean"] is None
    assert "NaN" not in contents


def test_running_and_planned_jobs_are_snapshots_even_with_all_records(
    tmp_path, no_plots
):
    sweep, _ = _sweep(tmp_path, [_record(1), _record(2)], status="running")
    _write(
        sweep / "sweep.json",
        {
            "jobs": [
                {"name": "actor_value_seed0_workers8", "status": "running"},
                {"name": "mpnn_frozen_seed0_workers8", "status": "planned"},
            ]
        },
    )
    destination = learning_report.build_report(
        sweep, tmp_path / "report", expected_iterations=2
    )
    report = json.loads(destination.with_name("learning_report.json").read_text())
    assert report["state"] == "snapshot" and report["completed_runs"] == 0
    assert [row["observed_iterations"] for row in report["runs"]] == [2, 0]
    assert report["runs"][1]["windows"]["whole"]["mean_final_training_qed"] is None
    assert "SNAPSHOT" in destination.read_text()
    with pytest.raises(ValueError, match="Complete training sweep required"):
        learning_report.build_report(
            sweep, tmp_path / "strict", expected_iterations=2, require_complete=True
        )
    assert not (tmp_path / "strict").exists()


def test_completion_requires_all_500_distinct_iterations(tmp_path, no_plots):
    sweep, run = _sweep(tmp_path, [_record(i) for i in range(1, 501)])
    destination = learning_report.build_report(
        sweep, tmp_path / "report", require_complete=True
    )
    report = json.loads(destination.with_name("learning_report.json").read_text())
    assert report["runs"][0]["windows"]["last_fifty"]["first_iteration"] == 451
    records = [_record(i) for i in range(1, 501) if i != 250]
    records.append(_record(501))  # Still 500 records, but not the requested 1..500.
    _write(run / "measurements.json", records)
    with pytest.raises(ValueError, match="exactly iterations 1..500"):
        learning_report.build_report(sweep, tmp_path / "bad")


@pytest.mark.parametrize(
    "iterations, message", [([1, 1], "Duplicate"), ([0, 1], "Invalid")]
)
def test_invalid_iteration_records_are_rejected(tmp_path, iterations, message):
    sweep, _ = _sweep(tmp_path, [_record(i) for i in iterations])
    with pytest.raises(ValueError, match=message):
        learning_report.read_training_run(
            {"name": "actor_value_seed0_workers8", "status": "complete"}, sweep, 2
        )


def test_live_partial_measurement_and_log_writes_are_explicit(tmp_path, no_plots):
    sweep, run = _sweep(tmp_path, [_record(1)], status="running")
    path = _auxiliary_log(run, [(1, 0.0)])
    with path.open("a") as stream:
        stream.write('{"training_iteration":2,"info":')
    issues = []
    losses, _ = learning_report.read_auxiliary_losses(path, issues, live=True)
    assert losses == {1: 0.0} and "incomplete trailing" in issues[0]
    with pytest.raises(ValueError, match="Invalid log JSON"):
        learning_report.read_auxiliary_losses(path, [], live=False)
    (run / "measurements.json").write_text('[{"iteration":')
    destination = learning_report.build_report(sweep, tmp_path / "report")
    report = json.loads(destination.with_name("learning_report.json").read_text())
    assert report["runs"][0]["observed_iterations"] == 0
    assert "being written" in report["runs"][0]["issues"][0]


def test_bad_middle_log_line_is_not_silently_skipped(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{broken}\n{"training_iteration":2}\n')
    with pytest.raises(ValueError, match="result.json:1"):
        learning_report.read_auxiliary_losses(path, [], live=True)


def test_evaluations_read_modes_budgets_and_pending_iterations_without_zero_fill(
    tmp_path,
):
    evaluation = tmp_path / "evaluations"
    for iteration, explore, episodes in (
        (100, True, 100),
        (500, True, 1000),
        (500, False, 10),
    ):
        directory = evaluation / f"iteration{iteration}_{explore}"
        _write(
            directory / "evaluation_plan.json",
            {
                "checkpoint_iteration": iteration,
                "explore": explore,
                "total_checkpoints": 5,
                "episodes_per_checkpoint": episodes,
                "evaluation_seeds": list(range(episodes)),
                "sweep_directory": "/saved/training/sweep",
            },
        )
        _write(
            directory / "runs.json",
            [
                {
                    "name": "actor_value_seed0_workers8",
                    "configuration": "actor_value",
                    "training_seed": 0,
                    "checkpoint": f"/saved/checkpoints/iteration_{iteration:06}",
                    "directory": "runs/actor_value_seed0_workers8",
                    "episodes": episodes,
                    "mean_score": 0.7,
                    "unique_fraction": 0.8,
                    "fingerprint_diversity": 0.2,
                }
            ],
        )
    _write(
        evaluation / "iteration200" / "evaluation_plan.json",
        {
            "checkpoint_iteration": 200,
            "explore": True,
            "total_checkpoints": 5,
        },
    )
    report = learning_report.read_evaluations(evaluation)
    assert report["state"] == "snapshot"
    assert len(report["runs"]) == 3 and len(report["datasets"]) == 4
    assert {row["checkpoint_iteration"] for row in report["runs"]} == {100, 500}
    assert {(row["explore"], row["episodes"]) for row in report["runs"]} == {
        (True, 100),
        (True, 1000),
        (False, 10),
    }
    assert not any(row["mean_score"] == 0 for row in report["runs"])


def test_evaluation_comparison_fallback_infers_checkpoint_iteration_and_unknown_mode(
    tmp_path,
):
    _write(
        tmp_path / "comparison.json",
        {
            "total_checkpoints": 1,
            "runs": [
                {
                    "name": "mol2vec_actor_seed0_workers8",
                    "checkpoint_iteration": None,
                    "checkpoint": "/old/checkpoints/iteration_000013",
                    "episodes": 1000,
                    "mean_score": None,
                }
            ],
        },
    )
    report = learning_report.read_evaluations(tmp_path)
    assert report["state"] == "complete"
    assert report["runs"][0]["checkpoint_iteration"] == 13
    assert report["runs"][0]["explore"] is None
    assert report["runs"][0]["mean_score"] is None


def test_cli_and_html_escape_saved_run_names(tmp_path, no_plots, capsys):
    sweep, _ = _sweep(tmp_path, [_record(1)], name="actor_<script>_seed0_workers8")
    output = tmp_path / "report"
    learning_report.main(
        [
            "--sweep-directory",
            str(sweep),
            "--output",
            str(output),
            "--expected-iterations",
            "1",
            "--require-complete",
        ]
    )
    assert json.loads(capsys.readouterr().out)["report"] == str(output / "index.html")
    assert "actor_&lt;script&gt;_seed0_workers8" in (output / "index.html").read_text()


def test_fixture_figures_export_png_svg_and_local_html(tmp_path):
    sweep, run = _sweep(tmp_path, [_record(1), _record(2)])
    _auxiliary_log(run, [(1, 0.3), (2, 0.2)])
    evaluation = tmp_path / "evaluations"
    _write(
        evaluation / "comparison.json",
        {
            "total_checkpoints": 1,
            "explore": True,
            "checkpoint_iteration": 2,
            "runs": [
                {
                    "name": "actor_value_seed0_workers8",
                    "configuration": "actor_value",
                    "training_seed": 0,
                    "episodes": 10,
                    "mean_score": 0.6,
                }
            ],
        },
    )
    destination = learning_report.build_report(
        sweep,
        tmp_path / "report",
        expected_iterations=2,
        evaluation_directory=evaluation,
    )
    report = json.loads(destination.with_name("learning_report.json").read_text())
    assert report["plots"] == [
        "learning_curves",
        "optimization_diagnostics",
        "checkpoint_evaluations",
    ]
    for name in report["plots"]:
        assert destination.with_name(f"{name}.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in destination.with_name(f"{name}.svg").read_text()
        assert f"src='{name}.png'" in destination.read_text()
    assert Path(report["runs"][0]["sources"]["full_log"]["path"]).is_file()
