"""Seed summaries preserve worker groups and undefined completion quality."""

import json

import pytest

from scripts import experiment_report


def test_group_statistics_keep_workers_and_use_sample_deviation():
    rows = [
        {"name": "actor_seed0_workers8", "molecule_updates_per_second": 10.0},
        {"name": "actor_seed1_workers8", "molecule_updates_per_second": 20.0},
        {"name": "actor_seed2_workers8", "molecule_updates_per_second": 30.0},
        {"name": "actor_seed0_workers1", "molecule_updates_per_second": 4.0},
    ]
    groups = experiment_report.group_rows(rows, [row["name"] for row in rows])
    assert [group["name"] for group in groups] == ["actor_workers8", "actor_workers1"]
    assert groups[0]["n"] == 3
    assert groups[0]["metrics"]["molecule_updates_per_second"] == {
        "mean": 20.0,
        "sample_sd": 10.0,
        "n": 3,
    }
    assert groups[1]["metrics"]["molecule_updates_per_second"]["sample_sd"] is None
    assert groups[0]["metrics"]["mean_final_score"] == {
        "mean": None,
        "sample_sd": None,
        "n": 0,
    }


def test_report_handles_no_completed_episodes_and_marks_partial_jobs(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(experiment_report, "_plots", lambda groups, directory: [])
    run = tmp_path / "actor_seed0_workers8"
    run.mkdir()
    record = {
        "measured": True,
        "seconds": 2.0,
        "chemistry": {
            "updates": 10,
            "valid_molecules": 10,
            "episodes": 0,
            "quality_sum": 0.0,
            "requests": 10,
            "computations": 10,
            "hits": 0,
            "sampled_unique_fraction": None,
        },
        "timers": {"sample_time_ms": 1000.0, "learn_time_ms": 500.0},
        "peak_gpu_memory_mib": None,
    }
    (run / "measurements.json").write_text(json.dumps([record]))
    jobs = [
        {
            "name": "actor_seed0_workers8",
            "status": "complete",
            "directory": str(run),
            "diff": {},
        },
        {"name": "actor_seed1_workers8", "status": "running"},
        {"name": "actor_seed2_workers8", "status": "planned"},
    ]
    (tmp_path / "sweep.json").write_text(json.dumps({"jobs": jobs}))
    output = experiment_report.build_report(tmp_path)
    rows = json.loads((tmp_path / "comparison.json").read_text())
    grouped = json.loads((tmp_path / "grouped_comparison.json").read_text())
    assert rows[0]["mean_final_score"] is None
    assert rows[0]["mean_sampled_unique_fraction"] is None
    assert rows[0]["mean_smoothed_sample_time_ms"] == pytest.approx(1000)
    assert rows[0]["molecule_updates_per_second"] == pytest.approx(5)
    assert rows[0]["completed_valid_molecules"] == 0
    assert rows[0]["completed_valid_molecules_per_second"] == 0
    assert grouped["completed_jobs"] == 1 and grouped["total_jobs"] == 3
    assert grouped["groups"][0]["n"] == 1 and grouped["groups"][0]["planned_n"] == 3
    assert "Completed: 1 / 3 jobs" in output.read_text()
    assert "NaN" not in (tmp_path / "grouped_comparison.json").read_text()


def test_completed_molecule_rate_uses_measured_episode_totals(tmp_path):
    records = [
        {
            "measured": False,
            "seconds": 100,
            "chemistry": {"episodes": 100},
            "timers": {},
        },
        {"measured": True, "seconds": 2, "chemistry": {"episodes": 1}, "timers": {}},
        {"measured": True, "seconds": 3, "chemistry": {"episodes": 4}, "timers": {}},
    ]
    (tmp_path / "measurements.json").write_text(json.dumps(records))
    row = experiment_report._trial_row(
        {"name": "actor_seed0_workers8", "directory": str(tmp_path), "diff": {}}
    )
    assert row["completed_valid_molecules"] == 5
    assert row["completed_valid_molecules_per_second"] == 1
    grouped = experiment_report.group_rows([row])[0]
    assert grouped["metrics"]["completed_valid_molecules_per_second"] == {
        "mean": 1.0,
        "sample_sd": None,
        "n": 1,
    }
