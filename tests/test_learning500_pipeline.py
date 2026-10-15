import csv
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_learning500_evaluations as pipeline
from scripts.run_learning500_evaluations import (
    CONFIGURATIONS,
    audit_training,
    evaluation_stages,
    process_alive,
)


def _training_fixture(tmp_path, iterations=100):
    core_hashes = {"moldr/core.py": "fixture"}
    (tmp_path.parent / "execution_plan.json").write_text(
        json.dumps({"core_sha256": core_hashes})
    )
    jobs = []
    for name in CONFIGURATIONS:
        run = tmp_path / f"{name}_seed0_workers8"
        for iteration in range(100, iterations + 1, 100):
            checkpoint = run / "checkpoints" / f"iteration_{iteration:06d}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "algorithm_state.pkl").write_bytes(
                b"fixture checkpoint marker"
            )
        records = [
            {
                "iteration": index,
                "sampled_steps": 1024,
                "learner_stats": dict.fromkeys(
                    ("total_loss", "policy_loss", "vf_loss", "entropy", "kl"), 0.1
                ),
            }
            for index in range(1, iterations + 1)
        ]
        (run / "measurements.json").write_text(json.dumps(records))
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "iterations": iterations,
                    "weight_absolute_change": 1.0,
                    "model_device": "cuda:0",
                }
            )
        )
        (run / "manifest.json").write_text(
            json.dumps({"seed": 0, "source_sha256": core_hashes})
        )
        (run / "last_checkpoint.json").write_text(json.dumps({"iteration": iterations}))
        jobs.append({"name": run.name, "directory": str(run), "status": "complete"})
    (tmp_path / "sweep.json").write_text(json.dumps({"command": "train", "jobs": jobs}))
    return jobs


def test_audit_requires_every_job_every_iteration_and_real_cuda(tmp_path):
    jobs = _training_fixture(tmp_path)
    report = audit_training(tmp_path, iterations=100)
    assert len(report["jobs"]) == 5
    assert all(job["environment_decisions"] == 102400 for job in report["jobs"])
    run = tmp_path / jobs[0]["name"]
    rows = json.loads((run / "measurements.json").read_text())
    rows.pop(37)
    (run / "measurements.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="consecutive"):
        audit_training(tmp_path, iterations=100)


def test_planned_or_failed_job_does_not_pass_audit(tmp_path):
    jobs = _training_fixture(tmp_path)
    jobs[-1]["status"] = "running"
    (tmp_path / "sweep.json").write_text(json.dumps({"command": "train", "jobs": jobs}))
    with pytest.raises(ValueError, match="not complete"):
        audit_training(tmp_path, iterations=100)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("cpu", "CUDA"),
        ("unchanged", "parameter"),
        ("nan", "Nonfinite"),
        ("weight_nan", "parameter"),
        ("weight_inf", "parameter"),
        ("checkpoint", "Missing checkpoint"),
        ("core_hash", "core changed"),
    ],
)
def test_audit_rejects_false_success_evidence(tmp_path, fault, match):
    jobs = _training_fixture(tmp_path)
    run = tmp_path / jobs[0]["name"]
    if fault in {"cpu", "unchanged", "weight_nan", "weight_inf"}:
        path = run / "summary.json"
        row = json.loads(path.read_text())
        row["model_device" if fault == "cpu" else "weight_absolute_change"] = {
            "cpu": "cpu",
            "unchanged": 0.0,
            "weight_nan": float("nan"),
            "weight_inf": float("inf"),
        }[fault]
        path.write_text(json.dumps(row))
    elif fault == "nan":
        path = run / "measurements.json"
        rows = json.loads(path.read_text())
        rows[-1]["learner_stats"]["total_loss"] = float("nan")
        path.write_text(json.dumps(rows))
    elif fault == "core_hash":
        path = run / "manifest.json"
        row = json.loads(path.read_text())
        row["source_sha256"]["moldr/core.py"] = "changed"
        path.write_text(json.dumps(row))
    else:
        (run / "checkpoints" / "iteration_000100" / "algorithm_state.pkl").unlink()
    with pytest.raises(ValueError, match=match):
        audit_training(tmp_path, iterations=100)


@pytest.mark.parametrize("plan_state", ["missing", "empty"])
def test_audit_requires_nonempty_pinned_execution_plan(tmp_path, plan_state):
    _training_fixture(tmp_path)
    path = tmp_path.parent / "execution_plan.json"
    if plan_state == "missing":
        path.unlink()
    else:
        path.write_text(json.dumps({"core_sha256": {}}))
    with pytest.raises(ValueError, match="[Ee]xecution plan"):
        audit_training(tmp_path, iterations=100)


def test_audit_rejects_duplicate_jobs_before_dictionary_can_hide_them(tmp_path):
    jobs = _training_fixture(tmp_path)
    jobs.append(dict(jobs[0]))
    (tmp_path / "sweep.json").write_text(json.dumps({"command": "train", "jobs": jobs}))
    with pytest.raises(ValueError, match="exactly the five"):
        audit_training(tmp_path, iterations=100)


def test_audit_all_five_500_iteration_records_and_checkpoints(tmp_path):
    _training_fixture(tmp_path, iterations=500)
    report = audit_training(tmp_path)
    assert report["complete"] and len(report["jobs"]) == 5
    for job in report["jobs"]:
        assert job["iterations"] == 500
        assert job["environment_decisions"] == 512000
        assert job["training_core_hashes_match"]
        assert [path.rsplit("/", 1)[1] for path in job["checkpoints"]] == [
            f"iteration_{iteration:06d}" for iteration in (100, 200, 300, 400, 500)
        ]


def test_reused_pid_or_zombie_does_not_count_as_live(monkeypatch):
    monkeypatch.setattr(
        "scripts.run_learning500_evaluations.process_identity", lambda pid: ("S", "200")
    )
    assert process_alive(1234, "200")
    assert not process_alive(1234, "100")
    monkeypatch.setattr(
        "scripts.run_learning500_evaluations.process_identity", lambda pid: ("Z", "200")
    )
    assert not process_alive(1234, "200")


def test_proc_stat_start_token_parses_command_parentheses(monkeypatch):
    fields = ["S", *(str(number) for number in range(4, 23))]
    stat = f"123 (command ) containing ) parentheses) {' '.join(fields)}"
    monkeypatch.setattr(
        pipeline, "Path", lambda path: SimpleNamespace(read_text=lambda: stat)
    )
    assert pipeline.process_identity(123) == ("S", "22")


def test_stages_preserve_shared_seeds_and_distinguish_sampling_from_greedy(tmp_path):
    stages = evaluation_stages(
        tmp_path / "new", tmp_path / "prior", tmp_path / "evaluations"
    )
    assert len(stages) == 7
    assert [stage["episodes"] for stage in stages] == [
        1000,
        10,
        100,
        100,
        100,
        100,
        1000,
    ]
    assert [stage["checkpoint_iteration"] for stage in stages] == [
        500,
        500,
        100,
        200,
        300,
        400,
        13,
    ]
    assert sum(not stage["explore"] for stage in stages) == 1


def test_completed_comparisons_still_use_strict_evaluator_resume(tmp_path, monkeypatch):
    sweep, prior, output = (
        tmp_path / "training",
        tmp_path / "prior",
        tmp_path / "evaluations",
    )
    stages = evaluation_stages(sweep, prior, output)
    for stage in stages:
        directory = pipeline.Path(stage["output"])
        directory.mkdir(parents=True)
        (directory / "comparison.json").write_text('{"previously_complete":true}')
        (directory / "evaluation_plan.json").write_text("{}")
    calls, validations, audits = [], [], []

    def audit(directory):
        audits.append(directory)
        return {"complete": True, "jobs": []}

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline, "audit_training", audit)
    monkeypatch.setattr(
        pipeline, "validate_stage", lambda stage: validations.append(stage["name"])
    )
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    pipeline.main(
        [
            "--sweep-directory",
            str(sweep),
            "--prior-sweep",
            str(prior),
            "--output",
            str(output),
        ]
    )
    assert audits == [sweep]
    assert len(calls) == len(validations) == 7
    for stage, (command, options) in zip(stages, calls, strict=True):
        assert command[1:3] == ["-m", "scripts.evaluate_checkpoints"]
        assert "--resume" in command
        assert "--diagnostics" in command and options["check"] is True
        for flag, value in (
            ("--training-seeds", "0"),
            ("--episodes", str(stage["episodes"])),
            ("--checkpoint-iteration", str(stage["checkpoint_iteration"])),
            ("--evaluation-seed", "0"),
            ("--num-gpus", "1"),
        ):
            assert command[command.index(flag) + 1] == value
        assert ("--no-explore" in command) is (not stage["explore"])
    assert json.loads((output / "pipeline_complete.json").read_text())["complete"]


def test_failed_evaluator_stops_without_training_restart_or_false_completion(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        pipeline, "audit_training", lambda directory: {"complete": True}
    )

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    output = tmp_path / "evaluations"
    output.mkdir()
    (output / "pipeline_complete.json").write_text(
        '{"complete":true,"previous_run":true}'
    )
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.main(
            [
                "--sweep-directory",
                str(tmp_path / "training"),
                "--prior-sweep",
                str(tmp_path / "prior"),
                "--output",
                str(output),
            ]
        )
    assert len(calls) == 1 and calls[0][1:3] == ["-m", "scripts.evaluate_checkpoints"]
    assert not (output / "pipeline_complete.json").exists()


def test_failed_training_audit_starts_no_subprocess(tmp_path, monkeypatch):
    calls = []

    def fail_audit(directory):
        raise ValueError("Training is not complete")

    monkeypatch.setattr(pipeline, "audit_training", fail_audit)
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda *args, **kwargs: calls.append(args)
    )
    output = tmp_path / "evaluations"
    with pytest.raises(ValueError, match="not complete"):
        pipeline.main(
            [
                "--sweep-directory",
                str(tmp_path / "training"),
                "--prior-sweep",
                str(tmp_path / "prior"),
                "--output",
                str(output),
            ]
        )
    assert calls == []
    assert not (output / "pipeline_complete.json").exists()


@pytest.mark.parametrize(
    "failure", ["timeout", "os_error", "empty", "nonzero", "malformed"]
)
def test_gpu_telemetry_failure_keeps_waiting_and_records_unknown_values(
    tmp_path, monkeypatch, failure
):
    (tmp_path / "sweep.json").write_text(
        json.dumps({"jobs": [{"name": "actor_value", "status": "running"}]})
    )
    alive, sleeps, queries = iter((True, False)), [], []
    monkeypatch.setattr(pipeline, "process_alive", lambda pid, token: next(alive))
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            cpu_percent=lambda: 12.5,
            virtual_memory=lambda: SimpleNamespace(used=2**30),
        ),
    )

    def query(command, **kwargs):
        queries.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 10)
        if failure == "os_error":
            raise FileNotFoundError("nvidia-smi fixture unavailable")
        stdout = {"empty": "", "nonzero": "99, 500, 10\n", "malformed": "invalid\n"}[
            failure
        ]
        return subprocess.CompletedProcess(
            command, 1 if failure == "nonzero" else 0, stdout=stdout
        )

    monkeypatch.setattr(pipeline.subprocess, "run", query)
    pipeline.wait_for_training(123, "start", tmp_path, tmp_path, interval=15)
    with (tmp_path / "training_resources.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(queries) == 1 and sleeps == [15]
    row = rows[0]
    assert row["configuration"] == "actor_value"
    assert row["cpu_percent"] == "12.5" and row["ram_used_gib"] == "1.0"
    assert all(
        row[key] == "" for key in ("gpu_percent", "gpu_memory_mib", "gpu_power_w")
    )
    assert row["telemetry_error"]
