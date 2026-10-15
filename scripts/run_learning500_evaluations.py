"""Audit the five 500-iteration jobs, then evaluate their checkpoints serially.

The optional process wait uses a Linux process start token, not a stale lockfile.
It also records lightweight resource telemetry while that training process lives.
No training, architecture changes, or automatic training restarts occur here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CONFIGURATIONS = (
    "mol2vec_actor", "actor_value", "auxiliary_only", "mpnn_frozen", "mpnn_trainable",
)


def process_identity(pid):
    """Return (state, start ticks) or None when the process no longer exists."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(") ", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return fields[0], fields[19]


def process_alive(pid, start_token):
    identity = process_identity(pid)
    return identity is not None and identity[0] not in {"Z", "X"} and identity[1] == str(start_token)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def audit_training(sweep_directory, *, iterations=500, batch_size=1024):
    """Require actual complete records/checkpoints before allowing evaluation."""
    root = Path(sweep_directory).resolve()
    sweep = json.loads((root / "sweep.json").read_text())
    expected = [f"{name}_seed0_workers8" for name in CONFIGURATIONS]
    by_name = {job["name"]: job for job in sweep["jobs"]}
    if sweep.get("command") != "train" or len(sweep["jobs"]) != len(expected) or set(by_name) != set(expected):
        raise ValueError("Expected exactly the five seed-0 training jobs")
    execution_plan = root.parent / "execution_plan.json"
    if not execution_plan.is_file():
        raise ValueError("Missing execution plan with pinned training core hashes")
    expected_core = json.loads(execution_plan.read_text())["core_sha256"]
    if not expected_core:
        raise ValueError("Execution plan has no pinned training core hashes")
    audited = []
    for name in expected:
        job = by_name[name]
        if job["status"] != "complete":
            raise ValueError(f"Training is not complete: {name}: {job['status']}")
        directory = Path(job["directory"]).resolve()
        records = json.loads((directory / "measurements.json").read_text())
        if [row["iteration"] for row in records] != list(range(1, iterations + 1)):
            raise ValueError(f"Expected exactly {iterations} consecutive iterations: {name}")
        if any(row["sampled_steps"] != batch_size for row in records):
            raise ValueError(f"Unexpected training batch size: {name}")
        for row in records:
            for key in ("total_loss", "policy_loss", "vf_loss", "entropy", "kl"):
                value = row["learner_stats"][key]
                if value is None or not math.isfinite(value):
                    raise ValueError(f"Nonfinite learner statistic: {name}: {key}")
        summary = json.loads((directory / "summary.json").read_text())
        change = summary["weight_absolute_change"]
        if summary["iterations"] != iterations or not math.isfinite(change) or change <= 0:
            raise ValueError(f"Missing actual parameter updates: {name}")
        if not summary["model_device"].startswith("cuda"):
            raise ValueError(f"Expected the requested real CUDA learner: {name}")
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["seed"] != 0:
            raise ValueError(f"Wrong training seed: {name}")
        if any(manifest["source_sha256"].get(path) != digest for path, digest in expected_core.items()):
            raise ValueError(f"Training core changed during the comparison: {name}")
        last = json.loads((directory / "last_checkpoint.json").read_text())
        if last["iteration"] != iterations:
            raise ValueError(f"Final checkpoint is not iteration {iterations}: {name}")
        checkpoints = []
        for index in range(100, iterations + 1, 100):
            path = directory / "checkpoints" / f"iteration_{index:06d}"
            if not (path / "algorithm_state.pkl").is_file():
                raise ValueError(f"Missing checkpoint: {path}")
            checkpoints.append(str(path))
        audited.append({"name": name, "directory": str(directory), "iterations": iterations,
                        "environment_decisions": sum(row["sampled_steps"] for row in records),
                        "model_device": summary["model_device"],
                        "training_core_hashes_match": bool(expected_core),
                        "weight_absolute_change": summary["weight_absolute_change"],
                        "checkpoints": checkpoints})
    return {"complete": True, "training_seed": 0, "jobs": audited}


def wait_for_training(pid, start_token, sweep_directory, output, interval=15):
    import psutil

    path = Path(output) / "training_resources.csv"
    fields = ["utc_time", "configuration", "cpu_percent", "ram_used_gib", "gpu_percent", "gpu_memory_mib", "gpu_power_w", "telemetry_error"]
    psutil.cpu_percent()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        while process_alive(pid, start_token):
            configuration = "unknown"
            try:
                jobs = json.loads((Path(sweep_directory) / "sweep.json").read_text())["jobs"]
                configuration = next((job["name"] for job in jobs if job["status"] == "running"), "between_jobs")
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            row = {"utc_time": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
                   "cpu_percent": psutil.cpu_percent(), "ram_used_gib": psutil.virtual_memory().used / 2**30}
            try:
                query = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10, check=False,
                )
                lines = query.stdout.splitlines()
                values = lines[0].split(",") if lines else []
                if query.returncode == 0 and len(values) == 3:
                    row.update(dict(zip(("gpu_percent", "gpu_memory_mib", "gpu_power_w"), (value.strip() for value in values))))
                else:
                    row["telemetry_error"] = f"nvidia-smi exit={query.returncode}; missing or malformed GPU values"
            except (OSError, subprocess.TimeoutExpired) as error:
                row["telemetry_error"] = f"{type(error).__name__}: {error}"
            writer.writerow(row)
            stream.flush()
            time.sleep(interval)


def evaluation_stages(sweep_directory, prior_sweep, output):
    stages = [
        ("iteration_500_stochastic", sweep_directory, 500, 1000, True),
        ("iteration_500_greedy", sweep_directory, 500, 10, False),
        *[(f"iteration_{iteration}_stochastic", sweep_directory, iteration, 100, True)
          for iteration in (100, 200, 300, 400)],
        ("iteration_13_stochastic", prior_sweep, 13, 1000, True),
    ]
    return [{"name": name, "sweep_directory": str(Path(sweep).resolve()),
             "checkpoint_iteration": iteration, "episodes": episodes, "explore": explore,
             "output": str(Path(output).resolve() / name)}
            for name, sweep, iteration, episodes, explore in stages]


def validate_stage(stage):
    report = json.loads((Path(stage["output"]) / "comparison.json").read_text())
    expected_names = {f"{name}_seed0_workers8" for name in CONFIGURATIONS}
    expected_seeds = list(range(stage["episodes"]))
    if (report["completed_checkpoints"] != 5 or report["total_checkpoints"] != 5
            or report["episodes_per_checkpoint"] != stage["episodes"]
            or report["evaluation_seeds"] != expected_seeds
            or report["explore"] != stage["explore"] or report["training"] is not False
            or report.get("diagnostics") is not True
            or report.get("checkpoint_iteration") != stage["checkpoint_iteration"]
            or report.get("training_seeds") != [0]
            or {row["name"] for row in report["runs"]} != expected_names
            or len(report["runs"]) != 5):
        raise ValueError(f"Existing evaluation differs from the planned stage: {stage['name']}")
    sweep = json.loads((Path(stage["sweep_directory"]) / "sweep.json").read_text())
    sources = {job["name"]: Path(job["directory"]).resolve() for job in sweep["jobs"] if job["name"] in expected_names}
    for row in report["runs"]:
        checkpoint = sources[row["name"]] / "checkpoints" / f"iteration_{stage['checkpoint_iteration']:06d}"
        if row["training_seed"] != 0 or row["episodes"] != stage["episodes"] or Path(row["checkpoint"]).resolve() != checkpoint:
            raise ValueError(f"Evaluation checkpoint or seed differs: {row['name']}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-directory", required=True, type=Path)
    parser.add_argument("--prior-sweep", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait-training-pid", type=int)
    parser.add_argument("--training-start-token")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pipeline_complete.json").unlink(missing_ok=True)
    if args.wait_training_pid is not None:
        if args.training_start_token is None:
            parser.error("--wait-training-pid requires its --training-start-token")
        wait_for_training(args.wait_training_pid, args.training_start_token, args.sweep_directory, args.output)
    audit = audit_training(args.sweep_directory)
    _write(args.output / "training_audit.json", audit)
    stages = evaluation_stages(args.sweep_directory, args.prior_sweep, args.output)
    _write(args.output / "pipeline_plan.json", stages)
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MPLCONFIGDIR="/tmp/moldr_learning500_evaluation_matplotlib")
    for stage in stages:
        command = [sys.executable, "-m", "scripts.evaluate_checkpoints", "--sweep-directory", stage["sweep_directory"],
                   "--output", stage["output"], "--training-seeds", "0", "--checkpoint-iteration", str(stage["checkpoint_iteration"]),
                   "--episodes", str(stage["episodes"]), "--evaluation-seed", "0", "--num-gpus", "1", "--diagnostics"]
        if (Path(stage["output"]) / "evaluation_plan.json").is_file():
            command.append("--resume")
        if not stage["explore"]:
            command.append("--no-explore")
        print(json.dumps({"stage": stage["name"], "status": "starting", "command": command}), flush=True)
        subprocess.run(command, check=True, env=environment)
        validate_stage(stage)
        print(json.dumps({"stage": stage["name"], "status": "complete"}), flush=True)
    _write(args.output / "pipeline_complete.json", {"complete": True, "stages": stages, "training_audit": audit})


if __name__ == "__main__":
    main()
