"""Shared E2E/legacy training, evaluation, sampling, profiling and sweeps."""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from moldr.experiment import config_diff, load_config
from moldr.factory import make_environment, make_ppo_config, resolve_config
from moldr.provenance import (
    checkpoint_path,
    load_saved_config,
    prepare_restored_config,
    save_run,
    sha256_file,
    write_json,
)


def parser(default_command=None):
    result = argparse.ArgumentParser(description=__doc__)
    if default_command is None:
        result.add_argument("command", choices=["train", "evaluate", "sample", "profile", "sweep"])
    result.add_argument("--config", type=Path)
    result.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    result.add_argument("--checkpoint")
    result.add_argument("--mode", choices=["e2e", "legacy"])
    result.add_argument("--objective")
    result.add_argument("--epochs", "--iterations", dest="iterations", type=int)
    result.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int)
    result.add_argument("--num_gpus", "--num-gpus", dest="num_gpus", type=float)
    result.add_argument("--num-cpus", type=int)
    result.add_argument("--train-batch-size", type=int)
    result.add_argument("--minibatch-size", type=int)
    result.add_argument("--num-epochs", type=int)
    result.add_argument("--length", type=int)
    result.add_argument("--seed", type=int)
    result.add_argument("--logger", choices=["local", "tensorboard", "wandb"])
    result.add_argument("--project_name", "--project-name", dest="project_name")
    result.add_argument("--output", type=Path)
    result.add_argument("--episodes", type=int)
    result.add_argument("--selector", choices=["actor", "value_lookahead"])
    result.add_argument("--random", action="store_true", default=None)
    result.add_argument("--trace", action="store_true", default=None)
    result.add_argument("--dataset", choices=["GuacaMol", "guacamol", "ZINC", "zinc"])
    result.add_argument("--minsup", type=int)
    result.add_argument("--jt", action="store_true", default=None)
    result.add_argument("--sweep", type=Path)
    result.add_argument("--dry-run", action="store_true")
    return result


def explicit_values(args):
    mapping = {"mode": "environment.mode", "objective": "scoring.objective", "iterations": "training.iterations",
               "num_workers": "training.num_workers", "num_gpus": "training.num_gpus", "num_cpus": "training.num_cpus",
               "train_batch_size": "training.train_batch_size", "minibatch_size": "training.minibatch_size",
               "num_epochs": "training.num_epochs", "length": "environment.atom_limit", "seed": "training.seed",
               "logger": "output.logger", "project_name": "output.project_name", "episodes": "evaluation.episodes",
               "selector": "evaluation.selector", "random": "evaluation.random", "trace": "profiling.trace"}
    values = {path: getattr(args, key) for key, path in mapping.items() if getattr(args, key) is not None}
    if args.output is not None:
        values["output.directory"] = str(args.output)
    if args.checkpoint is not None:
        values["output.checkpoint"] = args.checkpoint
    if args.dataset is not None or args.minsup is not None or args.jt:
        dataset = (args.dataset or "guacamol").lower()
        if args.jt and dataset != "zinc":
            raise ValueError("--jt is supported only with --dataset ZINC")
        if args.minsup is not None and args.minsup <= 0:
            raise ValueError("--minsup must be positive")
        support = args.minsup or 10000
        filename = "gspan_jt.datavocab.csv" if dataset == "zinc" and args.jt else f"{'guacamol_' if dataset == 'guacamol' else ''}s{support}_l2.csv"
        values["fragments.path"] = f"data/building_blocks/{dataset}/{filename}"
    return values


def _init_ray(config):
    import ray
    import torch
    from rdkit import RDLogger

    torch.set_num_threads(config.training.torch_num_threads)
    RDLogger.DisableLog("rdApp.*")
    if config.training.num_gpus and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no working CUDA device is visible")
    ray.init(num_cpus=config.training.num_cpus or max(2, config.training.num_workers + 2),
             num_gpus=config.training.num_gpus, include_dashboard=False, log_to_driver=False,
             object_store_memory=512 * 2**20,
             runtime_env={"worker_process_setup_hook": _worker_setup,
                          "env_vars": {"OMP_NUM_THREADS": str(config.training.torch_num_threads),
                                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}})


def _worker_setup():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def _directory(config, command):
    stamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d_%H-%M-%S-%f")
    return Path(config.output.directory) / (config.output.run_name or f"{command}_{stamp}_seed{config.training.seed}")


def _save_checkpoint(algo, directory, iteration):
    saved = algo.save(str(directory / "checkpoints" / f"iteration_{iteration:06d}"))
    path = Path(saved.checkpoint.path).resolve()
    write_json(directory / "last_checkpoint.json", {"path": str(path.relative_to(directory)), "iteration": iteration})
    return str(path)


def run_training(config, *, profile=False):
    import numpy as np
    import ray
    import torch
    from ray.tune.logger import CSVLogger, JsonLogger, TBXLogger, UnifiedLogger

    from moldr.metrics import MoleculeMetrics, collect_metrics, metric_delta

    if config.profiling.trace and not profile:
        raise ValueError("Trace capture uses the profile command; run it separately from training speed measurements")
    config = prepare_restored_config(config)
    config, blocks, catalog = resolve_config(config)
    if config.training.num_gpus and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no working CUDA device is visible")
    directory = save_run(_directory(config, "profile" if profile else "train"), config, blocks=blocks, catalog=catalog)
    (directory / "logs").mkdir()
    ppo = make_ppo_config(config).callbacks(MoleculeMetrics)
    loggers = (JsonLogger, CSVLogger, TBXLogger) if config.output.logger == "tensorboard" else (JsonLogger, CSVLogger)
    ppo.debugging(logger_creator=lambda c: UnifiedLogger(c, str(directory / "logs"), loggers=loggers))
    wb = None
    if config.output.logger == "wandb":
        from ray.air.integrations.wandb import setup_wandb

        wb = setup_wandb(project=config.output.project_name, name=directory.name, config=config.to_dict())
    _init_ray(config)
    algo = None
    records = []
    try:
        algo = ppo.build()
        if config.output.checkpoint:
            algo.restore(checkpoint_path(config.output.checkpoint))
        policy = algo.get_policy()
        initial = {k: v.detach().cpu().clone() for k, v in policy.model.state_dict().items() if torch.is_floating_point(v)}
        if config.training.num_gpus:
            torch.cuda.reset_peak_memory_stats()
        iterations = (config.profiling.warmup_iterations + config.profiling.measured_iterations) if profile else config.training.iterations
        measured_start = None
        for index in range(iterations):
            measuring = not profile or index >= config.profiling.warmup_iterations
            before = collect_metrics(algo)
            if measuring and measured_start is None:
                measured_start = time.perf_counter()
            if config.training.num_gpus:
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = algo.train()
            if config.training.num_gpus:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if profile and result["num_env_steps_sampled_this_iter"] != config.training.train_batch_size:
                raise RuntimeError("Profiling requires an exact common train batch; configure a fragment length that divides it")
            stats = result["info"]["learner"]["default_policy"]["learner_stats"]
            diagnostics = getattr(policy.model, "kl_diagnostics", None)
            kl_diagnostics = diagnostics() if callable(diagnostics) else None
            if not all(np.isfinite(stats[key]) for key in ("total_loss", "policy_loss", "vf_loss", "entropy", "kl")):
                write_json(directory / "numerical_failure.json", {
                    "iteration": index + 1,
                    "learner_stats": {key: repr(value) if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
                                      for key, value in stats.items()},
                    "kl_diagnostics": kl_diagnostics,
                })
                raise RuntimeError(f"Nonfinite learner statistics: {stats}")
            # Explained variance is undefined for constant return targets.
            # Preserve that as JSON null after verifying the actual losses.
            stats = {key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
                     for key, value in stats.items()}
            if not all(torch.isfinite(p).all() for p in policy.model.parameters()):
                raise RuntimeError("Nonfinite policy parameters")
            after = collect_metrics(algo)
            record = {"iteration": index + 1, "measured": measuring, "seconds": elapsed,
                      "sampled_steps": result["num_env_steps_sampled_this_iter"],
                      "learner_stats": stats, "timers": result.get("timers", {}),
                      "kl_diagnostics": kl_diagnostics,
                      "chemistry": metric_delta(before, after, elapsed),
                      "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20 if config.training.num_gpus else None}
            records.append(record)
            write_json(directory / "measurements.json", records)
            if wb is not None:
                wb.log({"loss": stats["total_loss"], "seconds": elapsed}, step=index + 1)
            print(json.dumps({"run": directory.name, "iteration": index + 1, "seconds": elapsed, "loss": stats["total_loss"]}), flush=True)
            if not profile and (index + 1) % config.output.checkpoint_every == 0:
                _save_checkpoint(algo, directory, index + 1)
            if profile and measuring and config.profiling.time_budget_seconds and time.perf_counter() - measured_start >= config.profiling.time_budget_seconds:
                break
        change = sum((v.detach().cpu() - initial[k]).abs().sum().item() for k, v in policy.model.state_dict().items() if k in initial)
        if records and change == 0:
            raise RuntimeError("PPO did not update any parameters")
        summary = {"model_device": str(next(policy.model.parameters()).device), "weight_absolute_change": change,
                   "iterations": len(records), "trace_in_speed_measurements": False}
        _save_checkpoint(algo, directory, len(records))
        if profile and config.profiling.trace:
            _trace_iteration(algo, config, directory)
        write_json(directory / "summary.json", summary)
        print(json.dumps({"output": str(directory), **summary}), flush=True)
        return directory
    finally:
        if algo is not None:
            algo.stop()
        if wb is not None:
            wb.finish()
        ray.shutdown()


def _trace_iteration(algo, config, directory):
    import torch

    from moldr.profiling import (
        disable_snapshot,
        disable_workers,
        enable,
        enable_workers,
        span,
    )
    from scripts.profile_analysis import analyze_trace, merge_trace
    from scripts.profile_viewer import COMPACT_EVENT_THRESHOLD, build_viewer

    epoch = time.time_ns()
    enable_workers(algo, epoch, max_events=config.profiling.max_events)
    enable(start_epoch_ns=epoch, process_name="driver", max_events=config.profiling.max_events,
           torch_ranges=True, include_rllib=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if config.training.num_gpus:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(activities=activities) as profiler, span("trace_iteration", "iteration"):
            algo.train()
        profiler.export_chrome_trace(str(directory / "torch_trace.json"))
    finally:
        logical = disable_snapshot()
        workers = disable_workers(algo)
    logical_traces = [logical, *workers]
    # The existing converter owns Torch/Chrome clock normalization.
    custom_events = [event for snapshot in logical_traces for event in snapshot["traceEvents"]]
    trace = merge_trace(custom_events, json.loads((directory / "torch_trace.json").read_text()), epoch)
    trace["moldr_profile_metadata"] = {"trace_separate_from_speed_measurement": True,
                                        "dropped_events": sum(snapshot.get("dropped_events", 0) for snapshot in logical_traces)}
    if trace["moldr_profile_metadata"]["dropped_events"]:
        raise RuntimeError("Trace capacity exceeded; increase profiling.max_events")
    write_json(directory / "trace.json", trace)
    write_json(directory / "trace_analysis.json", analyze_trace(trace))
    build_viewer(
        directory / "trace.json", directory / "timeline.html", directory / "trace_analysis.json",
        compact=len(trace["traceEvents"]) >= COMPACT_EVENT_THRESHOLD,
    )


def run_evaluation(config, checkpoint=None):
    import numpy as np
    import ray
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    from moldr.sampling import Sampler

    config = copy.deepcopy(config)
    checkpoint = checkpoint or config.output.checkpoint
    if checkpoint is not None:
        config.output.checkpoint = str(checkpoint)
    config = prepare_restored_config(config)
    config, blocks, catalog = resolve_config(config)
    if not config.evaluation.random and not checkpoint:
        raise ValueError("Actor evaluation requires --checkpoint; pass --random for a baseline")
    directory = save_run(_directory(config, "evaluation"), config, blocks=blocks, catalog=catalog)
    if checkpoint is not None:
        actual_checkpoint = Path(checkpoint_path(checkpoint))
        write_json(directory / "evaluation_checkpoint.json", {
            "path": str(actual_checkpoint),
            "sha256": {str(path.relative_to(actual_checkpoint)): sha256_file(path)
                       for path in sorted(actual_checkpoint.rglob("*")) if path.is_file()},
        })
    env = None
    algo = None
    ray_was_initialized = ray.is_initialized()
    try:
        env = make_environment(config)
        if not config.evaluation.random:
            _init_ray(config)
            algo = make_ppo_config(config).build()
            algo.restore(checkpoint_path(checkpoint))
        sampler = Sampler(env, algo, selector=config.evaluation.selector, gamma=config.training.gamma,
                          explore=config.evaluation.explore)
        rows = []
        for index in range(config.evaluation.episodes):
            start = time.perf_counter()
            episode = sampler.episode(seed=config.training.seed + index, random=config.evaluation.random)
            # EpisodeResult is the same contract for all end conditions.
            rows.append({"seed": config.training.seed + index, "smiles": episode.smiles,
                         "score": episode.score, "reward": episode.total_reward,
                         "steps": episode.steps, "terminated": episode.terminated, "truncated": episode.truncated,
                         "seconds": time.perf_counter() - start, "scoring": episode.scoring,
                         "lookahead_previews": sum(info["lookahead"]["score_requests"] for info in episode.infos if "lookahead" in info),
                         "lookahead_attachment_choices": sum("lookahead" in info for info in episode.infos)})
        write_json(directory / "molecules.json", rows)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        mols = [Chem.MolFromSmiles(row["smiles"]) for row in rows]
        fps = [gen.GetFingerprint(mol) for mol in mols if mol is not None]
        similarities = [DataStructs.TanimotoSimilarity(left, right) for i, left in enumerate(fps) for right in fps[:i]]
        report = {"episodes": len(rows), "mean_score": float(np.mean([row["score"] for row in rows])),
                  "validity": len(fps) / len(rows), "unique_fraction": len({Chem.MolToSmiles(m) for m in mols if m is not None}) / len(rows),
                  "diversity": 1 - float(np.mean(similarities)) if similarities else None,
                  "seconds": sum(row["seconds"] for row in rows),
                  "episodes_per_second": len(rows) / sum(row["seconds"] for row in rows),
                  "lookahead_previews": sum(row["lookahead_previews"] for row in rows),
                  "lookahead_attachment_choices": sum(row["lookahead_attachment_choices"] for row in rows),
                  "scoring": env.scorer.stats if hasattr(env, "scorer") else None}
        write_json(directory / "evaluation.json", report)
        print(json.dumps({"output": str(directory), **report}), flush=True)
        return directory
    finally:
        try:
            if env is not None:
                env.close()
            if algo is not None:
                algo.stop()
        finally:
            if not ray_was_initialized and ray.is_initialized():
                ray.shutdown()


def run_sweep(base, path, *, dry_run=False):
    spec = yaml.safe_load(Path(path).read_text())
    if not isinstance(spec, dict) or set(spec) - {"jobs", "command"} or not isinstance(spec.get("jobs"), list):
        raise ValueError("Sweep must explicitly list jobs (no implicit Cartesian product)")
    command = spec.get("command", "profile")
    if command not in ("train", "profile"):
        raise ValueError("Sweep command must be train or profile")
    configs = []
    names = set()
    for job in spec["jobs"]:
        if not isinstance(job, dict) or set(job) - {"name", "set", "config"} or not isinstance(job.get("name"), str):
            raise ValueError("Each sweep job requires a name and optional set/config")
        if job["name"] in names or Path(job["name"]).name != job["name"]:
            raise ValueError("Sweep names must be unique plain filenames")
        names.add(job["name"])
        overrides = job.get("set", [])
        if not isinstance(overrides, list) or not all(isinstance(v, str) for v in overrides):
            raise ValueError("Sweep set must be a list of section.key=value strings")
        job_config = Path(path).parent / job["config"] if job.get("config") else None
        config = load_config(job_config, overrides, base=base)
        config.output.run_name = job["name"]
        config, _, _ = resolve_config(config)
        configs.append(config)
    directory = Path(base.output.directory) / ("sweep_" + datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d_%H-%M-%S-%f"))
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [{"name": config.output.run_name, "diff": config_diff(base, config), "status": "planned"} for config in configs]
    write_json(directory / "sweep.json", {"command": command, "serial": True, "jobs": jobs})
    for config, job in zip(configs, jobs):
        config.output.directory = str(directory)
        if dry_run:
            print(json.dumps(job), flush=True)
            continue
        job["status"] = "running"
        write_json(directory / "sweep.json", {"command": command, "serial": True, "jobs": jobs})
        try:
            job["directory"] = str(run_training(config, profile=command == "profile"))
            job["status"] = "complete"
        except Exception as error:
            job["status"], job["error"] = "failed", str(error)
            raise
        finally:
            write_json(directory / "sweep.json", {"command": command, "serial": True, "jobs": jobs})
    if not dry_run:
        from scripts.experiment_report import build_report

        build_report(directory)
    return directory


def main(argv=None, default_command=None):
    args = parser(default_command).parse_args(argv)
    command = default_command or args.command
    if args.sweep is not None and command != "sweep":
        raise ValueError("--sweep is only supported by the sweep command")
    values = explicit_values(args)
    initial = load_config(args.config, args.overrides, values, validate=False)
    if initial.output.checkpoint:
        config, _ = load_saved_config(initial.output.checkpoint, args.overrides, values, config_path=args.config)
        config.output.checkpoint = initial.output.checkpoint
    else:
        config = load_config(args.config, args.overrides, values)
    if command == "sweep":
        if not args.sweep:
            raise ValueError("sweep requires --sweep path.yaml")
        return run_sweep(config, args.sweep, dry_run=args.dry_run)
    if args.dry_run:
        resolved, _, _ = resolve_config(config)
        print(yaml.safe_dump(resolved.to_dict(), sort_keys=False))
        return resolved
    if command in ("train", "profile"):
        return run_training(config, profile=command == "profile")
    return run_evaluation(config, config.output.checkpoint)


if __name__ == "__main__":
    main()
