"""Profile actual distributed CPU rollouts and CUDA PPO learning.

Exports Chrome/Perfetto trace JSON, a standalone timeline, and uninstrumented
worker-scaling measurements. Run from the repository root; no cloud resources
are created. Clock alignment is valid for the local, single-host Ray cluster.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
from guacamol.standard_benchmarks import qed_benchmark
from rdkit import RDLogger

from moldr.config import get_default_config_v1
from moldr.e2e.config import get_e2e_config
from moldr.generative.vocab import FragmentVocab
from moldr.profiling import (
    disable_snapshot,
    disable_workers,
    enable,
    enable_workers,
    span,
)


def checked_train(algo, batch_size):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = algo.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    sampled = result["num_env_steps_sampled_this_iter"]
    trained = result["num_env_steps_trained_this_iter"]
    if sampled != batch_size or trained != batch_size:
        raise RuntimeError(
            f"Unequal workload: sampled={sampled}, trained={trained}, expected={batch_size}"
        )
    stats = result["info"]["learner"]["default_policy"]["learner_stats"]
    if not all(
        np.isfinite(stats[k])
        for k in ("total_loss", "policy_loss", "vf_loss", "kl", "entropy")
    ):
        raise RuntimeError(f"Nonfinite learner statistics: {stats}")
    return {
        "seconds": elapsed,
        "env_steps": sampled,
        "trained_steps": trained,
        "steps_per_second": sampled / elapsed,
        "learner_stats": stats,
        "ray_smoothed_timers": result.get("timers", {}),
    }


def make_config(mode, worker_count, blocks, args):
    common = {
        "model_path": "models/model_300dim.pkl",
        "num_workers": worker_count,
        "num_gpus": 1,
        "score_threshold": 1.1,
        "length": args.length,
    }
    if mode == "legacy":
        config = get_default_config_v1(qed_benchmark(), blocks, **common)
    else:
        config = get_e2e_config(
            qed_benchmark(),
            blocks,
            **common,
            option_pool_size=args.option_pool_size,
            max_attach=args.max_attach,
            hidden=args.hidden,
            max_episode_steps=2 * args.length,
        )
    config.training(
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        num_epochs=args.num_epochs,
    )
    config.env_runners(
        rollout_fragment_length=args.train_batch_size // worker_count,
        batch_mode="truncate_episodes",
    )
    config.reporting(
        min_sample_timesteps_per_iteration=args.train_batch_size,
        min_time_s_per_iteration=0,
    )
    config.debugging(seed=args.seed, log_level="ERROR")
    return config


def profile_iteration(algo, mode, workers, args, directory):
    from scripts.profile_analysis import analyze_trace, merge_trace
    from scripts.profile_viewer import build_overview, build_viewer

    common_epoch = time.time_ns()
    # Install before starting the measured span; local driver recorder remains
    # independent of remote worker callbacks.
    enable_workers(algo, common_epoch, max_events=args.max_events)
    enable(
        start_epoch_ns=common_epoch,
        process_name="driver / CUDA learner",
        max_events=args.max_events,
        torch_ranges=True,
        include_rllib=True,
    )
    torch_trace_path = directory / "torch_trace.json"
    try:
        with (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as profiler,
            span(
                "PPO.iteration",
                "iteration",
                {
                    "mode": mode,
                    "workers": workers,
                    "global_train_batch": args.train_batch_size,
                },
            ),
        ):
            profiled_run = checked_train(algo, args.train_batch_size)
        profiler.export_chrome_trace(str(torch_trace_path))
    finally:
        driver = disable_snapshot()
        worker_snapshots = disable_workers(algo)
    custom = driver["traceEvents"] + [
        event for snapshot in worker_snapshots for event in snapshot["traceEvents"]
    ]
    torch_trace = json.loads(torch_trace_path.read_text())
    trace = merge_trace(custom, torch_trace, common_epoch)
    metadata = {
        "mode": mode,
        "workers": workers,
        "clock_alignment": "single-host shared wall clock + monotonic per-process spans",
        "global_train_batch": args.train_batch_size,
        "fragment_length": args.train_batch_size // workers,
        "minibatch_size": args.minibatch_size,
        "optimization_epochs": args.num_epochs,
        "profiled_run": profiled_run,
        "warmup_iterations": args.warmup,
        "prior_measured_iterations": 0 if args.profile_only else args.iterations,
        "dropped_events": driver.get("dropped_events", 0)
        + sum(s.get("dropped_events", 0) for s in worker_snapshots),
        "note": "Instrumented iteration is separate from uninstrumented throughput trials. CPU+GPU spans overlap; nested inclusive durations are not additive.",
    }
    trace["moldr_profile_metadata"] = metadata
    if metadata["dropped_events"]:
        raise RuntimeError(
            f"Trace incomplete: {metadata['dropped_events']} events dropped; increase --max-events"
        )
    if not any(e.get("cat") == "kernel" for e in trace["traceEvents"]):
        raise RuntimeError("No CUDA kernel events captured")
    trace_path = directory / "trace.json"
    trace_path.write_text(json.dumps(trace, separators=(",", ":")))
    summary = analyze_trace(trace)
    if len(summary["workers"]) != workers or any(
        worker["sample_calls"] < 1 for worker in summary["workers"]
    ):
        raise RuntimeError(
            "Missing measured sampler spans for one or more rollout workers"
        )
    fragments = [
        event
        for event in trace["traceEvents"]
        if event.get("name") == "SyncSampler.next"
        and event.get("args", {}).get("source_type") == "custom"
    ]
    if (
        fragments
        and sum(event["args"].get("sampled_steps", 0) for event in fragments)
        != args.train_batch_size
    ):
        raise RuntimeError(
            "Traced worker fragments do not match the global training batch"
        )
    summary["configuration"] = metadata
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    build_viewer(trace_path, directory / "timeline.html", summary_path)
    build_overview(trace_path, directory / "overview.svg", summary_path)
    return {
        "trace": str(trace_path),
        "timeline": str(directory / "timeline.html"),
        "overview": str(directory / "overview.svg"),
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/profiling/2026-10-13")
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("legacy", "e2e"), default=["legacy", "e2e"]
    )
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--trace-workers", nargs="+", type=int, default=[2, 8])
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--option-pool-size", type=int, default=32)
    parser.add_argument("--max-attach", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-events", type=int, default=500000)
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Refresh traces while retaining existing uninstrumented scaling measurements",
    )
    args = parser.parse_args()
    if (
        min(
            args.train_batch_size, args.minibatch_size, args.num_epochs, args.max_events
        )
        < 1
    ):
        parser.error(
            "batch sizes, optimization epochs and event limit must be positive"
        )
    for workers in args.workers:
        if workers < 1 or args.train_batch_size % workers:
            parser.error("workers must be positive divisors of the global train batch")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("at least one warmup and measured iteration are required")
    if args.train_batch_size % args.minibatch_size:
        parser.error("train batch must be divisible by minibatch size")
    if not torch.cuda.is_available():
        raise RuntimeError("A visible CUDA GPU is required")
    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.*")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = (
        pd.read_csv("data/building_blocks/guacamol/guacamol_s10000_l2.csv", index_col=0)
        .iloc[:, 0]
        .tolist()
    )
    blocks = FragmentVocab.from_smiles_list(raw).smiles
    report = {
        "settings": vars(args) | {"output_dir": str(args.output_dir)},
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "logical_cpus": os.cpu_count(),
            "cpu": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            ),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ray": ray.__version__,
            "cuda": torch.version.cuda,
        },
        "vocab_size": len(blocks),
        "cases": [],
        "interpretation": "Same global batch/optimization settings within each mode. Different worker counts change sampled trajectories. Timings are throughput checks, not paper reproduction or a quality comparison. All clocks are aligned only within this host.",
    }
    report_path = args.output_dir / "scaling.json"
    if args.profile_only:
        if not report_path.exists():
            parser.error("--profile-only requires an existing scaling.json")
        report = json.loads(report_path.read_text())
        for key in (
            "train_batch_size",
            "minibatch_size",
            "num_epochs",
            "length",
            "option_pool_size",
            "max_attach",
            "hidden",
            "seed",
        ):
            if report["settings"][key] != getattr(args, key):
                parser.error(
                    f"--profile-only must preserve original {key}={report['settings'][key]}"
                )

    def save():
        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
            )
            + "\n"
        )

    ray.init(
        num_cpus=max(args.workers) + 1,
        num_gpus=1,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=2 * 2**30,
        runtime_env={
            "env_vars": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        },
    )
    try:
        for mode in args.modes:
            for workers in args.workers:
                if args.profile_only and workers not in args.trace_workers:
                    continue
                print(
                    json.dumps({"stage": "building", "mode": mode, "workers": workers}),
                    flush=True,
                )
                algo = make_config(mode, workers, blocks, args).build()
                try:
                    actual_device = str(
                        next(algo.get_policy().model.parameters()).device
                    )
                    if not actual_device.startswith("cuda"):
                        raise RuntimeError(f"Learner is on {actual_device}")
                    for _ in range(args.warmup):
                        checked_train(algo, args.train_batch_size)
                    if args.profile_only:
                        directory = args.output_dir / f"{mode}-workers-{workers}"
                        directory.mkdir(exist_ok=True)
                        profile = profile_iteration(
                            algo, mode, workers, args, directory
                        )
                        case = next(
                            c
                            for c in report["cases"]
                            if c["mode"] == mode and c["workers"] == workers
                        )
                        case["profile"] = profile
                        save()
                        print(
                            json.dumps(
                                {
                                    "stage": "profile refreshed",
                                    "mode": mode,
                                    "workers": workers,
                                }
                            ),
                            flush=True,
                        )
                        continue
                    runs = []
                    for iteration in range(args.iterations):
                        run = checked_train(algo, args.train_batch_size)
                        runs.append(run)
                        print(
                            json.dumps(
                                {
                                    "stage": "measuring",
                                    "mode": mode,
                                    "workers": workers,
                                    "iteration": iteration + 1,
                                    "seconds": run["seconds"],
                                }
                            ),
                            flush=True,
                        )
                    case = {
                        "mode": mode,
                        "workers": workers,
                        "fragment_length": args.train_batch_size // workers,
                        "device": actual_device,
                        "unprofiled_runs": runs,
                        "mean_seconds": float(np.mean([r["seconds"] for r in runs])),
                        "std_seconds": float(np.std([r["seconds"] for r in runs])),
                        "mean_steps_per_second": float(
                            np.mean([r["steps_per_second"] for r in runs])
                        ),
                    }
                    if workers in args.trace_workers:
                        directory = args.output_dir / f"{mode}-workers-{workers}"
                        directory.mkdir(exist_ok=True)
                        case["profile"] = profile_iteration(
                            algo, mode, workers, args, directory
                        )
                    report["cases"].append(case)
                    save()
                finally:
                    algo.stop()
    finally:
        ray.shutdown()
    from scripts.profile_viewer import build_index

    build_index(report_path, args.output_dir / "index.html")
    print(f"Saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
