"""Paired wall-clock benchmark of Python and C++ reassembly during RLlib PPO."""

import argparse
import gc
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import ray
import torch
from ray.rllib.utils.metrics import LEARN_ON_BATCH_TIMER, SAMPLE_TIMER

from moldr.cli import _init_ray
from moldr.experiment import ExperimentConfig
from moldr.factory import make_ppo_config, resolve_config
from moldr.metrics import MoleculeMetrics, collect_metrics, metric_delta
from moldr.provenance import sha256_file, write_json


def worker_status(worker):
    return {
        "pid": os.getpid(),
        "backend": worker.foreach_env(
            lambda env: env.candidate_generator.config["backend"]
        ),
        "native_loaded": "moldr.core._native_reassembly" in sys.modules,
    }


def run(config, warmup, measured):
    config, blocks, _ = resolve_config(config)
    start = time.perf_counter()
    algo = make_ppo_config(config).callbacks(MoleculeMetrics).build()
    setup_seconds = time.perf_counter() - start
    try:
        policy = algo.get_policy()
        device = next(policy.model.parameters()).device
        expected_device = "cuda" if config.training.num_gpus else "cpu"
        if device.type != expected_device:
            raise RuntimeError(f"Learner on {device}, expected {expected_device}")
        initial = [p.detach().clone() for p in policy.model.parameters()]
        records = []
        timers = {
            "sampling_seconds": algo._timers[SAMPLE_TIMER],
            "learning_seconds": algo._timers[LEARN_ON_BATCH_TIMER],
        }
        total_steps = 0
        for index in range(warmup + measured):
            before = collect_metrics(algo)
            prior = {name: timer._total_time for name, timer in timers.items()}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = algo.train()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall = time.perf_counter() - start
            steps = result["num_env_steps_sampled_this_iter"]
            total_steps += steps
            if (
                steps != config.training.train_batch_size
                or result["num_env_steps_trained"] != total_steps
            ):
                raise RuntimeError("Unexpected number of sampled or trained steps")
            stats = result["info"]["learner"]["default_policy"]["learner_stats"]
            record = {
                "iteration": index + 1,
                "measured": index >= warmup,
                "wall_seconds": wall,
                "steps": steps,
                "trained_steps_total": total_steps,
                **{name: t._total_time - prior[name] for name, t in timers.items()},
                "chemistry": metric_delta(before, collect_metrics(algo), wall),
                "loss": float(stats["total_loss"]),
            }
            records.append(record)
            print(
                config.chemistry.backend,
                config.training.seed,
                index + 1,
                f"{wall:.2f} s, {steps / wall:.1f} steps/s",
                flush=True,
            )
        workers = algo.env_runner_group.foreach_env_runner(
            worker_status, local_env_runner=False
        )
        if (
            len({w["pid"] for w in workers}) != config.training.num_workers
            or any(w["backend"] != [config.chemistry.backend] for w in workers)
            or (
                config.chemistry.backend == "cpp"
                and not all(w["native_loaded"] for w in workers)
            )
        ):
            raise RuntimeError(f"Unexpected remote rollout worker state: {workers}")
        change = sum(
            (p.detach() - p0).abs().sum().item()
            for p, p0 in zip(policy.model.parameters(), initial, strict=True)
        )
        if not (0 < change < float("inf")):
            raise RuntimeError("PPO weights failed to update")
        return {
            "config": config.to_dict(),
            "fragment_count": len(blocks),
            "workers": workers,
            "build_seconds": setup_seconds,
            "iterations": records,
            "weight_l1_change": change,
            "model_device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        }
    finally:
        algo.stop()
        del algo
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Repeat to benchmark multiple seeds (default: 11 and 29).",
    )
    parser.add_argument("--num-gpus", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/native-reassembly/ppo.json")
    )
    args = parser.parse_args()
    config = ExperimentConfig()
    config.fragments.pool_size = 128
    config.training.num_workers = args.workers
    config.training.num_gpus = args.num_gpus
    config.training.train_batch_size = args.batch
    config.training.num_epochs = args.epochs
    config.training.object_store_memory_mib = 3072
    if min(args.workers, args.warmup, args.measured) < 1 or args.batch % args.workers:
        parser.error(
            "Use positive workers, warmup and measured counts, and an even batch split."
        )
    from moldr.core import _native_reassembly

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ("rdkit", "ray", "torch")},
        "native_sha256": sha256_file(_native_reassembly.__file__),
        "warmup": args.warmup,
        "measured": args.measured,
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    _init_ray(config)
    report["ray_init_seconds"] = time.perf_counter() - start
    try:
        for index, seed in enumerate(args.seed or [11, 29]):
            order = ("python", "cpp") if index % 2 == 0 else ("cpp", "python")
            for backend in order:
                config.training.seed = seed
                config.chemistry.backend = backend
                report["runs"].append(run(config, args.warmup, args.measured))
                write_json(args.output, report)
    finally:
        ray.shutdown()
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
