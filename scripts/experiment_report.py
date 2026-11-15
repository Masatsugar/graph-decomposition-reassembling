"""Build local benchmark tables and standalone scientific throughput plots."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

METRICS = (
    "seconds",
    "iterations",
    "molecule_updates_per_second",
    "valid_molecules_per_second",
    "completed_valid_molecules",
    "completed_valid_molecules_per_second",
    "score_requests",
    "score_computations",
    "score_cache_hits",
    "mean_final_score",
    "mean_sampled_unique_fraction",
    "mean_smoothed_sample_time_ms",
    "mean_smoothed_learn_time_ms",
    "gpu_peak_mib",
)


def group_name(name):
    """Collapse the seed suffix while retaining each worker configuration."""
    return re.sub(r"_seed[0-9]+(?=_|$)", "", name, count=1)


def _finite(values):
    return [
        float(value) for value in values if value is not None and np.isfinite(value)
    ]


def _mean(values):
    values = _finite(values)
    return float(np.mean(values)) if values else None


def seed_statistics(values):
    values = _finite(values)
    return {
        "mean": float(np.mean(values)) if values else None,
        "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "n": len(values),
    }


def group_rows(rows, planned_names=()):
    groups = defaultdict(list)
    planned = Counter(group_name(name) for name in planned_names)
    for row in rows:
        groups[group_name(row["name"])].append(row)
    return [
        {
            "name": name,
            "n": len(trials),
            "planned_n": planned.get(name, len(trials)),
            "runs": [row["name"] for row in trials],
            "metrics": {
                metric: seed_statistics(row.get(metric) for row in trials)
                for metric in METRICS
            },
        }
        for name, trials in groups.items()
    ]


def _trial_row(job):
    run = Path(job["directory"])
    records = [
        record
        for record in json.loads((run / "measurements.json").read_text())
        if record["measured"]
    ]
    seconds = sum(record["seconds"] for record in records)
    counts = {
        key: sum(record["chemistry"].get(key, 0) for record in records)
        for key in (
            "updates",
            "valid_molecules",
            "episodes",
            "quality_sum",
            "requests",
            "computations",
            "hits",
        )
    }
    peaks = _finite(record.get("peak_gpu_memory_mib") for record in records)
    return {
        "name": job["name"],
        "seconds": seconds,
        "iterations": len(records),
        "molecule_updates_per_second": counts["updates"] / seconds
        if seconds > 0
        else None,
        "valid_molecules_per_second": counts["valid_molecules"] / seconds
        if seconds > 0
        else None,
        "completed_valid_molecules": counts["episodes"],
        "completed_valid_molecules_per_second": counts["episodes"] / seconds
        if seconds > 0
        else None,
        "score_requests": counts["requests"],
        "score_computations": counts["computations"],
        "score_cache_hits": counts["hits"],
        "mean_final_score": counts["quality_sum"] / counts["episodes"]
        if counts["episodes"]
        else None,
        "mean_sampled_unique_fraction": _mean(
            record["chemistry"].get("sampled_unique_fraction") for record in records
        ),
        "mean_smoothed_sample_time_ms": _mean(
            record["timers"].get("sample_time_ms") for record in records
        ),
        "mean_smoothed_learn_time_ms": _mean(
            record["timers"].get("learn_time_ms") for record in records
        ),
        "gpu_peak_mib": max(peaks) if peaks else None,
        "config_diff": job["diff"],
    }


def _bars(axis, groups, metric, label, *, color="#315d89"):
    positions = [
        index
        for index, group in enumerate(groups)
        if group["metrics"][metric]["mean"] is not None
    ]
    values = [groups[index]["metrics"][metric]["mean"] for index in positions]
    errors = [groups[index]["metrics"][metric]["sample_sd"] or 0 for index in positions]
    axis.bar(positions, values, yerr=errors, capsize=3, color=color)
    axis.set_xticks(
        np.arange(len(groups)),
        [f"{group['name']}\nn={group['n']}/{group['planned_n']}" for group in groups],
        rotation=60,
        ha="right",
        fontsize=7,
    )
    axis.set_ylabel(label)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    if not positions:
        axis.text(0.5, 0.5, "No measured values", ha="center", transform=axis.transAxes)


def _plots(groups, directory):
    if not groups:
        return []
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/moldr_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    width = max(7, min(14, len(groups) * 0.7))
    outputs = []

    def save(figure, stem):
        figure.tight_layout()
        figure.savefig(directory / f"{stem}.png", dpi=160)
        figure.savefig(directory / f"{stem}.svg")
        plt.close(figure)
        outputs.append(stem)

    figure, axes = plt.subplots(2, 1, figsize=(width, 9))
    _bars(
        axes[0],
        groups,
        "molecule_updates_per_second",
        "Valid selected molecular updates / second",
    )
    _bars(
        axes[1],
        groups,
        "completed_valid_molecules_per_second",
        "Valid completed final molecules / second",
        color="#24826f",
    )
    figure.suptitle("PPO throughput: seed mean and sample SD; tracing disabled")
    save(figure, "throughput")

    figure, axes = plt.subplots(2, 1, figsize=(width, 9))
    _bars(
        axes[0],
        groups,
        "mean_final_score",
        "Mean final training objective score",
        color="#24826f",
    )
    _bars(
        axes[1],
        groups,
        "mean_sampled_unique_fraction",
        "Rolling sampled uniqueness fraction",
        color="#715695",
    )
    axes[1].set_ylim(0, 1.05)
    figure.suptitle("Training quality proxies: seed mean and sample SD")
    save(figure, "quality_proxies")

    return outputs


def _display(stats, digits=2):
    if stats["mean"] is None:
        return "—"
    value = f"{stats['mean']:.{digits}f}"
    return (
        value
        if stats["sample_sd"] is None
        else f"{value} ± {stats['sample_sd']:.{digits}f}"
    )


def build_report(directory):
    directory = Path(directory)
    sweep = json.loads((directory / "sweep.json").read_text())
    rows = [_trial_row(job) for job in sweep["jobs"] if job["status"] == "complete"]
    groups = group_rows(rows, [job["name"] for job in sweep["jobs"]])
    grouped = {
        "completed_jobs": len(rows),
        "total_jobs": len(sweep["jobs"]),
        "groups": groups,
    }
    (directory / "comparison.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    (directory / "grouped_comparison.json").write_text(
        json.dumps(grouped, indent=2, allow_nan=False) + "\n"
    )
    plots = _plots(groups, directory)
    columns = (
        ("molecule_updates_per_second", 2),
        ("completed_valid_molecules_per_second", 2),
        ("completed_valid_molecules", 0),
        ("score_computations", 0),
        ("mean_final_score", 3),
        ("mean_sampled_unique_fraction", 3),
        ("mean_smoothed_sample_time_ms", 1),
        ("mean_smoothed_learn_time_ms", 1),
        ("gpu_peak_mib", 1),
    )
    table = "".join(
        f"<tr><td>{html.escape(group['name'])}</td><td>{group['n']}/{group['planned_n']}</td>"
        + "".join(
            f"<td>{html.escape(_display(group['metrics'][metric], digits))}</td>"
            for metric, digits in columns
        )
        + "</tr>"
        for group in groups
    )
    images = "".join(
        f"<a href='{stem}.svg'><img src='{stem}.png' alt='{stem.replace('_', ' ')}: means and sample standard deviations across completed seeds'></a>"
        for stem in plots
    )
    (directory / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>MOLDR comparison</title>"
        "<style>body{font:15px system-ui;margin:2em;max-width:1500px}td,th{padding:.5em;border-bottom:1px solid #ddd}img{max-width:100%}.table{overflow:auto}</style>"
        f"<h1>MOLDR experiment comparison</h1><p><strong>Completed: {len(rows)} / {len(sweep['jobs'])} jobs.</strong> "
        "Only completed runs contribute. Measured iterations exclude warmup; trace capture runs separately.</p>"
        "<p>Groups retain worker count and collapse only the seed. Values are arithmetic means across completed seeds; "
        "whiskers are sample standard deviations when at least two seeds have values. Missing completion scores remain undefined.</p>"
        "<p>The legacy valid_molecules field counts valid selected molecular updates. Valid completed final molecules count episode outputs, "
        "including terminal, truncated, and dead-end episodes. Every environment base has passed its chemistry validation gate. "
        "Rolling sampled uniqueness uses bounded environment histories, "
        "including warmup; it is a training proxy. Fingerprint diversity is reported by post-training sampling. "
        "Sample/learn timers are RLlib smoothed values and are shown separately. These short runs do not establish convergence.</p>"
        + images
        + "<div class='table'><table><thead><tr><th>Configuration</th><th>Seeds complete/planned</th><th>Valid selected updates/s</th>"
        "<th>Valid completed final molecules/s</th><th>Completed valid molecules</th>"
        "<th>Score computations</th><th>Final training score</th><th>Rolling uniqueness</th><th>Smoothed sample ms</th>"
        "<th>Smoothed learn ms</th><th>Peak allocated CUDA MiB</th></tr></thead><tbody>"
        + table
        + "</tbody></table></div><p><a href='comparison.json'>Per-job measurements</a> · "
        "<a href='grouped_comparison.json'>Seed statistics</a> · <a href='sweep.json'>Configuration differences and job state</a></p>"
    )
    return directory / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build_report(parser.parse_args().directory)
