"""Render measured Chrome Trace JSON as a self-contained, offline timeline.

No browser libraries or Python plotting packages are required. The original JSON
can also be opened in Perfetto or chrome://tracing for detailed analysis.

    python -m scripts.profile_viewer --trace tmp/profile/trace.json \
        --html tmp/profile/timeline.html --svg tmp/profile/overview.svg
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import html
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

_COLORS = (
    "#2563eb",
    "#0d9488",
    "#7c3aed",
    "#dc2626",
    "#d97706",
    "#0891b2",
    "#db2777",
    "#4f46e5",
    "#65a30d",
    "#9333ea",
    "#ea580c",
    "#0284c7",
)
_RAW_CATEGORIES = {
    "cpu_op",
    "cuda_runtime",
    "cuda_driver",
    "fwdbwd",
    "ac2g",
    "python_function",
    "python",
    "gpu_memcpy",
    "gpu_memset",
    "Trace",
    "triton",
    "user_annotation",
    "gpu_user_annotation",
    "overhead",
}
_CATEGORY_COLORS = {
    "iteration": "#2563eb",
    "driver": "#2563eb",
    "sampling_wait": "#2563eb",
    "optimization": "#d97706",
    "learner": "#d97706",
    "sync": "#0891b2",
    "rollout": "#0d9488",
    "env": "#65a30d",
    "environment": "#65a30d",
    "chemistry": "#dc2626",
    "embedding": "#7c3aed",
    "scoring": "#db2777",
    "inference": "#9333ea",
    "kernel": "#d946ef",
    "gpu_kernel": "#d946ef",
    "gpu_memcpy": "#0284c7",
    "gpu_memset": "#0891b2",
    "cpu_op": "#64748b",
    "user_annotation": "#94a3b8",
    "cuda_runtime": "#475569",
}


def _category_color(category: str) -> str:
    return _CATEGORY_COLORS.get(
        category.rsplit(".", 1)[-1], _COLORS[sum(map(ord, category)) % len(_COLORS)]
    )


def _read_json(path: str | Path) -> Any:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    """Total busy time; nested or simultaneous events are counted once."""
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _is_gpu(event: dict[str, Any], process_name: str = "") -> bool:
    # A GPU process can also contain profiler annotations. Only actual device
    # activity contributes to the measured busy-time union.
    del process_name
    category = str(event.get("cat", "")).lower()
    return category in {"kernel", "gpu_kernel", "gpu_memcpy", "gpu_memset"}


def _is_detail_category(category: str) -> bool:
    lower = category.lower()
    return category in _RAW_CATEGORIES or any(
        word in lower for word in ("chemistry", "embedding")
    )


def _prepare_trace(trace_path: str | Path) -> dict[str, Any]:
    trace = _read_json(trace_path)
    source_events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    if not isinstance(source_events, list):
        raise TypeError("traceEvents must be a list")
    processes: dict[str, dict[str, Any]] = {}
    threads: dict[tuple[str, str], dict[str, Any]] = {}
    for event in source_events:
        if not isinstance(event, dict) or event.get("ph") != "M":
            continue
        pid, tid = str(event.get("pid", 0)), str(event.get("tid", 0))
        args = event.get("args") or {}
        process = processes.setdefault(pid, {"id": pid, "name": f"Process {pid}"})
        thread = threads.setdefault((pid, tid), {"id": tid, "name": f"Thread {tid}"})
        if event.get("name") == "process_name":
            process["name"] = str(args.get("name", process["name"]))
        elif event.get("name") == "process_sort_index":
            process["sort"] = _finite_number(args.get("sort_index"))
        elif event.get("name") == "thread_name":
            thread["name"] = str(args.get("name", thread["name"]))
        elif event.get("name") == "thread_sort_index":
            thread["sort"] = _finite_number(args.get("sort_index"))

    events: list[dict[str, Any]] = []
    counters: list[dict[str, Any]] = []
    begins: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in source_events:
        if not isinstance(event, dict):
            continue
        phase = event.get("ph")
        pid, tid = str(event.get("pid", 0)), str(event.get("tid", 0))
        if phase == "B":
            begins[(pid, tid)].append(event)
            continue
        if phase == "E":
            stack = begins[(pid, tid)]
            if not stack:
                continue
            begin = stack.pop()
            event = {
                **begin,
                "dur": _finite_number(event.get("ts"))
                - _finite_number(begin.get("ts")),
            }
            phase = "X"
        if phase not in {"X", "C"}:
            continue
        processes.setdefault(pid, {"id": pid, "name": f"Process {pid}"})
        threads.setdefault((pid, tid), {"id": tid, "name": f"Thread {tid}"})
        category = str(event.get("cat", "uncategorized")) or "uncategorized"
        compact = {
            "n": str(event.get("name", "unnamed")),
            "c": category,
            "s": _finite_number(event.get("ts")),
            "d": max(0.0, _finite_number(event.get("dur"))),
            "p": pid,
            "t": tid,
            "a": event.get("args") or {},
            "g": _is_gpu(event, processes[pid]["name"]),
        }
        if phase == "C":
            # Chrome counters use a numeric value for each args key.
            values = {
                str(key): float(value)
                for key, value in compact["a"].items()
                if isinstance(value, (int, float)) and math.isfinite(value)
            }
            if values:
                compact["a"] = values
                counters.append(compact)
        else:
            events.append(compact)
    events.sort(key=lambda event: (event["s"], -event["d"]))
    counters.sort(key=lambda event: event["s"])
    if not events and not counters:
        raise ValueError("The trace contains no complete duration or counter events")

    start = min(event["s"] for event in events + counters)
    end = max(event["s"] + event["d"] for event in events + counters)
    duration = max(1.0, end - start)
    categories = sorted({event["c"] for event in events})
    active_processes = {event["p"] for event in events + counters}
    active_threads = {(event["p"], event["t"]) for event in events + counters}
    process_list = sorted(
        (
            process
            for process in processes.values()
            if process["id"] in active_processes
        ),
        key=lambda item: (
            item.get("sort", 1000),
            "gpu" in item["name"].lower(),
            item["name"],
            item["id"],
        ),
    )
    for process in process_list:
        process["threads"] = [
            {**thread, "pid": process["id"]}
            for key, thread in sorted(
                threads.items(),
                key=lambda item: (
                    item[1].get("sort", 1000),
                    item[1]["name"],
                    item[0][1],
                ),
            )
            if key[0] == process["id"] and key in active_threads
        ]
        process["gpu"] = any(
            event["g"] and event["p"] == process["id"] for event in events
        )
    gpu_intervals = [
        (event["s"], event["s"] + event["d"]) for event in events if event["g"]
    ]
    return {
        "events": events,
        "counters": counters,
        "processes": process_list,
        "categories": [
            {
                "name": category,
                "color": _category_color(category),
                "detail": _is_detail_category(category),
            }
            for category in categories
        ],
        "start": start,
        "end": end,
        "duration": duration,
        "gpuBusy": _union_duration(gpu_intervals),
        "source": Path(trace_path).name,
        "traceMetadata": {
            key: value for key, value in trace.items() if key != "traceEvents"
        }
        if isinstance(trace, dict)
        else {},
    }


def build_viewer(
    trace_path: str | Path,
    output_html: str | Path,
    summary_path: str | Path | None = None,
) -> Path:
    """Write an offline HTML viewer; every bar comes from the supplied trace."""
    data = _prepare_trace(trace_path)
    data["summary"] = _read_json(summary_path) if summary_path else None
    # Script data is JSON, not interpolated JavaScript. Escape the HTML parser's
    # closing-tag characters as well as JavaScript's legacy line separators.
    payload = json.dumps(
        data, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    payload = (
        payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _HTML_TEMPLATE.replace("__TRACE_DATA__", payload), encoding="utf-8"
    )
    return output


def _assign_tracks(events: list[dict[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    """Allocate independent rows to overlapping/nested complete spans."""
    occupied: list[tuple[float, int]] = []
    free: list[int] = []
    next_track = 0
    result = []
    for event in sorted(events, key=lambda item: (item["s"], -item["d"])):
        while occupied and occupied[0][0] <= event["s"]:
            _, track = heapq.heappop(occupied)
            heapq.heappush(free, track)
        if free:
            track = heapq.heappop(free)
        else:
            track = next_track
            next_track += 1
        heapq.heappush(occupied, (event["s"] + event["d"], track))
        result.append((event, track))
    return result


def build_overview(
    trace_path: str | Path,
    output_svg: str | Path,
    summary_path: str | Path | None = None,
) -> Path:
    """Write a shareable SVG overview of high-level spans and GPU busy time."""
    # summary_path is accepted to match build_viewer's API. No timings are
    # inferred from a separate summary: the overview always uses trace events.
    del summary_path
    data = _prepare_trace(trace_path)
    colors = {item["name"]: item["color"] for item in data["categories"]}
    included = {item["name"] for item in data["categories"] if not item["detail"]}
    rows: list[tuple[str, list[tuple[dict[str, Any], int]], int]] = []
    for process in data["processes"]:
        for thread in process["threads"]:
            events = [
                event
                for event in data["events"]
                if event["p"] == process["id"]
                and event["t"] == thread["id"]
                and event["c"] in included
            ]
            if not events:
                continue
            label = process["name"]
            if len(process["threads"]) > 1:
                label += " / " + thread["name"]
            if process["gpu"]:
                # Tiny kernels are legible as busy strips at the full-run scale;
                # coalesce overlap, preserving the full intervals in the HTML.
                intervals = sorted(
                    (event["s"], event["s"] + event["d"])
                    for event in events
                    if event["g"]
                )
                merged: list[tuple[float, float]] = []
                for start, end in intervals:
                    if merged and start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                    else:
                        merged.append((start, end))
                rows.append(
                    (
                        label,
                        [
                            (
                                {
                                    "n": "GPU busy",
                                    "c": "kernel",
                                    "s": start,
                                    "d": end - start,
                                },
                                0,
                            )
                            for start, end in merged
                        ],
                        1,
                    )
                )
            else:
                assigned = _assign_tracks(events)
                tracks = min(4, max(track for _, track in assigned) + 1)
                rows.append(
                    (
                        label,
                        [(event, min(track, 3)) for event, track in assigned],
                        tracks,
                    )
                )

    left, right, width = 282, 38, 1480
    plot_width = width - left - right
    top, row_gap, bar_height = 174, 18, 20
    height = (
        top + sum(tracks * bar_height + row_gap + 18 for _, _, tracks in rows) + 120
    )
    escape = lambda value: html.escape(str(value), quote=True)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:"Noto Sans CJK JP","Noto Sans JP",system-ui,sans-serif;fill:#243449} .small{font-size:12px} .label{font-size:13px;font-weight:600}</style>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        '<text x="30" y="42" font-size="25" font-weight="700">GPU 学習・実行プロファイル</text>',
        f'<text x="30" y="68" class="small">実測 Chrome Trace: {escape(data["source"])}</text>',
        f'<text x="30" y="102" font-size="15">トレース区間 {data["duration"] / 1e6:.3f} s ・ {len(data["events"]):,} spans ・ GPU 実行区間 {data["gpuBusy"] / 1e6:.3f} s ({100 * data["gpuBusy"] / data["duration"]:.1f}%)</text>',
        '<text x="30" y="129" class="small">横幅は時間、上下の段は重なる処理。CPU と GPU、worker 間の処理は同時に進みます。詳細は timeline.html を開いて拡大。</text>',
    ]
    plot_bottom = height - 90
    for index in range(9):
        x = left + plot_width * index / 8
        seconds = data["duration"] * index / 8 / 1e6
        parts.append(
            f'<line x1="{x:.2f}" y1="{top - 20}" x2="{x:.2f}" y2="{plot_bottom}" stroke="#dce3ec"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top - 30}" text-anchor="middle" class="small">{seconds:.2f} s</text>'
        )
    y = top
    for label, events, tracks in rows:
        # SVG text stays inside the dedicated label column.
        display_label = label if len(label) <= 34 else label[:31] + "…"
        parts.append(
            f'<text x="24" y="{y + 15}" class="label"><title>{escape(label)}</title>{escape(display_label)}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{plot_width}" height="{tracks * bar_height}" fill="#edf2f7"/>'
        )
        for event, track in events:
            x = left + (event["s"] - data["start"]) / data["duration"] * plot_width
            event_width = max(0.8, event["d"] / data["duration"] * plot_width)
            event_y = y + track * bar_height
            color = colors.get(event["c"], "#db2777")
            title = f"{event['n']} · {event['d'] / 1e3:.3f} ms · {(event['s'] - data['start']) / 1e6:.3f} s"
            parts.append(
                f'<rect x="{x:.2f}" y="{event_y + 1}" width="{event_width:.2f}" height="{bar_height - 2}" rx="2" fill="{color}"><title>{escape(title)}</title></rect>'
            )
            if event_width > 90:
                maximum = max(1, int((event_width - 10) / 7))
                text = event["n"]
                text = text if len(text) <= maximum else text[: maximum - 1] + "…"
                parts.append(
                    f'<text x="{x + 5:.2f}" y="{event_y + 14}" style="fill:white;font-size:11px">{escape(text)}</text>'
                )
        y += tracks * bar_height + row_gap + 18
    parts.extend(
        [
            f'<text x="30" y="{height - 48}" class="small">GPU 実行区間は kernel・copy の重なりを除いた時間です。ハードウェアの使用率や速度比較を表す値ではありません。</text>',
            f'<text x="30" y="{height - 26}" class="small">chemistry・embedding・PyTorch CPU 演算の細部は概要から省略。タイムラインではカテゴリを選んで確認できます。</text>',
            "</svg>",
        ]
    )
    output = Path(output_svg)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


def _scaling_plot(cases: list[dict[str, Any]], metric: str, mode: str) -> str:
    """A small SVG plot of measured means, with descriptive standard deviations."""
    cases = sorted(cases, key=lambda case: case["workers"])
    baseline = next((case for case in cases if case["workers"] == 1), None)
    color = "#2563eb" if mode == "legacy" else "#0d9488"
    points = []
    for case in cases:
        if metric == "seconds":
            value, error = case["mean_seconds"], case.get("std_seconds", 0)
        elif metric == "throughput":
            value = case["mean_steps_per_second"]
            runs = [run["steps_per_second"] for run in case.get("unprofiled_runs", [])]
            error = statistics.pstdev(runs) if runs else 0
        else:
            if not baseline:
                continue
            value, error = baseline["mean_seconds"] / case["mean_seconds"], 0
        points.append((case["workers"], value, error))
    if not points:
        return '<p class="empty">1 worker の基準値がまだありません。</p>'
    width, height = 610, 280
    left, right, top, bottom = 67, 25, 26, 51
    plot_width, plot_height = width - left - right, height - top - bottom
    workers = [point[0] for point in points]
    highest = max(value + error for _, value, error in points)
    if metric == "speedup":
        highest = max(highest, max(workers))
    highest = max(0.001, highest * 1.12)
    power = 10 ** math.floor(math.log10(highest))
    ymax = next(factor * power for factor in (1, 2, 5, 10) if factor * power >= highest)
    log_min, log_max = math.log2(min(workers)), math.log2(max(workers))
    x_at = lambda count: (
        left + (math.log2(count) - log_min) / max(1, log_max - log_min) * plot_width
    )
    y_at = lambda value: top + plot_height * (1 - value / ymax)
    titles = {
        "seconds": "イテレーション時間 (s)",
        "throughput": "throughput (steps/s)",
        "speedup": "1 worker に対する高速化 (倍)",
    }
    esc = lambda value: html.escape(str(value), quote=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(mode)}: {esc(titles[metric])}" style="width:100%;display:block">'
    ]
    svg.append(
        '<style>text{font-family:system-ui,"Noto Sans JP",sans-serif;fill:#65748a;font-size:11px}</style>'
    )
    svg.append(f'<text x="{left}" y="15">{esc(titles[metric])}</text>')
    for index in range(6):
        value = ymax * index / 5
        y = y_at(value)
        precision = 2 if ymax < 1 else 1 if ymax < 10 else 0
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5eaf1"/>'
        )
        svg.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{value:.{precision}f}</text>'
        )
    for count in workers:
        x = x_at(count)
        svg.append(
            f'<text x="{x:.2f}" y="{height - 28}" text-anchor="middle">{count}</text>'
        )
    svg.append(
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 7}" text-anchor="middle">Ray rollout worker 数</text>'
    )
    if metric == "speedup":
        ideal = " ".join(f"{x_at(count):.2f},{y_at(count):.2f}" for count in workers)
        svg.append(
            f'<polyline points="{ideal}" fill="none" stroke="#a8b4c6" stroke-width="1.5" stroke-dasharray="5,5"><title>理想的な worker 数比例の高速化</title></polyline>'
        )
    line = " ".join(
        f"{x_at(count):.2f},{y_at(value):.2f}" for count, value, _ in points
    )
    svg.append(
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.5"/>'
    )
    for count, value, error in points:
        x, y = x_at(count), y_at(value)
        if error:
            y1, y2 = y_at(value + error), y_at(max(0, value - error))
            svg.append(
                f'<path d="M{x:.2f},{y1:.2f}V{y2:.2f} M{x - 5:.2f},{y1:.2f}H{x + 5:.2f} M{x - 5:.2f},{y2:.2f}H{x + 5:.2f}" stroke="{color}" fill="none" stroke-width="1.3"/>'
            )
        title = f"{count} workers: {value:.4f}" + (
            f" ± {error:.4f} (標準偏差)" if error else ""
        )
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" stroke="white" stroke-width="1.5"><title>{esc(title)}</title></circle>'
        )
    svg.append("</svg>")
    return "\n".join(svg)


def build_index(scaling_json_path: str | Path, output_html: str | Path) -> Path:
    """Create an offline entry point with measured scaling and relative links."""
    report = _read_json(scaling_json_path)
    cases = report.get("cases", [])
    if not cases:
        raise ValueError("The scaling report contains no measured cases")
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings, hardware = report.get("settings", {}), report.get("hardware", {})
    esc = lambda value: html.escape(str(value), quote=True)

    def artifact_link(target: str | Path) -> str:
        target = Path(target)
        if (
            not target.is_absolute()
            and not target.exists()
            and (output.parent / target).exists()
        ):
            target = output.parent / target
        return quote(
            os.path.relpath(target.resolve(), output.parent.resolve()).replace(
                os.sep, "/"
            ),
            safe="/.-_",
        )

    mode_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        mode_cases[case["mode"]].append(case)
    charts, tables, profile_cards = [], [], []
    for mode, measured in mode_cases.items():
        measured.sort(key=lambda case: case["workers"])
        baseline = next((case for case in measured if case["workers"] == 1), None)
        charts.append(f'<section class="detail-card"><h2>{esc(mode)}</h2>')
        for metric in ("seconds", "throughput", "speedup"):
            hidden = " hidden" if metric != "seconds" else ""
            charts.append(
                f'<div data-metric="{metric}"{hidden}>{_scaling_plot(measured, metric, mode)}</div>'
            )
        charts.append("</section>")
        rows = []
        for case in measured:
            speedup = (
                f"{baseline['mean_seconds'] / case['mean_seconds']:.2f}×"
                if baseline
                else "—"
            )
            runs = case.get("unprofiled_runs", [])
            rates = [run["steps_per_second"] for run in runs]
            rate_std = statistics.pstdev(rates) if rates else 0
            profile = case.get("profile")
            if profile:
                summary = profile.get("summary", {})
                config = summary.get("configuration", {})
                profile_seconds = config.get("profiled_run", {}).get(
                    "seconds", summary.get("iteration_wall_ms", 0) / 1000
                )
                profiled = f'{profile_seconds:.3f} s <span class="muted">({profile_seconds / case["mean_seconds"]:.2f}×)</span>'
                artifacts = (
                    f'<a href="{artifact_link(profile["timeline"])}">タイムライン</a>'
                )
                phase_rows = "".join(
                    f'<tr><td class="event-name" title="{esc(phase["name"])}">{esc(phase["name"])}</td><td class="num">{phase["duration_ms"] / 1000:.3f} s</td><td class="num">{phase["wall_percent"]:.1f}%</td></tr>'
                    for phase in summary.get("driver_phases", [])
                )
                gpu = summary.get("gpu", {})
                gpu_text = f"GPU kernel / copy 実行区間 {gpu.get('busy_ms', 0) / 1000:.3f} s ({gpu.get('busy_percent_of_iteration', 0):.1f}%)"
                balance = summary.get("worker_balance", {})
                scope_name = (
                    "fragment"
                    if balance.get("scope") == "traced fragment"
                    else "worker sample"
                )
                balance_text = (
                    f"{scope_name} 完了時刻の差 {balance['completion_spread_ms'] / 1000:.3f} s"
                    if "completion_spread_ms" in balance
                    else ""
                )
                profile_cards.append(
                    f'<section class="detail-card profile-card"><h2>{esc(mode)} / {case["workers"]} workers</h2>'
                    f"<p>{esc(gpu_text)} · {esc(balance_text)}</p>"
                    f'<a href="{artifact_link(profile["timeline"])}"><img loading="lazy" src="{artifact_link(profile["overview"])}" alt="{esc(mode)} {case["workers"]} workers の実測タイムライン概要"></a>'
                    f'<table><thead><tr><th>driver フェーズ</th><th style="text-align:right">時間</th><th style="text-align:right">PPO 区間</th></tr></thead><tbody>{phase_rows}</tbody></table>'
                    f'<div class="artifact-links"><a class="primary-link" href="{artifact_link(profile["timeline"])}">拡大して見る</a>'
                    f'<a href="{artifact_link(profile["trace"])}" download>Chrome Trace JSON</a>'
                    f'<a href="{artifact_link(profile["overview"])}" download>SVG 概要</a></div></section>'
                )
            else:
                profiled, artifacts = "—", "—"
            rows.append(
                f'<tr><td>{case["workers"]}</td><td class="num">{case["mean_seconds"]:.3f} ± {case.get("std_seconds", 0):.3f}</td>'
                f'<td class="num">{case["mean_steps_per_second"]:.1f} ± {rate_std:.1f}</td><td class="num">{speedup}</td>'
                f'<td class="num">{len(runs)}</td><td class="num">{profiled}</td><td>{artifacts}</td></tr>'
            )
        tables.append(
            f'<section class="detail-card measurement-table"><h2>{esc(mode)} の実測値</h2><div class="table-scroll"><table><thead><tr><th>workers</th><th>通常時間 (s)</th><th>通常 throughput (steps/s)</th><th>高速化</th><th>測定回数</th><th>計測あり時間</th><th>詳細</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>'
        )
    hardware_text = " · ".join(
        str(hardware[key]) for key in ("gpu", "cpu") if key in hardware
    )
    subtitle = f"{hardware_text} · 1 host / 1 GPU"
    settings_text = f"global batch {settings.get('train_batch_size', '—')} · minibatch {settings.get('minibatch_size', '—')} · {settings.get('num_epochs', '—')} optimization epochs · warmup {settings.get('warmup', '—')}"
    base_css = _HTML_TEMPLATE.split("<style>", 1)[1].split("</style>", 1)[0]
    content = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU 学習のプロファイルと worker 数の比較</title><style>{base_css}
    a{{color:#1d4ed8;text-decoration:none}}a:hover{{text-decoration:underline}}.charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.metric-controls{{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 12px}}.metric-controls button[aria-pressed=true]{{background:#2563eb;color:#fff;border-color:#2563eb}}.measurement-table{{margin-top:16px}}.table-scroll{{overflow-x:auto}}.measurement-table table{{min-width:820px}}.muted{{color:var(--muted);font-size:11px}}.profiles{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:16px}}.profile-card img{{display:block;width:100%;margin:12px 0;border:1px solid var(--line);border-radius:5px}}.artifact-links{{display:flex;gap:15px;align-items:center;flex-wrap:wrap;margin-top:14px;font-size:12px}}.primary-link{{font-weight:600}}.lead-note{{background:#eff6ff;border:1px solid #d3e4ff;border-radius:8px;padding:12px 15px;margin-bottom:16px;color:#35506e}}.section-title{{font-size:19px;margin:25px 0 8px}}@media(max-width:800px){{.charts,.profiles{{grid-template-columns:1fr}}}}
    </style></head><body><header><h1>GPU 学習のプロファイルと worker 数の比較</h1><p>{esc(subtitle)}</p><p style="font-size:12px;margin-top:5px">{esc(settings_text)}</p></header><main>
    <p class="lead-note">通常実行の測定値で worker 数による速度の変化を比較しています。タイムラインは別に計測した 1 イテレーションの CPU・GPU・Ray worker の動作です。</p>
    <div class="metric-controls" role="group" aria-label="比較する指標"><button data-select="seconds" aria-pressed="true">イテレーション時間</button><button data-select="throughput" aria-pressed="false">throughput</button><button data-select="speedup" aria-pressed="false">1 worker に対する高速化</button></div>
    <div class="charts">{"".join(charts)}</div><p class="note">通常実行の平均 ± 標準偏差。global batch と学習設定は各 mode 内で共通です。legacy と e2e は異なる経路として比較してください。高速化の破線は worker 数に比例する理想値です。</p>
    {"".join(tables)}<p class="note">計測あり時間の括弧は通常平均との比率です。プロファイラのコストと、別イテレーションの軌跡差の両方を含みます。高速化は同じ mode の 1 worker の平均時間 / 各 worker 数の平均時間です。</p>
    <h2 class="section-title">実行タイムライン</h2><p>画像から詳細を開くと、拡大・横移動・worker や処理の絞り込みができます。JSON は Perfetto または Chrome Tracing に読み込めます。</p><div class="profiles">{"".join(profile_cards)}</div>
    <p class="note">GPU の割合は PPO 測定区間内で記録した kernel・copy の実行時間の重なりを除いた値です。ハードウェア使用率ではありません。この測定は単一ホストでの throughput 確認です。大規模分散学習や論文の性能値の再現は測定対象に含みません。</p>
    <details class="detail-card" style="margin-top:16px"><summary>測定条件・環境</summary><pre>{esc(json.dumps({"settings": settings, "hardware": hardware, "vocab_size": report.get("vocab_size"), "interpretation": report.get("interpretation")}, ensure_ascii=False, indent=2))}</pre></details>
    <p class="note"><a href="{artifact_link(scaling_json_path)}" download>全測定結果 scaling.json</a></p></main><script>
    for(const button of document.querySelectorAll('[data-select]'))button.addEventListener('click',()=>{{const selected=button.dataset.select;for(const chart of document.querySelectorAll('[data-metric]'))chart.hidden=chart.dataset.metric!==selected;for(const control of document.querySelectorAll('[data-select]'))control.setAttribute('aria-pressed',String(control.dataset.select===selected))}});
    </script></body></html>'''
    output.write_text(content, encoding="utf-8")
    return output


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPU 学習・実行プロファイル</title>
<style>
:root{--ink:#203149;--muted:#65748a;--line:#dce4ee;--blue:#2563eb;--bg:#f5f7fb}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Noto Sans JP",sans-serif}
header{padding:24px 28px 16px;background:#fff;border-bottom:1px solid var(--line)}h1{font-size:24px;margin:0 0 4px;letter-spacing:-.03em}p{margin:0;color:var(--muted)}
main{padding:18px 28px 28px;max-width:2000px;margin:auto}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:16px}.stat{background:#fff;border:1px solid var(--line);border-radius:9px;padding:12px 16px}.stat b{display:block;font-size:24px;font-variant-numeric:tabular-nums}.stat span{font-size:12px;color:var(--muted)}
.panel{background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}.toolbar{display:flex;gap:9px;align-items:center;padding:12px 14px;flex-wrap:wrap;border-bottom:1px solid var(--line)}button,select,input[type=search]{font:inherit;border:1px solid #cbd5e1;border-radius:6px;background:#fff;color:var(--ink);padding:6px 10px}button{cursor:pointer}button:hover{background:#eff6ff;border-color:#93b4fa}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--blue);outline-offset:2px}label{display:inline-flex;align-items:center;gap:5px;cursor:pointer;white-space:nowrap}input[type=checkbox]{accent-color:var(--blue)}#search{min-width:220px;flex:1}.range-label{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12px;min-width:170px}
.filters{padding:10px 14px;border-bottom:1px solid var(--line)}.filters summary{cursor:pointer;font-size:13px;color:var(--muted)}.chips{display:flex;flex-wrap:wrap;gap:7px;padding-top:8px}.chip{border:1px solid var(--line);border-radius:5px;padding:3px 8px;font-size:12px}.swatch{width:9px;height:9px;display:inline-block;border-radius:2px}.filter-group{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.filter-group strong{font-size:12px;min-width:80px}#processes .chip{max-width:340px;overflow:hidden;text-overflow:ellipsis}
.overview-wrap{padding:10px 14px 5px;border-bottom:1px solid var(--line)}#overview{width:100%;height:70px;display:block;cursor:crosshair}.slider-wrap{display:flex;align-items:center;gap:12px;font-size:12px;color:var(--muted);padding-top:4px}#position{flex:1;accent-color:var(--blue)}
#timeline-scroll{max-height:69vh;min-height:240px;overflow-y:auto;position:relative}#timeline{display:block;width:100%;cursor:grab}#timeline.dragging{cursor:grabbing}.hint{padding:9px 14px;font-size:12px;color:var(--muted);border-top:1px solid var(--line)}
.lower{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.detail-card{background:#fff;border:1px solid var(--line);border-radius:9px;padding:14px 16px;min-width:0}.detail-card h2{font-size:15px;margin:0 0 8px}.detail-card p{font-size:12px}.detail-card pre{white-space:pre-wrap;word-break:break-word;font-size:12px;margin:10px 0 0;max-height:240px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:5px 4px;border-bottom:1px solid #edf0f5}th{font-weight:600;color:var(--muted)}td.num{text-align:right;font-variant-numeric:tabular-nums}td.event-name{max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}#tooltip{display:none;position:fixed;z-index:10;max-width:430px;max-height:300px;overflow:hidden;background:#17263c;color:#fff;border-radius:8px;padding:10px 12px;pointer-events:none;box-shadow:0 5px 22px #0003;font-size:12px}#tooltip b{display:block;margin-bottom:4px;font-size:13px;word-break:break-word}#tooltip pre{font:11px/1.5 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;margin:6px 0 0;color:#cbd5e1}.note{font-size:12px;margin-top:12px}.empty{padding:25px;text-align:center;color:var(--muted)}
#measured-summary{margin:0 0 16px}.bar-row{display:grid;grid-template-columns:minmax(100px,150px) 1fr 100px;align-items:center;gap:8px;margin:8px 0;font-size:12px}.bar-name{overflow:hidden;white-space:nowrap;text-overflow:ellipsis}.bar-track{height:18px;background:#edf2f8;border-radius:3px;display:flex;overflow:hidden}.bar-segment{height:100%;min-width:0}.bar-number{text-align:right;font-variant-numeric:tabular-nums}.legend{display:flex;flex-wrap:wrap;gap:9px;font-size:11px;color:var(--muted);margin:10px 0}.legend span{display:flex;align-items:center;gap:4px}
@media(max-width:800px){header{padding:18px}main{padding:12px}.stats{grid-template-columns:1fr 1fr}.lower{grid-template-columns:1fr}.stat b{font-size:20px}#search{min-width:140px}.toolbar{gap:6px}.range-label{min-width:130px}}
</style>
</head>
<body>
<header><h1>GPU 学習・実行プロファイル</h1><p id="subtitle"></p></header>
<main>
<div class="stats"><div class="stat"><b id="total"></b><span>実測トレース区間</span></div><div class="stat"><b id="gpu"></b><span>GPU kernel / copy の実行区間</span></div><div class="stat"><b id="count"></b><span>記録した処理区間</span></div><div class="stat"><b id="workers"></b><span>記録したプロセス</span></div></div>
<div class="lower" id="measured-summary" hidden><section class="detail-card"><h2>学習イテレーションの内訳</h2><p id="phase-help"></p><div id="phase-bars"></div></section><section class="detail-card"><h2>worker の処理時間と内訳</h2><p id="worker-help"></p><div id="worker-bars"></div><div id="worker-legend" class="legend"></div></section></div>
<div class="panel">
<div class="toolbar"><button id="reset">全体に戻す</button><button id="zoom-in" aria-label="拡大">＋ 拡大</button><button id="zoom-out" aria-label="縮小">− 縮小</button><span id="range" class="range-label"></span><input id="search" type="search" placeholder="処理名で絞り込み" aria-label="処理名で絞り込み"><label><input id="details" type="checkbox">演算・化学処理の詳細</label></div>
<details class="filters" open><summary>表示するプロセスとカテゴリ</summary><div class="filter-group"><strong>プロセス</strong><div id="processes" class="chips"></div></div><div class="filter-group"><strong>カテゴリ</strong><div id="categories" class="chips"></div></div></details>
<div class="overview-wrap"><canvas id="overview" aria-label="全実行区間の概要"></canvas><div class="slider-wrap"><span>表示位置</span><input id="position" type="range" min="0" max="1000" value="0" aria-label="表示位置"><span id="visible-count"></span></div></div>
<div id="timeline-scroll"><canvas id="timeline" role="img" aria-label="実測処理のタイムライン"></canvas></div>
<div class="hint">ホイールで拡大・縮小 ／ ドラッグで横移動 ／ 上の概要をクリックして移動 ／ 処理にカーソルを合わせて詳細、クリックで固定</div>
</div>
<div class="lower"><section class="detail-card"><h2>選択した処理</h2><p id="selection-help">タイムラインの区間をクリックすると実測時間と引数を表示します。</p><pre id="selection"></pre></section><section class="detail-card"><h2>表示カテゴリの累積時間</h2><p>表示範囲とフィルタを反映。入れ子や並列区間を含むため、合計は経過時間を超える場合があります。</p><table><thead><tr><th>カテゴリ</th><th>区間数</th><th style="text-align:right">累積</th></tr></thead><tbody id="totals"></tbody></table></section></div>
<p class="note">GPU 実行区間は、記録された kernel・copy の重なりを除いた時間です。ハードウェアの使用率や速度比較を表す値ではありません。CPU、GPU、Ray worker の時間軸は同じ基準で表示します。概要では入れ子を最大 4 段、詳細では最大 8 段に表示します。</p>
<details class="detail-card" style="margin-top:14px"><summary>測定条件・メタデータ</summary><pre id="metadata"></pre></details>
</main><div id="tooltip"></div>
<script type="application/json" id="trace-data">__TRACE_DATA__</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById('trace-data').textContent);
const $=id=>document.getElementById(id);
const palette=new Map(D.categories.map(c=>[c.name,c.color]));
const selectedProcesses=new Set(D.processes.map(p=>p.id));
const selectedCategories=new Set(D.categories.filter(c=>!c.detail).map(c=>c.name));
const processById=new Map(D.processes.map(p=>[p.id,p]));
const timeline=$('timeline'),overview=$('overview'),tooltip=$('tooltip');
let viewStart=D.start,viewDuration=D.duration,lanes=[],hits=[],drawing=false,drag=null,lastHover=null,pinned=null,needle='',details=false;
let timelineWidth=1000,timelineHeight=300,labelWidth=245,overviewWidth=1000;
let selectedEvents=[],layoutVersion=0;
const fmt=(us,precision=2)=>Math.abs(us)>=1e6?(us/1e6).toFixed(precision)+' s':Math.abs(us)>=1000?(us/1000).toFixed(precision)+' ms':us.toFixed(precision)+' µs';
const rel=ts=>((ts-D.start)/1e6).toFixed(3)+' s';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const countFmt=n=>n.toLocaleString('ja-JP');
$('subtitle').textContent=D.source+' · 実測データを埋め込んだオフラインビューア';
$('total').textContent=fmt(D.duration,3);$('gpu').textContent=fmt(D.gpuBusy,3)+' · '+(100*D.gpuBusy/D.duration).toFixed(1)+'%';
$('count').textContent=countFmt(D.events.length);$('workers').textContent=D.processes.length;
$('metadata').textContent=JSON.stringify({trace:D.traceMetadata,measurement:D.summary},null,2);

function summaryBar(parent,name,number,segments){
 const row=document.createElement('div');row.className='bar-row';const label=document.createElement('span');label.className='bar-name';label.textContent=name;label.title=name;const track=document.createElement('div');track.className='bar-track';
 for(const segment of segments){const bar=document.createElement('span');bar.className='bar-segment';bar.style.width=clamp(segment.percent,0,100)+'%';bar.style.background=palette.get(segment.category)||'#64748b';bar.title=segment.title;track.append(bar)}
 const value=document.createElement('span');value.className='bar-number';value.textContent=number;row.append(label,track,value);parent.append(row);
}
if(D.summary&&Number.isFinite(D.summary.iteration_wall_ms)){
 const S=D.summary,wall=S.iteration_wall_ms;$('measured-summary').hidden=false;
 $('phase-help').textContent='実測 PPO 区間 '+fmt(wall*1000,3)+'。棒の横幅は経過時間に対する割合。';
 for(const phase of S.driver_phases||[])summaryBar($('phase-bars'),phase.name,fmt(phase.duration_ms*1000,3),[{category:phase.category,percent:phase.wall_percent,title:phase.name+' · '+fmt(phase.duration_ms*1000,3)+' · '+phase.wall_percent.toFixed(1)+'%'}]);
 const workers=S.workers||[],maximum=Math.max(1,...workers.map(w=>w.sample_ms));const balance=S.worker_balance||{};
 const fragment=workers.some(w=>w.sample_scope_kind==='fragment'),scope=fragment?'記録した fragment (SyncSampler.next)':'sample';
 $('worker-help').textContent='各棒は '+scope+' 区間内の排他的な処理時間。worker は並列に実行。'+(Number.isFinite(balance.completion_spread_ms)?(fragment?'fragment ':'')+'完了時刻の差 '+fmt(balance.completion_spread_ms*1000,3)+'。':'');
 const categories=new Set();for(const worker of workers){const segments=(worker.exclusive_categories||[]).map(category=>{categories.add(category.category);return {category:category.category,percent:category.self_ms/maximum*100,title:category.category+' · '+fmt(category.self_ms*1000,3)}});summaryBar($('worker-bars'),'worker '+worker.worker_id,fmt(worker.sample_ms*1000,3),segments)}
 for(const category of categories){const item=document.createElement('span'),swatch=document.createElement('i');swatch.className='swatch';swatch.style.background=palette.get(category)||'#64748b';item.append(swatch,document.createTextNode(category));$('worker-legend').append(item)}
}

function makeChip(parent,text,color,checked,callback,title){
 const label=document.createElement('label');label.className='chip';if(title)label.title=title;
 const input=document.createElement('input');input.type='checkbox';input.checked=checked;input.addEventListener('change',()=>callback(input.checked));label.append(input);
 if(color){const swatch=document.createElement('i');swatch.className='swatch';swatch.style.background=color;label.append(swatch)}
 const span=document.createElement('span');span.textContent=text;label.append(span);parent.append(label);return input;
}
D.processes.forEach(p=>makeChip($('processes'),p.name,null,true,on=>{on?selectedProcesses.add(p.id):selectedProcesses.delete(p.id);rebuild()},'PID '+p.id));
const catInputs=new Map();
D.categories.forEach(c=>catInputs.set(c.name,makeChip($('categories'),c.name,c.color,selectedCategories.has(c.name),on=>{on?selectedCategories.add(c.name):selectedCategories.delete(c.name);rebuild()})));

function layoutEvents(events,maxTracks){
 const ends=[],out=[];
 for(const e of events){let track=ends.findIndex(end=>end<=e.s);if(track<0){track=ends.length;ends.push(0)}ends[track]=e.s+e.d;out.push({e,track:Math.min(track,maxTracks-1)})}
 return {items:out,tracks:Math.min(maxTracks,Math.max(1,ends.length))};
}
function rebuild(){
 selectedEvents=D.events.filter(e=>selectedProcesses.has(e.p)&&selectedCategories.has(e.c)&&(!needle||e.n.toLowerCase().includes(needle)));
 const groups=new Map();for(const e of selectedEvents){const key=e.p+'\0'+e.t;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(e)}
 lanes=[];let y=36;
 for(const p of D.processes){if(!selectedProcesses.has(p.id))continue;
  let hasHeading=false;
  for(const t of p.threads){const group=groups.get(p.id+'\0'+t.id);if(!group||!group.length)continue;
   if(!hasHeading){lanes.push({heading:true,p,y,h:25});y+=25;hasHeading=true}
   const result=layoutEvents(group,details?8:4);const h=result.tracks*21+7;
   lanes.push({p,t,y,h,items:result.items});y+=h;
  }
  const counterGroups=new Map();for(const c of D.counters){if(c.p!==p.id)continue;for(const [key,value]of Object.entries(c.a)){const name=c.n+' / '+key;if(!counterGroups.has(name))counterGroups.set(name,[]);counterGroups.get(name).push({s:c.s,value,name})}}
  for(const [name,values]of counterGroups){if(needle&&!name.toLowerCase().includes(needle))continue;if(!hasHeading){lanes.push({heading:true,p,y,h:25});y+=25;hasHeading=true}lanes.push({counter:true,p,name,values,y,h:48});y+=48}
 }
 timelineHeight=Math.max(240,y+16);layoutVersion++;requestDraw();
}
function measureCanvas(canvas,width,height){const ratio=Math.min(window.devicePixelRatio||1,2);if(canvas.width!==Math.round(width*ratio)||canvas.height!==Math.round(height*ratio)){canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);canvas.style.height=height+'px'}const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);return ctx}
function requestDraw(){if(drawing)return;drawing=true;requestAnimationFrame(()=>{drawing=false;draw()})}
function fitView(){viewDuration=clamp(viewDuration,Math.min(1,D.duration),D.duration);viewStart=clamp(viewStart,D.start,D.end-viewDuration)}
function niceStep(duration,width){const target=duration/Math.max(2,width/120);const power=10**Math.floor(Math.log10(target));for(const factor of [1,2,5,10])if(power*factor>=target)return power*factor;return power*10}
function ellipsis(ctx,text,width){if(width<=12)return '';if(ctx.measureText(text).width<=width)return text;let low=0,high=text.length;while(low<high){const mid=Math.ceil((low+high)/2);if(ctx.measureText(text.slice(0,mid)+'…').width<=width)low=mid;else high=mid-1}return text.slice(0,low)+'…'}
function draw(){
 fitView();timelineWidth=timeline.parentElement.clientWidth;labelWidth=Math.min(265,Math.max(135,timelineWidth*.24));const plotWidth=Math.max(1,timelineWidth-labelWidth-12);
 const ctx=measureCanvas(timeline,timelineWidth,timelineHeight);ctx.clearRect(0,0,timelineWidth,timelineHeight);ctx.fillStyle='#fff';ctx.fillRect(0,0,timelineWidth,timelineHeight);hits=[];
 ctx.font='11px system-ui';ctx.fillStyle='#64748b';ctx.fillText('プロセス / スレッド',12,23);
 const step=niceStep(viewDuration,plotWidth);const first=Math.ceil((viewStart-D.start)/step)*step+D.start;
 for(let ts=first;ts<=viewStart+viewDuration;ts+=step){const x=labelWidth+(ts-viewStart)/viewDuration*plotWidth;ctx.strokeStyle='#e5eaf1';ctx.beginPath();ctx.moveTo(x,34);ctx.lineTo(x,timelineHeight);ctx.stroke();ctx.fillStyle='#64748b';ctx.fillText(rel(ts),Math.min(timelineWidth-65,x+3),23)}
 const totals=new Map();let visible=0;
 for(const lane of lanes){
  if(lane.heading){ctx.fillStyle='#edf2f8';ctx.fillRect(0,lane.y,timelineWidth,lane.h);ctx.font='bold 12px system-ui';ctx.fillStyle='#253b56';ctx.fillText(ellipsis(ctx,lane.p.name+(lane.p.gpu?' · GPU':''),timelineWidth-28),12,lane.y+17);continue}
  ctx.fillStyle='#f7f9fc';ctx.fillRect(0,lane.y,labelWidth-4,lane.h);ctx.strokeStyle='#edf0f5';ctx.beginPath();ctx.moveTo(0,lane.y+lane.h);ctx.lineTo(timelineWidth,lane.y+lane.h);ctx.stroke();ctx.font='11px system-ui';ctx.fillStyle='#5b6d85';
  const label=lane.counter?lane.name:lane.t.name;ctx.fillText(ellipsis(ctx,label,labelWidth-22),12,lane.y+17);
  if(lane.counter){
   const values=lane.values.filter(v=>v.s>=viewStart&&v.s<=viewStart+viewDuration);const previous=lane.values.findLast(v=>v.s<viewStart);if(previous)values.unshift({...previous,s:viewStart});const next=lane.values.find(v=>v.s>viewStart+viewDuration);if(next)values.push({...next,s:viewStart+viewDuration});
   const maximum=Math.max(1,...lane.values.map(v=>v.value)),minimum=Math.min(0,...lane.values.map(v=>v.value));ctx.save();ctx.beginPath();ctx.rect(labelWidth,lane.y,plotWidth,lane.h);ctx.clip();ctx.strokeStyle='#0891b2';ctx.lineWidth=1.5;ctx.beginPath();values.forEach((v,i)=>{const x=labelWidth+(v.s-viewStart)/viewDuration*plotWidth,y=lane.y+lane.h-7-(v.value-minimum)/(maximum-minimum)*(lane.h-15);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.restore();ctx.fillStyle='#0891b2';ctx.fillText('max '+maximum.toLocaleString('ja-JP'),labelWidth+6,lane.y+13);continue;
  }
  ctx.save();ctx.beginPath();ctx.rect(labelWidth,lane.y,plotWidth,lane.h);ctx.clip();
  for(const item of lane.items){const e=item.e,end=e.s+e.d;if(end<viewStart||e.s>viewStart+viewDuration)continue;
   visible++;const clipped=Math.max(0,Math.min(end,viewStart+viewDuration)-Math.max(e.s,viewStart));const total=totals.get(e.c)||{duration:0,count:0};total.duration+=clipped;total.count++;totals.set(e.c,total);
   const rawX=labelWidth+(e.s-viewStart)/viewDuration*plotWidth;const x=Math.max(labelWidth,rawX),right=Math.min(timelineWidth-12,labelWidth+(end-viewStart)/viewDuration*plotWidth);const w=Math.max(.9,right-x);const y=lane.y+item.track*21+3;
   ctx.fillStyle=palette.get(e.c)||'#64748b';ctx.fillRect(x,y,w,18);
   if(w>42){ctx.fillStyle='#fff';ctx.font='11px system-ui';const label=ellipsis(ctx,e.n,w-8);ctx.fillText(label,x+4,y+13)}
   if(pinned===e){ctx.strokeStyle='#111827';ctx.lineWidth=2;ctx.strokeRect(x,y,w,18);ctx.lineWidth=1}
   hits.push({x,y,w,h:18,e,lane});
  }ctx.restore();
 }
 if(!lanes.length){ctx.fillStyle='#64748b';ctx.font='14px system-ui';ctx.fillText('フィルタに一致する処理がありません。',labelWidth+20,100)}
 $('range').textContent=rel(viewStart)+' – '+rel(viewStart+viewDuration);$('position').value=D.duration>viewDuration?Math.round((viewStart-D.start)/(D.duration-viewDuration)*1000):0;$('position').disabled=viewDuration>=D.duration;
 $('visible-count').textContent=countFmt(visible)+' 区間を表示';
 const body=$('totals');body.replaceChildren();Array.from(totals).sort((a,b)=>b[1].duration-a[1].duration).slice(0,15).forEach(([name,value])=>{const row=document.createElement('tr');const n=document.createElement('td');n.textContent=name;n.style.color=palette.get(name);const c=document.createElement('td');c.textContent=countFmt(value.count);const d=document.createElement('td');d.className='num';d.textContent=fmt(value.duration,3);row.append(n,c,d);body.append(row)});
 drawOverview();
}
function drawOverview(){overviewWidth=overview.clientWidth;const ctx=measureCanvas(overview,overviewWidth,70);ctx.fillStyle='#f1f5fa';ctx.fillRect(0,0,overviewWidth,70);
 const groups=[{gpu:false,y:8,h:20},{gpu:true,y:37,h:20}];for(const group of groups){ctx.save();ctx.beginPath();ctx.rect(0,group.y,overviewWidth,group.h);ctx.clip();for(const e of selectedEvents){if(e.g!==group.gpu)continue;const x=(e.s-D.start)/D.duration*overviewWidth,w=Math.max(.8,e.d/D.duration*overviewWidth);ctx.fillStyle=palette.get(e.c)||'#64748b';ctx.globalAlpha=group.gpu?.75:.45;ctx.fillRect(x,group.y,w,group.h)}ctx.restore();ctx.globalAlpha=1}
 const x=(viewStart-D.start)/D.duration*overviewWidth,w=viewDuration/D.duration*overviewWidth;ctx.fillStyle='#fff9';ctx.fillRect(0,0,x,70);ctx.fillRect(x+w,0,overviewWidth-x-w,70);ctx.strokeStyle='#2563eb';ctx.lineWidth=2;ctx.strokeRect(x+1,2,Math.max(1,w-2),66);ctx.fillStyle='#243b55';ctx.font='bold 10px system-ui';ctx.fillText('CPU',5,21);ctx.fillText('GPU',5,50);
}
function eventAt(ev){const rect=timeline.getBoundingClientRect(),x=ev.clientX-rect.left,y=ev.clientY-rect.top;for(let i=hits.length-1;i>=0;i--){const hit=hits[i];if(x>=hit.x&&x<=hit.x+Math.max(hit.w,3)&&y>=hit.y&&y<=hit.y+hit.h)return hit}return null}
function eventDescription(hit){const e=hit.e;return {name:e.n,category:e.c,start_seconds:(e.s-D.start)/1e6,duration_ms:e.d/1000,process:hit.lane.p.name,pid:e.p,thread:hit.lane.t.name,tid:e.t,args:e.a}}
function hover(ev){if(drag)return;const hit=eventAt(ev);lastHover=hit;if(!hit){tooltip.style.display='none';return}tooltip.replaceChildren();const b=document.createElement('b');b.textContent=hit.e.n;const text=document.createElement('div');text.textContent=fmt(hit.e.d,3)+' · '+rel(hit.e.s)+' · '+hit.e.c;const worker=document.createElement('div');worker.textContent=hit.lane.p.name+' / '+hit.lane.t.name;tooltip.append(b,text,worker);if(Object.keys(hit.e.a).length){const pre=document.createElement('pre');let args=JSON.stringify(hit.e.a,null,2);pre.textContent=args.length>1200?args.slice(0,1200)+'…':args;tooltip.append(pre)}tooltip.style.display='block';const width=tooltip.offsetWidth,height=tooltip.offsetHeight;tooltip.style.left=clamp(ev.clientX+14,8,window.innerWidth-width-8)+'px';tooltip.style.top=clamp(ev.clientY+14,8,window.innerHeight-height-8)+'px'}
function zoom(factor,anchor=.5){const timestamp=viewStart+viewDuration*anchor;viewDuration=clamp(viewDuration*factor,Math.min(1,D.duration),D.duration);viewStart=timestamp-viewDuration*anchor;fitView();requestDraw();tooltip.style.display='none'}
$('reset').addEventListener('click',()=>{viewStart=D.start;viewDuration=D.duration;requestDraw()});$('zoom-in').addEventListener('click',()=>zoom(.5));$('zoom-out').addEventListener('click',()=>zoom(2));
$('details').addEventListener('change',ev=>{details=ev.target.checked;for(const c of D.categories){if(!c.detail)continue;details?selectedCategories.add(c.name):selectedCategories.delete(c.name);catInputs.get(c.name).checked=details}rebuild()});
$('position').addEventListener('input',ev=>{viewStart=D.start+(D.duration-viewDuration)*Number(ev.target.value)/1000;requestDraw()});
let debounce;$('search').addEventListener('input',ev=>{clearTimeout(debounce);debounce=setTimeout(()=>{needle=ev.target.value.trim().toLowerCase();rebuild()},150)});
timeline.addEventListener('wheel',ev=>{ev.preventDefault();const rect=timeline.getBoundingClientRect(),anchor=clamp((ev.clientX-rect.left-labelWidth)/(timelineWidth-labelWidth-12),0,1);zoom(Math.exp(ev.deltaY*.002),anchor)},{passive:false});
timeline.addEventListener('pointerdown',ev=>{if(ev.button!==0)return;drag={x:ev.clientX,start:viewStart,moved:false,hit:eventAt(ev)};timeline.setPointerCapture(ev.pointerId);timeline.classList.add('dragging');tooltip.style.display='none'});
timeline.addEventListener('pointermove',ev=>{if(!drag){hover(ev);return}const delta=ev.clientX-drag.x;if(Math.abs(delta)>3)drag.moved=true;viewStart=drag.start-delta/(timelineWidth-labelWidth-12)*viewDuration;fitView();requestDraw()});
timeline.addEventListener('pointerup',ev=>{if(!drag)return;if(!drag.moved&&drag.hit){pinned=drag.hit.e;$('selection-help').textContent=drag.hit.lane.p.name+' / '+drag.hit.lane.t.name;$('selection').textContent=JSON.stringify(eventDescription(drag.hit),null,2)}drag=null;timeline.classList.remove('dragging');timeline.releasePointerCapture(ev.pointerId);requestDraw()});
timeline.addEventListener('pointercancel',()=>{drag=null;timeline.classList.remove('dragging')});timeline.addEventListener('pointerleave',()=>{tooltip.style.display='none'});
overview.addEventListener('pointerdown',ev=>{const rect=overview.getBoundingClientRect();viewStart=D.start+clamp((ev.clientX-rect.left)/overviewWidth,0,1)*D.duration-viewDuration/2;fitView();requestDraw()});
new ResizeObserver(requestDraw).observe(timeline.parentElement);rebuild();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", type=Path, required=True, help="Chrome Trace JSON (or .json.gz)"
    )
    parser.add_argument(
        "--html", type=Path, required=True, help="Self-contained interactive HTML"
    )
    parser.add_argument("--svg", type=Path, help="Optional static overview SVG")
    parser.add_argument(
        "--summary", type=Path, help="Optional measurement metadata JSON"
    )
    args = parser.parse_args()
    print(build_viewer(args.trace, args.html, args.summary))
    if args.svg:
        print(build_overview(args.trace, args.svg, args.summary))


if __name__ == "__main__":
    main()
