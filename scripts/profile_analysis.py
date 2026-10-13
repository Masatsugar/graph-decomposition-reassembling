"""Merge local Chrome traces and summarize the measured training iteration.

Only the logical scopes supply exclusive CPU wall time. PyTorch operator and
GPU events overlap those scopes, so they are analyzed separately. This module
has no Torch or Ray dependency and can inspect exported traces offline.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any

_GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _identity(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def merge_trace(
    custom_events: list[dict] | dict,
    torch_trace_dict: dict | None,
    start_epoch_ns: int,
) -> dict:
    """Align a PyTorch trace with local recorder events and avoid lane collisions.

    Recorder timestamps are already microseconds from ``start_epoch_ns``.
    Kineto exports microseconds from ``baseTimeNanoseconds``; subtract the
    integer epoch origins before division to retain timestamp precision.
    This clock alignment is valid for the single host used by the recorder.

    ``custom_events`` can be a bare event list or a recorder payload. A payload
    retains its extra metadata, including any recorded event-drop counts.
    Inputs are never modified. Passing ``None`` skips the PyTorch trace.
    """
    if not _finite_number(start_epoch_ns) or start_epoch_ns <= 0:
        raise ValueError("start_epoch_ns must be a positive finite epoch timestamp")
    if int(start_epoch_ns) != start_epoch_ns:
        raise ValueError("start_epoch_ns must contain whole nanoseconds")
    if isinstance(custom_events, dict):
        result = deepcopy(custom_events)
        custom = result.pop("traceEvents", [])
    else:
        result = {}
        custom = deepcopy(custom_events)
    if not isinstance(custom, list) or any(not isinstance(e, dict) for e in custom):
        raise ValueError("custom_events must contain a list of event dictionaries")
    for event in custom:
        event.setdefault("args", {})["source_type"] = "custom"

    merged = custom
    if torch_trace_dict is not None:
        torch_events = torch_trace_dict.get("traceEvents", [])
        if not isinstance(torch_events, list):
            raise ValueError("PyTorch traceEvents must be a list")
        if torch_events:
            base = torch_trace_dict.get("baseTimeNanoseconds")
            if not _finite_number(base) or base <= 0 or int(base) != base:
                raise ValueError(
                    "PyTorch trace requires a valid baseTimeNanoseconds to align clocks"
                )
            offset_us = (int(base) - int(start_epoch_ns)) / 1_000
            pids = {
                _identity(event.get("pid", 0)): event.get("pid", 0)
                for event in torch_events
            }
            used_pids = {
                event["pid"]
                for event in custom
                if isinstance(event.get("pid"), int)
            }
            first_pid = max([0, *used_pids]) + 1_000
            pid_map = {key: first_pid + i for i, key in enumerate(pids)}
            tid_map: dict[tuple[str, str], int] = {}
            flow_map: dict[str, str] = {}
            used_flows = set()
            for event in custom:
                for key in ("id", "bind_id"):
                    if key in event:
                        used_flows.add(_identity(event[key]))
                used_flows.update(
                    _identity(value) for value in event.get("id2", {}).values()
                )

            def flow_id(value):
                key = _identity(value)
                if key not in flow_map:
                    candidate = f"torch:{key}"
                    while _identity(candidate) in used_flows:
                        candidate = f"torch:{candidate}"
                    used_flows.add(_identity(candidate))
                    flow_map[key] = candidate
                return flow_map[key]

            gpu_pids = {
                _identity(event.get("pid", 0))
                for event in torch_events
                if event.get("cat") in _GPU_CATEGORIES
            }
            named_pids = set()
            for original in torch_events:
                event = deepcopy(original)
                original_pid = event.get("pid", 0)
                pid_key = _identity(original_pid)
                event["pid"] = pid_map[pid_key]
                args = event.setdefault("args", {})
                args.update(source_type="torch", original_pid=original_pid)
                if "tid" in event:
                    original_tid = event["tid"]
                    tid_key = (pid_key, _identity(original_tid))
                    if tid_key not in tid_map:
                        tid_map[tid_key] = len(tid_map) + 1
                    event["tid"] = tid_map[tid_key]
                    args["original_tid"] = original_tid
                if "ts" in event:
                    if not _finite_number(event["ts"]):
                        raise ValueError("PyTorch trace contains an invalid timestamp")
                    event["ts"] += offset_us
                for key in ("id", "bind_id"):
                    if key in event:
                        event[key] = flow_id(event[key])
                if "id2" in event:
                    event["id2"] = {
                        key: flow_id(value) for key, value in event["id2"].items()
                    }
                if event.get("ph") == "M" and event.get("name") == "process_name":
                    lane = "GPU" if pid_key in gpu_pids else "CPU"
                    args["name"] = f"PyTorch {lane}: {args.get('name', original_pid)}"
                    named_pids.add(pid_key)
                merged.append(event)
            for pid_key, original_pid in pids.items():
                if pid_key not in named_pids:
                    lane = "GPU" if pid_key in gpu_pids else "CPU"
                    merged.append(
                        {
                            "ph": "M",
                            "name": "process_name",
                            "pid": pid_map[pid_key],
                            "tid": 0,
                            "args": {
                                "name": f"PyTorch {lane} ({original_pid})",
                                "source_type": "torch",
                                "original_pid": original_pid,
                            },
                        }
                    )
            result["torch_clock"] = {
                "baseTimeNanoseconds": int(base),
                "offset_us": offset_us,
            }

    # Metadata precedes timed events. Negative pre-iteration timestamps remain
    # intact so launch-to-kernel flow relationships retain their original order.
    merged.sort(
        key=lambda event: (
            0 if event.get("ph") == "M" else 1,
            event.get("ts", 0),
            -event.get("dur", 0),
        )
    )
    result.update(
        traceEvents=merged,
        displayTimeUnit="ms",
        start_epoch_ns=int(start_epoch_ns),
        clock="microseconds from start_epoch_ns; aligned on a single host",
    )
    return result


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _union(intervals))


def _range(event: dict) -> tuple[float, float]:
    return event["ts"], event["ts"] + event["dur"]


def _intersect(event: dict, windows: list[tuple[float, float]]):
    start, end = _range(event)
    return [
        (max(start, left), min(end, right))
        for left, right in windows
        if start < right and end > left
    ]


def analyze_trace(trace: dict) -> dict:
    """Summarize exact logical phases, worker balance, and traced GPU activity.

    GPU busy percentage is the union of recorded kernel/copy/set intervals
    inside the profiled iteration. It is not whole-device NVML utilization.
    Exclusive category times sum CPU wall work across recorded scopes and can
    exceed iteration wall time when several workers execute concurrently.
    """
    events = trace.get("traceEvents", [])
    complete = [
        event
        for event in events
        if event.get("ph") == "X"
        and _finite_number(event.get("ts"))
        and _finite_number(event.get("dur"))
        and event["dur"] >= 0
    ]

    def custom(event):
        return event.get("args", {}).get("source_type", "custom") == "custom"

    iterations = [
        e for e in complete if custom(e) and e.get("name") == "PPO.iteration"
    ]
    if not iterations:
        iterations = [
            e
            for e in complete
            if custom(e)
            and e.get("cat") == "iteration"
            and (
                "iteration" in e.get("name", "").lower()
                or e.get("name") == "PPO.training_step"
            )
        ]
    if not iterations:
        raise ValueError("Trace has no measured PPO.iteration or iteration scope")
    driver_pids = {e.get("pid") for e in iterations}
    windows = _union([_range(e) for e in iterations])
    wall_us = _duration(windows)
    origin_us = windows[0][0]
    selected = [e for e in complete if _intersect(e, windows)]
    names = {
        e.get("pid"): e.get("args", {}).get("name", str(e.get("pid")))
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }

    phases = defaultdict(list)
    for event in selected:
        if (
            custom(event)
            and event.get("pid") in driver_pids
            and event.get("cat") in {"sampling_wait", "optimization", "sync"}
        ):
            phases[event.get("name", "unknown")].append(event)
    driver_phases = []
    for name, spans in phases.items():
        duration_us = _duration([i for e in spans for i in _intersect(e, windows)])
        driver_phases.append(
            {
                "name": name,
                "category": spans[0].get("cat"),
                "duration_ms": duration_us / 1_000,
                "self_ms": sum(e.get("args", {}).get("self_us", 0) for e in spans)
                / 1_000,
                "count": len(spans),
                "wall_percent": 100 * duration_us / wall_us if wall_us else 0,
            }
        )
    driver_phases.sort(key=lambda phase: -phase["duration_ms"])

    sample_candidates = defaultdict(lambda: defaultdict(list))
    for event in selected:
        if custom(event) and event.get("name") in {
            "RolloutWorker.sample",
            "SyncSampler.next",
        }:
            sample_candidates[event.get("pid")][event["name"]].append(event)
    # Ray can cache actor method dispatch before class instrumentation. The
    # bound input reader still exposes a genuine fragment scope; never relabel
    # it as the actor's complete sample call or infer missing wall-time spans.
    samples = {}
    sample_scopes = {}
    for pid, candidates in sample_candidates.items():
        scope = (
            "RolloutWorker.sample"
            if candidates["RolloutWorker.sample"]
            else "SyncSampler.next"
        )
        samples[pid] = candidates[scope]
        sample_scopes[pid] = scope
    worker_rows = []
    partial_self_spans = 0
    for pid, spans in samples.items():
        label = names.get(pid, f"worker process {pid}")
        worker_match = re.search(r"(?:rollout )?worker\s+(\d+)", label)
        worker_id = int(worker_match[1]) if worker_match else str(pid)
        sample_intervals = _union([i for e in spans for i in _intersect(e, windows)])
        categories = defaultdict(lambda: {"self_us": 0.0, "count": 0})
        for event in selected:
            if not custom(event) or event.get("pid") != pid:
                continue
            clipped = _intersect(event, sample_intervals)
            if not clipped:
                continue
            self_us = event.get("args", {}).get("self_us")
            if not _finite_number(self_us) or self_us < 0:
                continue
            clipped_us = sum(end - start for start, end in clipped)
            # Self-time locations cannot be recovered from a scalar. Omit any
            # partially captured span instead of inventing a proportional split.
            if not math.isclose(clipped_us, event["dur"], abs_tol=1e-6):
                partial_self_spans += 1
                continue
            category = event.get("cat", "unknown")
            categories[category]["self_us"] += min(self_us, event["dur"])
            categories[category]["count"] += 1
        worker_rows.append(
            {
                "worker_id": worker_id,
                "pid": pid,
                "process_name": label,
                "sample_scope": sample_scopes[pid],
                "sample_scope_kind": (
                    "sample"
                    if sample_scopes[pid] == "RolloutWorker.sample"
                    else "fragment"
                ),
                "sample_scope_note": (
                    "Full RolloutWorker.sample call, before Ray serialization."
                    if sample_scopes[pid] == "RolloutWorker.sample"
                    else "Measured SyncSampler.next fragment; excludes later worker "
                    "batch concatenation, compression and Ray serialization."
                ),
                "sample_ms": _duration(sample_intervals) / 1_000,
                "sample_calls": len(spans),
                "first_sample_start_ms": (
                    min(start for start, _ in sample_intervals) - origin_us
                )
                / 1_000,
                "last_sample_end_ms": (
                    max(end for _, end in sample_intervals) - origin_us
                )
                / 1_000,
                "exclusive_categories": sorted(
                    [
                        {
                            "category": category,
                            "self_ms": values["self_us"] / 1_000,
                            "count": values["count"],
                        }
                        for category, values in categories.items()
                    ],
                    key=lambda row: -row["self_ms"],
                ),
            }
        )
    worker_rows.sort(key=lambda row: (str(row["worker_id"]).zfill(12)))

    balance = {}
    if worker_rows:
        durations = [w["sample_ms"] for w in worker_rows]
        starts = [w["first_sample_start_ms"] for w in worker_rows]
        ends = [w["last_sample_end_ms"] for w in worker_rows]
        sampling_spans = [
            e
            for e in selected
            if custom(e)
            and e.get("pid") in driver_pids
            and e.get("cat") == "sampling_wait"
        ]
        gap_ms = (
            (max(e["ts"] + e["dur"] for e in sampling_spans) - origin_us) / 1_000
            - max(ends)
            if sampling_spans
            else None
        )
        full_samples = all(w["sample_scope_kind"] == "sample" for w in worker_rows)
        balance = {
            "scope": "full sample calls" if full_samples else "traced fragment",
            "fastest_sample_ms": min(durations),
            "slowest_sample_ms": max(durations),
            "sample_duration_spread_ms": max(durations) - min(durations),
            "worker_start_spread_ms": max(starts) - min(starts),
            "completion_spread_ms": max(ends) - min(ends),
            "worker_sample_sum_ms": sum(durations),
            "post_last_traced_scope_gap_ms": gap_ms,
            "post_last_sample_gap_ms": gap_ms if full_samples else None,
            "post_last_traced_fragment_gap_ms": gap_ms if not full_samples else None,
            "post_last_traced_scope_gap_note": (
                "After the last measured worker scope: may include worker "
                "postprocessing, Ray serialization/transport and driver aggregation; "
                "this is not a direct network-time measurement."
            ),
        }

    gpu_events = [e for e in selected if e.get("cat") in _GPU_CATEGORIES]
    busy_us = _duration([i for e in gpu_events for i in _intersect(e, windows)])
    by_process = defaultdict(list)
    for event in gpu_events:
        by_process[event.get("pid")].extend(_intersect(event, windows))
    summary = {
        "iteration_wall_ms": wall_us / 1_000,
        "iteration_count": len(iterations),
        "elapsed_span_ms": (windows[-1][1] - windows[0][0]) / 1_000,
        "driver_phases": driver_phases,
        "workers": worker_rows,
        "worker_balance": balance,
        "gpu": {
            "busy_ms": busy_us / 1_000,
            "busy_percent_of_iteration": 100 * busy_us / wall_us if wall_us else 0,
            "kernel_count": sum(e.get("cat") == "kernel" for e in gpu_events),
            "memcpy_count": sum(e.get("cat") == "gpu_memcpy" for e in gpu_events),
            "memset_count": sum(e.get("cat") == "gpu_memset" for e in gpu_events),
            "busy_by_process_ms": {
                str(pid): _duration(intervals) / 1_000
                for pid, intervals in by_process.items()
            },
            "scope": "Union of traced GPU activity inside the profiled iteration; "
            "not whole-device utilization. CUDA runtime CPU calls are excluded.",
        },
        "event_count": len(events),
        "partial_exclusive_spans_omitted": partial_self_spans,
        "notes": [
            "Driver phases use fresh measured scopes, not RLlib's smoothed timers.",
            (
                "Worker scope sum is concurrent CPU wall work, not iteration wall time; "
                "sample_scope identifies a full sample call or a traced fragment."
            ),
            "Exclusive categories use args.self_us and do not add parent/child time.",
            "sync_weights measures driver publish/dispatch; remote updates are asynchronous.",
        ],
    }
    for container in (
        trace,
        trace.get("metadata", {}),
        trace.get("moldr_profile_metadata", {}),
    ):
        if isinstance(container, dict) and "dropped_events" in container:
            summary["dropped_events"] = deepcopy(container["dropped_events"])
    return summary
