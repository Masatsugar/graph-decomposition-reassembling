import json

from scripts.profile_analysis import analyze_trace
from scripts.profile_viewer import _prepare_trace, build_index, build_viewer


def test_viewer_gpu_busy_excludes_annotation_and_overlap(tmp_path):
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "GPU"}},
        {"ph": "M", "name": "process_name", "pid": 777, "args": {"name": "unused"}},
        {
            "ph": "X",
            "pid": 1,
            "tid": 1,
            "ts": 0,
            "dur": 1000000,
            "name": "range",
            "cat": "gpu_user_annotation",
        },
        {
            "ph": "X",
            "pid": 1,
            "tid": 1,
            "ts": 1000,
            "dur": 5000,
            "name": "kernel",
            "cat": "kernel",
        },
        {
            "ph": "X",
            "pid": 1,
            "tid": 2,
            "ts": 4000,
            "dur": 6000,
            "name": "copy",
            "cat": "gpu_memcpy",
        },
    ]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    prepared = _prepare_trace(path)
    assert prepared["gpuBusy"] == 9000
    assert len(prepared["processes"]) == 1


def test_index_links_survive_downloaded_bundle(tmp_path):
    trace_dir = tmp_path / "legacy-workers-2"
    trace_dir.mkdir()
    profile = {
        "trace": str(trace_dir / "trace.json"),
        "timeline": str(trace_dir / "timeline.html"),
        "overview": str(trace_dir / "overview.svg"),
        "summary": {},
    }
    report = {
        "cases": [
            {
                "mode": "legacy",
                "workers": 2,
                "mean_seconds": 1,
                "std_seconds": 0.1,
                "mean_steps_per_second": 1024,
                "unprofiled_runs": [{"steps_per_second": 1024}],
                "profile": profile,
            }
        ]
    }
    source = tmp_path / "scaling.json"
    source.write_text(json.dumps(report))
    output = build_index(source, tmp_path / "index.html")
    rendered = output.read_text()
    assert 'href="legacy-workers-2/timeline.html"' in rendered
    assert 'href="legacy-workers-2/trace.json"' in rendered
    assert str(tmp_path) not in rendered


def _compact_fixture():
    def event(name, category, start, duration, *, args=None, pid=10, tid=1):
        return {"ph": "X", "name": name, "cat": category, "ts": start,
                "dur": duration, "pid": pid, "tid": tid, "args": args or {}}

    return {"traceEvents": [
        event("raw_envelope", "cpu_op", -100, 1_400),
        event("PPO.iteration", "iteration", 0, 1_000, args={"source_type": "custom", "self_us": 10}),
        event("graph_encoder", "gnn", 100, 600, args={"source_type": "custom"}),
        event("candidate_values", "candidate_value", 100, 300, args={"source_type": "custom"}),
        event("logical_cpu_scope", "cpu_op", 110, 50, args={"source_type": "custom"}),
        event("raw_runtime", "cuda_runtime", 10, 20),
        {"ph": "B", "name": "raw_driver", "cat": "cuda_driver", "ts": 20, "pid": 10, "tid": 1},
        {"ph": "E", "ts": 40, "pid": 10, "tid": 1},
        event("kernel_1", "kernel", 100, 300, pid=20),
        event("kernel_2", "gpu_kernel", 200, 400, pid=20, tid=2),
        event("copy", "gpu_memcpy", 500, 200, pid=20),
        event("memset", "gpu_memset", 650, 50, pid=20),
        event("device_annotation", "gpu_user_annotation", 100, 700, pid=20),
        {"ph": "C", "name": "memory", "cat": "cpu_op", "ts": 900,
         "pid": 10, "tid": 1, "args": {"allocated": 4_096}},
    ], "moldr_profile_metadata": {"trace_separate_from_speed_measurement": True}}


def _viewer_payload(path):
    payload = path.read_text().split('<script type="application/json" id="trace-data">', 1)[1]
    return json.loads(payload.split("</script>", 1)[0])


def test_compact_view_preserves_logical_device_counters_and_full_time_envelope(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_compact_fixture()))
    full = _prepare_trace(path)
    compact = _prepare_trace(path, compact=True)
    retained = {event["n"] for event in compact["events"]}
    assert retained == {
        "PPO.iteration", "graph_encoder", "candidate_values", "logical_cpu_scope",
        "kernel_1", "kernel_2", "copy", "memset", "device_annotation",
    }
    assert compact["counters"] == full["counters"]
    assert [event for event in compact["events"] if event["g"]] == [
        event for event in full["events"] if event["g"]
    ]
    assert compact["gpuBusy"] == full["gpuBusy"] == 600
    assert (compact["start"], compact["end"], compact["duration"]) == (
        full["start"], full["end"], full["duration"]
    ) == (-100, 1_300, 1_400)
    assert compact["viewFilter"] == {
        "mode": "compact", "omittedEvents": 3,
        "omittedCategories": {"cpu_op": 1, "cuda_driver": 1, "cuda_runtime": 1},
        "sourceDurationEvents": 12, "retainedDurationEvents": 9,
    }


def test_build_compact_html_leaves_full_trace_analysis_and_default_view_unchanged(tmp_path):
    trace_dir = tmp_path / "full traces"
    trace_dir.mkdir()
    trace = trace_dir / "trace.json"
    trace.write_text(json.dumps(_compact_fixture()))
    summary = trace_dir / "analysis.json"
    summary.write_text(json.dumps(analyze_trace(_compact_fixture())))
    original = (trace.read_bytes(), summary.read_bytes())
    full = build_viewer(trace, tmp_path / "full.html", summary)
    compact = build_viewer(trace, tmp_path / "compact.html", summary, compact=True)
    full_data, compact_data = _viewer_payload(full), _viewer_payload(compact)
    assert full_data["viewFilter"]["mode"] == "full"
    assert full_data["viewFilter"]["omittedEvents"] == 0
    assert {event["n"] for event in full_data["events"]} >= {
        "raw_envelope", "raw_runtime", "raw_driver",
    }
    assert compact_data["summary"] == full_data["summary"]
    assert compact_data["fullTraceLink"] == "full%20traces/trace.json"
    assert compact_data["viewFilter"]["omittedEvents"] == 3
    assert compact_data["gpuBusy"] == full_data["gpuBusy"]
    assert 'id="trace-omissions"' in compact.read_text()
    assert "完全なトレース" in compact.read_text()
    assert (trace.read_bytes(), summary.read_bytes()) == original


def test_compact_raw_only_trace_retains_source_extent_and_omission_count(tmp_path):
    path = tmp_path / "operators.json"
    path.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "name": "aten::add", "cat": "cpu_op", "ts": 3, "dur": 10},
    ]}))
    compact = _prepare_trace(path, compact=True)
    assert compact["events"] == []
    assert compact["gpuBusy"] == 0
    assert (compact["start"], compact["end"]) == (3, 13)
    assert compact["viewFilter"]["omittedEvents"] == 1
