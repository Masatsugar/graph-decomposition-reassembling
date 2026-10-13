import json

from scripts.profile_viewer import _prepare_trace, build_index


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
