"""Synthetic timing checks for offline trace analysis, without Torch/Ray."""

from copy import deepcopy

import pytest

from scripts.profile_analysis import analyze_trace, merge_trace


def event(name, cat, start, duration, *, pid=10, tid=1, self_us=None):
    args = {} if self_us is None else {"self_us": self_us}
    return {
        "ph": "X", "name": name, "cat": cat, "ts": start,
        "dur": duration, "pid": pid, "tid": tid, "args": args,
    }


def test_merge_preserves_epoch_precision_and_namespaces_cpu_gpu_and_flows():
    origin = 1_798_800_000_123_456_000
    custom = {"traceEvents": [event("PPO.iteration", "iteration", 0, 5_000)],
              "dropped_events": {"driver": 2}}
    torch = {
        "baseTimeNanoseconds": origin - 1_000_000,
        "traceEvents": [
            event("aten::mm", "cpu_op", 1_010, 20, pid=10, tid=99),
            event("kernel", "kernel", 1_020, 20, pid=0, tid=99),
            {"ph": "s", "name": "flow", "ts": 1_010, "pid": 10,
             "tid": 99, "id": 7},
            {"ph": "f", "name": "flow", "ts": 1_020, "pid": 0,
             "tid": 99, "id": 7},
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 99,
             "args": {"name": "stream 99"}},
        ],
    }
    before = deepcopy((custom, torch))
    merged = merge_trace(custom, torch, origin)
    complete = [e for e in merged["traceEvents"] if e["ph"] == "X"]
    cpu = next(e for e in complete if e["name"] == "aten::mm")
    gpu = next(e for e in complete if e["cat"] == "kernel")
    assert cpu["ts"] == 10
    assert gpu["ts"] == 20
    assert len({10, cpu["pid"], gpu["pid"]}) == 3
    assert cpu["tid"] != gpu["tid"]
    flows = [e for e in merged["traceEvents"] if e["ph"] in {"s", "f"}]
    assert flows[0]["id"] == flows[1]["id"] != 7
    assert merged["dropped_events"] == {"driver": 2}
    assert (custom, torch) == before


@pytest.mark.parametrize("base", [None, 0, -1, float("nan"), "123"])
def test_merge_rejects_unalignable_torch_clocks(base):
    trace = {"traceEvents": [event("aten::mm", "cpu_op", 1, 2)]}
    if base is not None:
        trace["baseTimeNanoseconds"] = base
    with pytest.raises(ValueError, match="baseTimeNanoseconds"):
        merge_trace([], trace, 1_000)


def test_analysis_uses_worker_self_times_and_gpu_union_inside_iteration():
    trace = {
        "traceEvents": [
            event("PPO.iteration", "iteration", 0, 1_000, self_us=50),
            event("synchronous_parallel_sample", "sampling_wait", 50, 700,
                  self_us=700),
            event("multi_gpu_train_one_step", "optimization", 750, 200,
                  self_us=100),
            event("sync_weights", "sync", 950, 20, self_us=20),
            {"ph": "M", "name": "process_name", "pid": 11,
             "args": {"name": "rollout worker 1"}},
            {"ph": "M", "name": "process_name", "pid": 12,
             "args": {"name": "rollout worker 2"}},
            event("RolloutWorker.sample", "rollout", 60, 500, pid=11,
                  self_us=100),
            event("MolEnvE2E.step", "environment", 70, 400, pid=11,
                  self_us=100),
            event("merge_node", "chemistry", 80, 300, pid=11, self_us=300),
            event("RolloutWorker.sample", "rollout", 70, 650, pid=12,
                  self_us=650),
            event("RolloutWorker.set_weights", "sync", 960, 10, pid=11,
                  self_us=10),
            event("k1", "kernel", 770, 100, pid=100, tid=1),
            event("k2", "kernel", 820, 100, pid=100, tid=2),
            event("copy", "gpu_memcpy", 900, 100, pid=100, tid=3),
            event("set", "gpu_memset", 990, 30, pid=100, tid=3),
            event("cudaLaunchKernel", "cuda_runtime", 0, 1_000, pid=10),
            event("outside", "kernel", 1_100, 500, pid=100),
        ],
        "metadata": {"dropped_events": 3},
    }
    summary = analyze_trace(trace)
    assert summary["iteration_wall_ms"] == 1
    phases = {p["category"]: p for p in summary["driver_phases"]}
    assert phases["sampling_wait"]["duration_ms"] == pytest.approx(0.7)
    assert phases["sampling_wait"]["wall_percent"] == pytest.approx(70)
    worker = summary["workers"][0]
    exclusive = {r["category"]: r["self_ms"] for r in worker["exclusive_categories"]}
    assert exclusive == {"rollout": 0.1, "environment": 0.1, "chemistry": 0.3}
    assert sum(exclusive.values()) == worker["sample_ms"]
    balance = summary["worker_balance"]
    assert balance["worker_sample_sum_ms"] == pytest.approx(1.15)
    assert balance["sample_duration_spread_ms"] == pytest.approx(0.15)
    assert balance["completion_spread_ms"] == pytest.approx(0.16)
    assert balance["post_last_sample_gap_ms"] == pytest.approx(0.03)
    assert summary["gpu"]["busy_ms"] == pytest.approx(0.23)
    assert summary["gpu"]["busy_percent_of_iteration"] == pytest.approx(23)
    assert summary["gpu"]["kernel_count"] == 2
    assert summary["dropped_events"] == 3


def test_analysis_rejects_missing_iteration_and_accepts_old_stack_scope():
    with pytest.raises(ValueError, match="iteration scope"):
        analyze_trace({"traceEvents": []})
    summary = analyze_trace({"traceEvents": [
        event("PPO.training_step", "iteration", 1_000, 500),
    ]})
    assert summary["iteration_wall_ms"] == 0.5
    assert summary["gpu"]["busy_ms"] == 0


def test_analysis_fragment_fallback_does_not_claim_complete_actor_sample():
    trace = {"traceEvents": [
        event("PPO.iteration", "iteration", 0, 1_000),
        event("synchronous_parallel_sample", "sampling_wait", 0, 900),
        {"ph": "M", "name": "process_name", "pid": 11,
         "args": {"name": "rollout worker 1"}},
        event("SyncSampler.next", "rollout", 20, 700, pid=11, self_us=100),
        event("MolEnvE2E.step", "environment", 30, 600, pid=11, self_us=600),
    ]}
    summary = analyze_trace(trace)
    worker = summary["workers"][0]
    assert worker["sample_scope"] == "SyncSampler.next"
    assert worker["sample_scope_kind"] == "fragment"
    assert worker["sample_ms"] == 0.7
    assert "excludes later" in worker["sample_scope_note"]
    balance = summary["worker_balance"]
    assert balance["scope"] == "traced fragment"
    assert balance["post_last_sample_gap_ms"] is None
    assert balance["post_last_traced_fragment_gap_ms"] == pytest.approx(0.18)
    assert "postprocessing" in balance["post_last_traced_scope_gap_note"]


def test_analysis_prefers_full_sample_when_fragment_is_nested():
    trace = {"traceEvents": [
        event("PPO.iteration", "iteration", 0, 1_000),
        event("RolloutWorker.sample", "rollout", 20, 800, pid=11, self_us=100),
        event("SyncSampler.next", "rollout", 30, 700, pid=11, self_us=700),
    ]}
    worker = analyze_trace(trace)["workers"][0]
    assert worker["sample_scope"] == "RolloutWorker.sample"
    assert worker["sample_calls"] == 1
    assert worker["sample_ms"] == 0.8
    assert worker["exclusive_categories"] == [
        {"category": "rollout", "self_ms": 0.8, "count": 2}
    ]
