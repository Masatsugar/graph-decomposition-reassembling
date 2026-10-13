"""Profiling contracts without launching Ray actors or requiring a GPU."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from moldr import profiling


def _events(snapshot):
    return [event for event in snapshot["traceEvents"] if event["ph"] == "X"]


def test_nested_spans_exclude_children_and_preserve_chrome_timeline(monkeypatch):
    clock = iter([100, 101, 102, 110, 130, 150])
    monkeypatch.setattr(profiling.time, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(profiling.time, "time_ns", lambda: 1_000_000)
    recorder = profiling.TraceRecorder(900_000)
    with recorder.span("parent", "environment"), recorder.span("child", "chemistry"):
        pass
    parent, child = _events(recorder.snapshot())
    assert parent["ts"] == pytest.approx(100.002)
    assert parent["dur"] == pytest.approx(0.048)
    assert child["ts"] == pytest.approx(100.010)
    assert child["dur"] == pytest.approx(0.020)
    assert parent["args"]["self_us"] == pytest.approx(0.028)
    assert child["args"]["self_us"] == child["dur"]
    assert (parent["args"]["depth"], child["args"]["depth"]) == (0, 1)
    assert parent["pid"] == child["pid"]
    assert parent["tid"] == child["tid"]


def test_shared_epoch_aligns_two_recorders():
    epoch = time.time_ns() - 1_000_000_000
    first = profiling.TraceRecorder(epoch)
    second = profiling.TraceRecorder(epoch)
    with first.span("first"):
        pass
    with second.span("second"):
        pass
    first_event = _events(first.snapshot())[0]
    second_event = _events(second.snapshot())[0]
    assert 1_000_000 <= first_event["ts"] <= second_event["ts"]
    assert second_event["ts"] - first_event["ts"] < 100_000


def test_disabled_wrapper_calls_original_without_clock_or_scope(monkeypatch):
    recorder = profiling.TraceRecorder()
    recorder.enabled = False
    monkeypatch.setattr(profiling, "_recorder", recorder)

    def fail_clock():
        raise AssertionError("disabled profiling queried the clock")

    monkeypatch.setattr(profiling.time, "perf_counter_ns", fail_clock)
    calls = []

    def original(value):
        calls.append(value)
        return value + 1

    wrapper = profiling._wrap(original, "function", "test")
    assert wrapper(2) == 3
    with profiling.span("disabled"):
        pass
    assert calls == [2]
    assert _events(recorder.snapshot()) == []


def test_memory_is_bounded_and_drops_are_explicit():
    recorder = profiling.TraceRecorder(max_events=3)
    for _ in range(10):
        with recorder.span("bounded"):
            pass
    snapshot = recorder.snapshot()
    assert len(_events(snapshot)) == 3
    assert snapshot["dropped_events"] == 7
    assert len(snapshot["traceEvents"]) == 6  # process x2, thread x1, spans x3
    with pytest.raises(ValueError, match="positive"):
        profiling.TraceRecorder(max_events=0)


def test_thread_stacks_and_metadata_are_independent():
    recorder = profiling.TraceRecorder(process_name="rollout worker 2")
    from threading import Barrier

    barrier = Barrier(2)

    def work():
        with recorder.span("thread_parent"):
            barrier.wait(timeout=5)
            with recorder.span("thread_child"):
                pass

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: work(), range(2)))
    snapshot = recorder.snapshot()
    events = _events(snapshot)
    assert len({event["tid"] for event in events}) == 2
    assert sorted(event["args"]["depth"] for event in events) == [0, 0, 1, 1]
    thread_metadata = [
        event for event in snapshot["traceEvents"] if event["name"] == "thread_name"
    ]
    assert len(thread_metadata) == 2
    assert json.loads(json.dumps(snapshot))["process_name"] == "rollout worker 2"


def test_exception_is_recorded_and_propagated():
    recorder = profiling.TraceRecorder()
    with pytest.raises(ValueError, match="original error"), recorder.span("failing"):
        raise ValueError("original error")
    with recorder.span("after"):
        pass
    events = _events(recorder.snapshot())
    assert events[0]["args"]["exception"] == "ValueError"
    assert events[1]["args"]["depth"] == 0


def test_snapshot_copies_events_and_stops_recording():
    recorder = profiling.TraceRecorder()
    with recorder.span("kept"):
        pass
    snapshot = recorder.snapshot(disable=True)
    _events(snapshot)[0]["args"]["self_us"] = -1
    assert _events(recorder.snapshot())[0]["args"]["self_us"] >= 0
    with recorder.span("disabled"):
        pass
    assert len(_events(recorder.snapshot())) == 1


def test_aliases_are_wrapped_once_and_results_are_unchanged(monkeypatch):
    def original(value):
        return [value]

    owner = SimpleNamespace(merge=original)
    caller = SimpleNamespace(imported_merge=original)
    recorder = profiling.TraceRecorder()
    monkeypatch.setattr(profiling, "_recorder", recorder)
    profiling._patch_aliases(owner, "merge", [caller], "merge", "chemistry")
    wrapper = owner.merge
    profiling._patch_aliases(owner, "merge", [caller], "merge", "chemistry")
    assert owner.merge is caller.imported_merge is wrapper
    assert caller.imported_merge(3) == [3]
    events = _events(recorder.snapshot())
    assert len(events) == 1
    assert events[0]["args"]["result_count"] == 1


def test_environment_score_callable_is_traced_without_changing_result(monkeypatch):
    recorder = profiling.TraceRecorder()
    monkeypatch.setattr(profiling, "_recorder", recorder)
    env = SimpleNamespace(scoring_function=lambda smiles: len(smiles))
    wrapper = profiling._wrap(
        lambda self: self.scoring_function("CCO"),
        "env.step",
        "environment",
        env_scoring=True,
    )
    assert wrapper(env) == 3
    scorer = env.scoring_function
    assert wrapper(env) == 3
    assert env.scoring_function is scorer
    events = _events(recorder.snapshot())
    assert [event["name"] for event in events] == [
        "env.step", "objective.score", "env.step", "objective.score"
    ]


def test_worker_collection_excludes_local_driver():
    calls = []

    class Group:
        def foreach_env_runner(self, function, **kwargs):
            calls.append(kwargs)
            return ["remote"]

    algo = SimpleNamespace(env_runner_group=Group())
    assert profiling.enable_workers(algo, 123) == ["remote"]
    assert profiling.disable_workers(algo) == ["remote"]
    assert calls == [{"local_env_runner": False}, {"local_env_runner": False}]


def test_worker_instances_trace_under_cached_actor_dispatch(monkeypatch):
    recorder = profiling.TraceRecorder()
    monkeypatch.setattr(profiling, "_recorder", recorder)

    class SyncSampler:
        def next(self):
            with profiling.span("environment.step", "environment"):
                return SimpleNamespace(count=16)

    class Policy:
        def set_weights(self, weights):
            self.weights = weights

    class Worker:
        def __init__(self):
            self.input_reader = SyncSampler()
            self.policy_map = {"default": Policy()}

        def sample(self):
            return self.input_reader.next()

        def set_weights(self, weights):
            self.policy_map["default"].set_weights(weights)

    # Model Ray actor executors retaining methods from before class patches.
    cached_sample = Worker.sample
    cached_set_weights = Worker.set_weights
    profiling._patch(Worker, "sample", "RolloutWorker.sample", "rollout")
    profiling._patch(Worker, "set_weights", "RolloutWorker.set_weights", "sync")
    worker = Worker()
    assert profiling._install_worker_instances(worker) == [
        "SyncSampler.next", "Policy.set_weights"
    ]
    reader_wrapper = worker.input_reader.next
    policy_wrapper = worker.policy_map["default"].set_weights
    profiling._install_worker_instances(worker)
    assert worker.input_reader.next is reader_wrapper
    assert worker.policy_map["default"].set_weights is policy_wrapper
    assert cached_sample(worker).count == 16
    cached_set_weights(worker, {"tensor": 1})
    assert worker.policy_map["default"].weights == {"tensor": 1}
    events = _events(recorder.snapshot())
    assert [event["name"] for event in events] == [
        "SyncSampler.next", "environment.step", "Policy.set_weights"
    ]
    assert events[0]["args"]["sampled_steps"] == 16
    assert events[1]["args"]["depth"] == 1
    assert events[2]["args"]["depth"] == 0
