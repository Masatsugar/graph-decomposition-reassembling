"""Opt-in logical Chrome tracing for a single-host RLlib training run.

Importing this module does not import Torch/Ray or change training. ``enable``
installs wrappers in the current process; workers must be enabled separately.
When recording is disabled, wrappers call their original function directly.

Each process anchors its monotonic clock to the host wall clock once. Passing
the same ``start_epoch_ns`` aligns driver and local Ray actors without letting
later wall-clock adjustments change durations. This is a *single-host* clock
scheme: distributed hosts require a separately measured clock offset.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial, wraps
from typing import Any

_recorder: TraceRecorder | None = None
_install_lock = threading.Lock()


class TraceRecorder:
    """A bounded, thread-safe recorder of nested complete (``X``) events.

    ``args.self_us`` is exclusive CPU wall time, after subtracting directly
    nested logical spans in the same thread. It can be summed across categories
    without counting parents and children twice. It is neither GPU kernel time
    nor CPU utilization; concurrent processes/threads accumulate separately.
    """

    def __init__(
        self,
        start_epoch_ns: int | None = None,
        *,
        process_name: str = "driver",
        max_events: int = 250_000,
        torch_ranges: bool = False,
        process_sort_index: int = 0,
    ):
        if max_events < 1:
            raise ValueError("max_events must be positive")
        before = time.perf_counter_ns()
        anchor_epoch_ns = time.time_ns()
        after = time.perf_counter_ns()
        self._anchor_perf_ns = (before + after) // 2
        self._anchor_epoch_ns = anchor_epoch_ns
        self.start_epoch_ns = (
            anchor_epoch_ns if start_epoch_ns is None else int(start_epoch_ns)
        )
        self.process_name = str(process_name)
        self.process_sort_index = int(process_sort_index)
        self.process_id = os.getpid()
        self.max_events = int(max_events)
        self.enabled = True
        self._lock = threading.Lock()
        self._local = threading.local()
        self._events: list[dict[str, Any]] = []
        self._threads: dict[int, str] = {}
        self.dropped_events = 0
        self._record_function = None
        if torch_ranges:
            from torch.profiler import record_function

            self._record_function = record_function

    def _timestamp_us(self, perf_ns: int) -> float:
        return (
            self._anchor_epoch_ns
            - self.start_epoch_ns
            + perf_ns
            - self._anchor_perf_ns
        ) / 1_000

    def span(self, name: str, category: str = "moldr", args: dict | None = None):
        """Record a scope; inactive scopes avoid querying the clock."""
        if not self.enabled:
            return nullcontext()
        return _TraceSpan(self, name, category, args)

    def snapshot(self, *, disable: bool = False) -> dict[str, Any]:
        """Copy recorded events with Chrome process/thread metadata.

        Call after sampling/training has finished. Scopes still open when
        ``disable=True`` are intentionally not emitted as partial events.
        """
        with self._lock:
            if disable:
                self.enabled = False
            metadata = [
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": self.process_id,
                    "tid": 0,
                    "args": {"name": self.process_name},
                },
                {
                    "ph": "M",
                    "name": "process_sort_index",
                    "pid": self.process_id,
                    "tid": 0,
                    "args": {"sort_index": self.process_sort_index},
                },
            ]
            metadata.extend(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": self.process_id,
                    "tid": tid,
                    "args": {"name": name},
                }
                for tid, name in self._threads.items()
            )
            # A snapshot owns its dictionaries, so merging/converting clocks
            # in a caller never corrupts the recorder's next snapshot.
            events = [dict(event, args=dict(event["args"])) for event in self._events]
            events.sort(key=lambda event: (event["ts"], -event["dur"]))
            return {
                "traceEvents": metadata + events,
                "displayTimeUnit": "ms",
                "process_id": self.process_id,
                "process_name": self.process_name,
                "start_epoch_ns": self.start_epoch_ns,
                "dropped_events": self.dropped_events,
                "max_events": self.max_events,
                "clock": "host-wall-clock-anchored monotonic; single host only",
            }


class _TraceSpan:
    def __init__(self, recorder, name, category, args):
        self.recorder = recorder
        self.name = name
        self.category = category
        self.args = dict(args or {})
        self.child_ns = 0
        self._torch_scope = None

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        self.tid = threading.get_ident()
        self.thread_name = threading.current_thread().name
        stack = getattr(self.recorder._local, "stack", None)
        if stack is None:
            stack = self.recorder._local.stack = []
        self.depth = len(stack)
        stack.append(self)
        if self.recorder._record_function is not None:
            self._torch_scope = self.recorder._record_function(f"MOLDR/{self.name}")
            self._torch_scope.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._torch_scope is not None:
            self._torch_scope.__exit__(exc_type, exc_value, traceback)
        end_ns = time.perf_counter_ns()
        elapsed_ns = end_ns - self.start_ns
        stack = self.recorder._local.stack
        stack.pop()
        if stack:
            stack[-1].child_ns += elapsed_ns
        self.args.update(
            self_us=max(0, elapsed_ns - self.child_ns) / 1_000,
            depth=self.depth,
        )
        if exc_type is not None:
            self.args["exception"] = exc_type.__name__
        with self.recorder._lock:
            if not self.recorder.enabled:
                return False
            if len(self.recorder._events) >= self.recorder.max_events:
                self.recorder.dropped_events += 1
                return False
            self.recorder._threads[self.tid] = self.thread_name
            self.recorder._events.append(
                {
                    "ph": "X",
                    "name": self.name,
                    "cat": self.category,
                    "pid": self.recorder.process_id,
                    "tid": self.tid,
                    "ts": self.recorder._timestamp_us(self.start_ns),
                    "dur": elapsed_ns / 1_000,
                    "args": self.args,
                }
            )
        return False


def get_recorder() -> TraceRecorder | None:
    return _recorder


def span(name: str, category: str = "moldr", args: dict | None = None):
    """Record a driver scope when profiling is enabled."""
    recorder = _recorder
    if recorder is None or not recorder.enabled:
        return nullcontext()
    return recorder.span(name, category, args)


def _wrap(function: Callable, name: str, category: str, *, env_scoring=False):
    if getattr(function, "_moldr_trace_wrapper", False):
        return function

    @wraps(function)
    def traced(*args, **kwargs):
        recorder = _recorder
        if recorder is None or not recorder.enabled:
            return function(*args, **kwargs)
        # E2E invokes SCORE_FUNCTION directly. Wrap the per-instance callable
        # once so its time is visible even when it is a plain function/lambda.
        if env_scoring and args:
            env = args[0]
            scorer = getattr(env, "scoring_function", None)
            if scorer is not None and not getattr(scorer, "_moldr_trace_wrapper", False):
                env.scoring_function = _wrap(scorer, "objective.score", "scoring")
        with recorder.span(name, category) as current:
            result = function(*args, **kwargs)
            if isinstance(result, list):
                current.args["result_count"] = len(result)
            if category == "rollout":
                sampled_steps = getattr(result, "count", None)
                if isinstance(sampled_steps, int):
                    current.args["sampled_steps"] = sampled_steps
            return result

    traced._moldr_trace_wrapper = True
    return traced


def _patch(owner, attribute, name, category, *, env_scoring=False):
    function = getattr(owner, attribute, None)
    if function is not None:
        setattr(owner, attribute, _wrap(function, name, category, env_scoring=env_scoring))


def _patch_aliases(owner, attribute, aliases, name, category):
    """Patch owner plus previously imported globals (``from x import y``)."""
    function = getattr(owner, attribute)
    wrapper = _wrap(function, name, category)
    for module in [owner, *aliases]:
        for attr, value in list(vars(module).items()):
            if value is function:
                setattr(module, attr, wrapper)


def install(*, include_rllib: bool = True) -> None:
    """Idempotently install selected logical scopes, without changing env files."""
    with _install_lock:
        legacy = importlib.import_module("moldr.env")
        e2e = importlib.import_module("moldr.e2e.env")
        reassemble = importlib.import_module("moldr.core.reassemble")
        molgraph = importlib.import_module("moldr.core.molgraph")
        mol2vec = importlib.import_module("moldr.mol2vec.model")
        proposer = importlib.import_module("moldr.generative.proposer")

        for cls, methods in [
            (
                legacy.MolEnvValueMax,
                {
                    "step": "environment",
                    "reset": "environment",
                    "_reassemble": "reassembly",
                    "compute_score": "scoring",
                    "_observation": "embedding",
                    "_decode_pool": "proposer",
                },
            ),
            (
                e2e.MolEnvE2E,
                {
                    "step": "environment",
                    "reset": "environment",
                    "_step_select_bb": "environment",
                    "_step_select_attach": "environment",
                    "_embed": "embedding",
                    "_make_obs": "embedding",
                    "_refresh_bb_pool": "proposer",
                },
            ),
        ]:
            for method, category in methods.items():
                _patch(
                    cls,
                    method,
                    f"{cls.__name__}.{method}",
                    category,
                    env_scoring=method in {"step", "reset"},
                )
        _patch(e2e, "reassemble_candidates", "reassemble_candidates", "reassembly")
        for function in ("merge_node", "merge_edge"):
            _patch_aliases(reassemble, function, [legacy, e2e], function, "chemistry")
        _patch_aliases(
            molgraph, "sanitize_molgraph", [legacy, e2e], "sanitize_molgraph", "chemistry"
        )
        _patch(mol2vec.Mol2Vec, "transform", "Mol2Vec.transform", "embedding")
        _patch(proposer.VAEProposer, "propose", "VAEProposer.propose", "proposer")

        if not include_rllib:
            return
        ppo = importlib.import_module("ray.rllib.algorithms.ppo.ppo")
        rollout_ops = importlib.import_module("ray.rllib.execution.rollout_ops")
        train_ops = importlib.import_module("ray.rllib.execution.train_ops")
        rollout_worker = importlib.import_module("ray.rllib.evaluation.rollout_worker")
        runner_group = importlib.import_module("ray.rllib.env.env_runner_group")
        policy = importlib.import_module("ray.rllib.policy.torch_policy_v2")
        model = importlib.import_module("moldr.e2e.model")

        _patch(ppo.PPO, "training_step", "PPO.training_step", "iteration")
        _patch_aliases(
            rollout_ops,
            "synchronous_parallel_sample",
            [ppo],
            "synchronous_parallel_sample",
            "sampling_wait",
        )
        for function in ("train_one_step", "multi_gpu_train_one_step"):
            _patch_aliases(train_ops, function, [ppo], function, "optimization")
        _patch(runner_group.EnvRunnerGroup, "sync_weights", "sync_weights", "sync")
        _patch(rollout_worker.RolloutWorker, "sample", "RolloutWorker.sample", "rollout")
        _patch(rollout_worker.RolloutWorker, "set_weights", "RolloutWorker.set_weights", "sync")
        for method, category in {
            "load_batch_into_buffer": "learner",
            "learn_on_loaded_batch": "learner",
            "learn_on_batch": "learner",
            "compute_gradients": "learner",
            "apply_gradients": "learner",
            "compute_actions_from_input_dict": "inference",
        }.items():
            _patch(policy.TorchPolicyV2, method, f"TorchPolicyV2.{method}", category)
        _patch(model.PointerPolicyNet, "forward", "PointerPolicyNet.forward", "inference")


def enable(
    start_epoch_ns: int | None = None,
    *,
    process_name: str = "driver",
    max_events: int = 250_000,
    torch_ranges: bool = False,
    include_rllib: bool = True,
    process_sort_index: int = 0,
) -> dict[str, Any]:
    """Start a new recording in this process, usually after training warmup."""
    global _recorder
    install(include_rllib=include_rllib)
    if _recorder is not None:
        _recorder.enabled = False
    _recorder = TraceRecorder(
        start_epoch_ns,
        process_name=process_name,
        max_events=max_events,
        torch_ranges=torch_ranges,
        process_sort_index=process_sort_index,
    )
    return {
        "process_id": _recorder.process_id,
        "process_name": _recorder.process_name,
        "start_epoch_ns": _recorder.start_epoch_ns,
    }


def disable_snapshot() -> dict[str, Any]:
    """Stop this process's recording and return its serializable snapshot."""
    recorder = _recorder
    if recorder is None:
        return {"traceEvents": [], "dropped_events": 0, "process_id": os.getpid()}
    return recorder.snapshot(disable=True)


def _install_worker_instances(worker) -> list[str]:
    """Wrap calls made *inside* Ray's cached actor method executors.

    Ray captures ``RolloutWorker.sample`` and ``set_weights`` before recording
    is enabled. Changing their module classes cannot reliably replace those
    actor dispatchers. The original sample method dynamically calls the input
    reader's ``next`` and the original weight handler calls each policy's
    ``set_weights``, so bound instance wrappers cover the actual execution.
    The reader scope is a fragment, not the whole actor sample RPC: subsequent
    concatenation, callbacks, compression and serialization are outside it.
    """
    installed = []
    reader = getattr(worker, "input_reader", None)
    if reader is not None and callable(getattr(reader, "next", None)):
        scope_name = f"{type(reader).__name__}.next"
        _patch(reader, "next", scope_name, "rollout")
        installed.append(scope_name)
    policy_map = getattr(worker, "policy_map", None)
    if policy_map is not None:
        for policy in policy_map.values():
            if callable(getattr(policy, "set_weights", None)):
                _patch(policy, "set_weights", "Policy.set_weights", "sync")
        installed.append("Policy.set_weights")
    return installed


def _enable_worker(worker, *, start_epoch_ns, max_events):
    import torch
    from rdkit import RDLogger

    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.*")
    index = int(getattr(worker, "worker_index", 0))
    metadata = enable(
        start_epoch_ns,
        process_name=f"rollout worker {index}",
        process_sort_index=index + 1,
        max_events=max_events,
        torch_ranges=False,
    )
    metadata["worker_instance_scopes"] = _install_worker_instances(worker)
    return metadata


def _disable_worker(worker):
    return disable_snapshot()


def _foreach_remote_worker(algo, function):
    group = getattr(algo, "env_runner_group", None)
    if group is None:
        group = algo.workers
    if hasattr(group, "foreach_env_runner"):
        return group.foreach_env_runner(function, local_env_runner=False)
    return group.foreach_worker(function, local_worker=False)


def enable_workers(algo, start_epoch_ns: int, *, max_events: int = 250_000):
    """Enable remote rollout actors, leaving the driver's recorder intact."""
    return _foreach_remote_worker(
        algo,
        partial(_enable_worker, start_epoch_ns=start_epoch_ns, max_events=max_events),
    )


def disable_workers(algo):
    """Collect and disable remote rollout actors after the measured iterations."""
    return _foreach_remote_worker(algo, _disable_worker)
