"""Evaluation selectors shared by CLI sampling and reproducible comparisons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from moldr.e2e.env import Phase
from moldr.profiling import span


@dataclass(frozen=True)
class LookaheadChoice:
    action: int
    returns: tuple[float, ...]
    rewards: tuple[float, ...]
    state_values: tuple[float, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    score_requests: int


def value_lookahead(
    env, value_fn: Callable[[dict], float], gamma: float = 0.99
) -> LookaheadChoice:
    """Compare ``r + gamma * (1-terminated) V(next_obs)`` at attachment time.

    Time limits continue to bootstrap. Previewing K options makes exactly K
    score requests; the subsequent real environment step adds request K+1.
    The objective cache can reduce computations but never the request count.
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between zero and one")
    env.validate_lookahead()
    if env.phase != Phase.SELECT_ATTACH:
        raise ValueError("value_lookahead is defined only for attachment choices")
    count = len(env._candidates)
    if count == 0:
        raise ValueError("No attachment candidates are available")
    start_requests = env.scorer.requests
    returns, rewards, values, terminated_flags, truncated_flags = [], [], [], [], []
    with span("value_lookahead", "evaluation", {"candidates": count}):
        for index in range(count):
            observation, reward, terminated, truncated, _ = env.peek_attachment(index)
            value = 0.0 if terminated else float(value_fn(observation))
            candidate_return = float(reward) + gamma * value
            if not np.isfinite(candidate_return):
                raise ValueError("Lookahead rewards and state values must be finite")
            returns.append(candidate_return)
            rewards.append(float(reward))
            values.append(value)
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
    return LookaheadChoice(
        action=int(np.argmax(returns)),
        returns=tuple(returns),
        rewards=tuple(rewards),
        state_values=tuple(values),
        terminated=tuple(terminated_flags),
        truncated=tuple(truncated_flags),
        score_requests=env.scorer.requests - start_requests,
    )


def select_value_lookahead(
    env, value_fn: Callable[[dict], float], gamma: float = 0.99
) -> int:
    return value_lookahead(env, value_fn, gamma).action


def policy_state_value(algo, observation) -> float:
    """Read the old-stack RLlib state value from the same observation contract."""
    prediction = algo.compute_single_action(observation, explore=False, full_fetch=True)
    if not isinstance(prediction, tuple) or len(prediction) < 3:
        raise ValueError(
            "Policy must expose vf_preds through full_fetch for value_lookahead"
        )
    extra = prediction[2]
    if "vf_preds" not in extra:
        raise ValueError("Policy prediction did not include vf_preds")
    return float(np.asarray(extra["vf_preds"]).reshape(-1)[0])
