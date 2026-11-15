"""Seeded, mask-aware episode sampling for E2E and legacy environments."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from moldr.e2e.env import Phase
from moldr.evaluation import policy_state_value, value_lookahead


@dataclass
class EpisodeResult:
    smiles: str
    score: float
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    infos: list[dict] = field(default_factory=list)
    scoring: dict[str, int] = field(default_factory=dict)

    @property
    def final_smiles(self):
        return self.smiles

    def to_dict(self):
        return asdict(self)


def _accepts_keyword(function, name):
    parameters = inspect.signature(function).parameters
    return name in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


class Sampler:
    def __init__(
        self,
        env,
        algo=None,
        threshold=0.9,
        n_jobs=1,
        *,
        selector="actor",
        gamma=0.99,
        value_fn: Callable | None = None,
        max_steps: int | None = None,
        explore: bool = False,
    ):
        if selector not in {"actor", "value_lookahead"}:
            raise ValueError("selector must be actor or value_lookahead")
        self.env, self.algo = env, algo
        self.threshold, self.n_jobs = threshold, int(n_jobs)
        self.selector, self.gamma = selector, float(gamma)
        self.value_fn = value_fn
        self.explore = bool(explore)
        self.max_steps = int(
            max_steps
            if max_steps is not None
            else getattr(env, "max_episode_steps", 2 * getattr(env, "length", 500))
        )
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if selector == "value_lookahead":
            self.env.validate_lookahead()

    def _reset(self, seed):
        if _accepts_keyword(self.env.reset, "seed"):
            return self.env.reset(seed=seed)
        # Compatibility for old Python environments predating Gym's seed kwarg.
        return self.env.reset()

    def _valid_actions(self, observation):
        if isinstance(observation, dict) and "action_mask" in observation:
            return np.flatnonzero(np.asarray(observation["action_mask"]) > 0)
        return np.arange(self.env.action_space.n)

    def _policy_action(self, observation):
        if self.algo is None:
            raise ValueError("Actor sampling requires a policy/algorithm")
        kwargs = (
            {"explore": self.explore}
            if _accepts_keyword(self.algo.compute_single_action, "explore")
            else {}
        )
        result = self.algo.compute_single_action(observation, **kwargs)
        return int(result[0] if isinstance(result, tuple) else result)

    def episode(self, *, seed=None, random=False) -> EpisodeResult:
        if seed is not None and self.explore and not random:
            import torch

            # Explicit evaluation seeds also govern stochastic policy sampling,
            # independently of random draws consumed by model initialization.
            torch.manual_seed(int(seed))
        scorer = getattr(self.env, "scorer", None)
        start_stats = dict(scorer.stats) if scorer is not None else {}
        observation, _ = self._reset(seed)
        rng = np.random.default_rng(seed)
        infos, total_reward = [], 0.0
        terminated, truncated = False, False
        for _ in range(self.max_steps):
            valid = self._valid_actions(observation)
            if not len(valid):
                terminated = True
                infos.append(
                    {"phase": "no_candidates", "final_smiles": self.env.base_smiles}
                )
                break
            lookahead = None
            if random:
                action = int(rng.choice(valid))
            elif (
                self.selector == "value_lookahead"
                and getattr(self.env, "phase", None) == Phase.SELECT_ATTACH
            ):
                value_fn = self.value_fn or (
                    lambda obs: policy_state_value(self.algo, obs)
                )
                lookahead = value_lookahead(self.env, value_fn, self.gamma)
                action = lookahead.action
            else:
                action = self._policy_action(observation)
            if action not in valid:
                raise ValueError(f"Policy returned masked action {action}")
            observation, reward, terminated, truncated, info = self.env.step(action)
            info = dict(info)
            if lookahead is not None:
                info["lookahead"] = asdict(lookahead)
            infos.append(info)
            total_reward += float(reward)
            if terminated or truncated:
                break
        else:
            truncated = True
        stats = (
            {
                key: int(value) - int(start_stats.get(key, 0))
                for key, value in scorer.stats.items()
            }
            if scorer is not None
            else {}
        )
        return EpisodeResult(
            smiles=self.env.base_smiles,
            score=float(
                getattr(self.env, "prev_score", getattr(self.env, "prev_reward", 0.0))
            ),
            total_reward=total_reward,
            steps=sum(
                "phase" not in info or info["phase"] != "no_candidates"
                for info in infos
            ),
            terminated=bool(terminated),
            truncated=bool(truncated),
            infos=infos,
            scoring=stats,
        )

    def sampling(self, y):
        _, seed = y
        result = self.episode(seed=seed)
        info = dict(result.infos[-1]) if result.infos else {}
        info.update(
            final_smiles=result.smiles,
            gen_smiles=[result.smiles],
            score=result.score,
            total_reward=result.total_reward,
            terminated=result.terminated,
            truncated=result.truncated,
            scoring=result.scoring,
        )
        return info

    def random(self, y):
        _, seed = y
        return self.episode(seed=seed, random=True).smiles

    def generate(self, number_samples):
        # Each environment is stateful; deterministic serial episodes are also
        # safe on a single GPU without duplicating its policy between processes.
        return [self.random((index, index)) for index in range(number_samples)]

    def generate_single(self, number_samples):
        return [self.sampling((index, index)) for index in range(number_samples)]

    def transition(self, seed=None):
        result = self.episode(seed=seed)
        return [
            *result.infos,
            {"prev_smiles": result.smiles, "final_smiles": result.smiles},
        ]
