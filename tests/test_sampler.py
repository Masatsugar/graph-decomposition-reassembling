import gymnasium as gym
import numpy as np

from run_moldr import Sampler


class TruncatingEnv:
    building_blocks = ("CC",)
    action_space = gym.spaces.Discrete(1)
    base_smiles = "CC"
    prev_reward = 0.5

    def reset(self):
        self.steps = 0
        return np.zeros(300, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        assert self.steps == 1, "Sampler must stop at truncation"
        return (
            np.zeros(300, dtype=np.float32),
            0.0,
            False,
            True,
            {"gen_smiles": ["CC"], "prev_smiles": "C"},
        )


class SingleObservationPolicy:
    def compute_single_action(self, obs):
        assert obs.shape == (300,)
        return 0


def test_truncated_sampling_passes_single_observation():
    sampler = Sampler(TruncatingEnv(), SingleObservationPolicy())
    assert sampler.sampling((0, 0))["gen_smiles"] == ["CC"]


def test_truncated_random_sampling():
    sampler = Sampler(TruncatingEnv(), SingleObservationPolicy())
    assert sampler.random((0, 0)) == "CC"


def test_transition_handles_dead_end_without_candidates():
    class DeadEndEnv(TruncatingEnv):
        def step(self, action):
            obs, _, _, _, info = super().step(action)
            info["gen_smiles"] = []
            return obs, self.prev_reward, True, False, info

    sampler = Sampler(DeadEndEnv(), SingleObservationPolicy())
    assert sampler.transition()[-1]["prev_smiles"] == "CC"
