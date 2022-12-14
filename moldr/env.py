import copy
from typing import List, Optional, Tuple

import gym
import numpy as np
from guacamol.standard_benchmarks import logP_benchmark
from gym.utils import seeding
from mi_collections.chemutils import get_mol, get_smiles
from mi_collections.mol2vec.model import Mol2Vec
from mi_collections.moldr.reassemble import merge_edge, merge_node
from mi_collections.molgraph import sanitize_molgraph
from numpy import ndarray
from rdkit.Chem import Draw

from moldr.config import get_default_config


class MolEnvValueMax(gym.Env):
    def __init__(self, env_config):
        self.action_space = env_config["ACTION_SPACE"]
        self.observation_space = env_config["OBS_SPACE"]
        self.building_blocks = env_config["BUILDING_BLOCKS"]
        self.scoring_function = env_config["SCORE_FUNCTION"]
        self.final_weight = env_config["FINAL_WEIGHT"]
        self.length = env_config["LENGTH"]
        self.threshold = env_config["SCORE_THRESHOLD"]
        self._base_smiles = env_config["BASE_SMILES"]
        self.mol2vec = Mol2Vec(model_path=env_config["MODEL_PATH"])

        self.mols = [get_mol(s) for s in self.building_blocks]
        self.env_step = 0
        self.prev_reward = 0.0
        self.base_mol = get_mol(self._base_smiles)
        self.action_mask = np.ones(len(self.building_blocks))

    def reset(self):
        self.base_smiles = copy.deepcopy(self._base_smiles)
        self.base_mol = get_mol(self.base_smiles)
        self.env_step = 0
        self.prev_reward = 0.0
        self.prev_smiles = self.base_smiles
        vec = self.mol2vec.fit_transform([self.base_mol])
        return vec.flatten()

    def render(self, mode="human"):
        mol = self.base_mol
        smiles = get_smiles(mol)
        reward = float(self.compute_score([smiles]))
        mol.SetProp("score", str(reward))
        return Draw.MolsToGridImage(
            [mol], subImgSize=(300, 300), legends=[mol.GetProp("score")]
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reassemble(self, action):
        attached_mol = self.mols[action]
        _mol1 = merge_node(self.base_mol, attached_mol)
        _mol2 = merge_edge(self.base_mol, attached_mol)
        _mol1.extend(_mol2)
        gen_mols = sanitize_molgraph(_mol1)
        smiles = [get_smiles(mol) for mol in gen_mols]
        return gen_mols, smiles

    def step(
        self, action: Optional[List[int]] = None
    ) -> Tuple[ndarray, float, bool, dict]:
        self.env_step += 1
        self.prev_smiles = self.base_smiles
        self.prev_action = action

        gen_mols, gen_smiles = self._reassemble(action)
        infos = {
            "gen_smiles": gen_smiles,
            "prev_action": self.prev_action,
            "prev_smiles": self.prev_smiles,
            "prev_score": self.prev_reward,
        }
        if len(gen_smiles) == 0:
            return np.zeros(300), 0.0, True, infos

        rewards = self.compute_score(gen_smiles)

        # TODO: SELECT NEW NODE
        idx = np.argmax(rewards)
        # idx = np.random.choice(len(rewards))
        self.base_mol = gen_mols[idx]
        self.base_smiles = get_smiles(self.base_mol)
        obs = self.mol2vec.fit_transform([self.base_mol]).flatten()
        reward = float(rewards[idx])
        done = self.is_done(reward)
        if done:
            # reward = float(rewards[idx]) * self.final_weight #self.prev_reward
            return obs, reward, done, infos
        else:
            reward_diff = reward - self.prev_reward
            self.prev_reward = reward
            reward = reward_diff
            reward = 0.0

        # self.running_reward += reward_mean
        # score = self.running_reward if done else 0
        return obs, reward, done, infos

    def compute_score(self, smiles) -> ndarray:
        scores = np.array([self.scoring_function(s) for s in smiles])
        return scores

    def is_done(self, reward):
        if reward >= self.threshold:
            return True
        if self.prev_reward > reward:
            return True
        if len(self.base_mol.GetAtoms()) > self.length:
            return True
        else:
            return False

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = copy.deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))
        return obs.flatten()


if __name__ == "__main__":
    objective = logP_benchmark(8.0)
    building_blocks_smiles = ["CC", "CCO", "CCCC"]
    config = get_default_config(
        MolEnvValueMax,
        objective,
        building_blocks_smiles,
        model_path="models/model_300dim.pkl",
    )
    env = MolEnvValueMax(config)
    obs = env.reset()
    assert len(obs) == 300  # mol2vec dim

    rewards = 0.0
    while True:
        action = np.random.choice(len(building_blocks_smiles))
        obs, reward, done, info = env.step(action)
        rewards += reward
        if done:
            break

    print(info)
