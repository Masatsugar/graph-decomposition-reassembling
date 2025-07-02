import copy
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

import gymnaysium as gym
from gym.utils import seeding


import torch.nn as nn

import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import Draw

from ray.rllib.agents.ppo import PPOTrainer, ppo

from moldr.chemutils import (
    get_mol,
    get_smiles,
    mol_from_graph,
    mol_to_graph,
    sanitize,
)
from moldr.mol2vec import Mol2Vec
from moldr.core.reassemble import merge_edge, merge_node
from moldr.core.molgraph import MolGraph


def sanitize_molgraph(m_graphs: List[MolGraph]):
    graphs = [mol_to_graph(m.mol) for m in m_graphs if m.mol is not None]
    mols = [sanitize(mol_from_graph(g)) for g in graphs]
    smiles = np.unique([get_smiles(m) for m in mols])
    mols = [get_mol(s) for s in smiles]
    return mols


# class ValueFunction(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin0 = nn.Linear(in_features=300, out_features=64)
#         self.lin1 = nn.Linear(in_features=64, out_features=1)
#
#     def forward(self, x):
#         x = F.relu(self.lin0(x))
#         x = self.lin0(x)
#         return x


class MolEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = env_config["ACTION_SPACE"]
        self.observation_space = env_config["OBS_SPACE"]
        self.building_blocks = env_config["BUILDING_BLOCKS"]
        self.scoring_function = env_config["SCORE_FUNCTION"]
        self.env_step = 0
        self.length = 40
        self.threshold = 0.95
        self.prev_reward = 0.0

        self.mols = [get_mol(s) for s in self.building_blocks]
        self.mol2vec = Mol2Vec(model_path=env_config["MOL2VEC"])
        # vecs = mol2vec.fit_transform(gen_mols)
        self.states_map = defaultdict()
        self.base_smiles = "C"
        self.base_mol = get_mol(self.base_smiles)

    def reset(self):
        self.base_mol = get_mol("C")
        self.env_step = 0
        vec = self.mol2vec.fit_transform([self.base_mol])
        return vec.flatten()

    def render(self, mode="human"):
        return Draw.MolToImage(self.base_mol).show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reassemble(self, action):
        attached_mol = self.mols[action]
        _mol1 = merge_node(self.base_mol, attached_mol)
        # _mol2 = merge_edge(self.base_mol, attached_mol)
        # _mol1.extend(_mol2)
        gen_mols = sanitize_molgraph(_mol1)
        smiles = [get_smiles(mol) for mol in gen_mols]
        return gen_mols, smiles

    def step(
        self, action: Optional[List[int]] = None
    ) -> Tuple[ndarray, float, bool, dict]:
        self.env_step += 1
        gen_mols, gen_smiles = self._reassemble(action)
        if len(gen_smiles) == 0:
            print("FALSE")
            return np.zeros(300), 0.0, True, {}

        obs = self.mol2vec.fit_transform(gen_mols)
        obs = np.mean(obs, axis=0).flatten()
        reward = self.reward(gen_smiles)
        idx = np.argmax(reward)
        self.base_mol = gen_mols[idx]
        self.base_smiles = get_smiles(self.base_mol)
        reward_mean = float(np.mean(reward))  # float
        # reward_mean = _reward_mean - self.prev_reward
        # self.prev_reward = _reward_mean
        # print(reward_mean)
        done = self.is_done(reward)
        return obs, reward_mean, done, {}

    def reward(self, smiles) -> ndarray:
        scores = np.array([self.scoring_function(s) for s in smiles])
        return scores

    def is_done(self, reward):
        if max(reward) > self.threshold:
            return True
        elif len(self.base_mol.GetAtoms()) > self.length:
            return True
        else:
            return False


def sim2(smiles):
    target = "Cc1c(C)c2c(c(C)c1O)CCC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2"
    true_mol = Chem.MolFromSmiles(target)
    mol = Chem.MolFromSmiles(smiles)
    fps = [Chem.RDKFingerprint(x) for x in [true_mol, mol]]
    return DataStructs.FingerprintSimilarity(fps[0], fps[1])


if __name__ == "__main__":
    building_blocks_smiles = [
        "CC1CCC2=CC=CC=C2O1",
        "CC",
        "COC",
        "c1ccccc1",
        "CC1C(=O)NC(=O)S1",
        "CO",
    ]
    candidate_mols = [get_mol(s) for s in building_blocks_smiles]
    gen_mols = merge_node(candidate_mols[0], candidate_mols[1])
    gen_smis = [get_smiles(m.mol) for m in gen_mols if m.mol is not None]
    gen_mols = [get_mol(s) for s in gen_smis]
    scores = [sim2(s) for s in gen_smis]

    high = np.array([np.finfo(np.float32).max for i in range(300)])
    observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    config = ppo.DEFAULT_CONFIG
    config.update(
        {
            "env": MolEnv,
            "ACTION_SPACE": gym.spaces.Discrete(len(building_blocks_smiles)),
            "OBS_SPACE": observation_space,
            "BUILDING_BLOCKS": building_blocks_smiles,
            "SCORE_FUNCTION": sim2,
            "MOL2VEC": "mi_collections/mol2vec/models/model_300dim.pkl",
            "model": {
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
                "max_seq_len": 100,
            },
            "framework": "torch",
            # Set up a separate evaluation worker set for the
            # `trainer.evaluate()` call after training (see below).
            "evaluation_num_workers": 1,
            # Only for evaluation runs, render the env.
            "evaluation_config": {
                "render_env": False,
            },
        }
    )
    #
    env = MolEnv(env_config=config)
    trainer = PPOTrainer(env=MolEnv, config={"env_config": config})

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    from tqdm import tqdm

    for _ in tqdm(range(1)):
        print(trainer.train())

    # # Evaluate the trained Trainer (and render each timestep to the shell's
    # # output).
    trainer.evaluate()
