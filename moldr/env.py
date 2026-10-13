import copy
from typing import Tuple

import gymnasium as gym
import numpy as np
from numpy import ndarray
from gymnasium.utils import seeding


from moldr.chemutils import get_mol, get_smiles, validate_reassembly_mol
from moldr.mol2vec.model import Mol2Vec
from moldr.core.reassemble import merge_edge, merge_node
from moldr.core.molgraph import sanitize_molgraph


class MolEnvValueMax(gym.Env):
    def __init__(self, env_config):
        self.action_space = env_config["ACTION_SPACE"]
        self.observation_space = env_config["OBS_SPACE"]
        self.building_blocks = list(env_config["BUILDING_BLOCKS"])
        self.scoring_function = env_config["SCORE_FUNCTION"]
        self.final_weight = env_config["FINAL_WEIGHT"]
        self.length = env_config["LENGTH"]
        self.max_episode_steps = int(env_config.get("MAX_EPISODE_STEPS", self.length))
        if self.length < 1 or self.max_episode_steps < 1:
            raise ValueError("LENGTH and MAX_EPISODE_STEPS must be positive.")
        self.threshold = env_config["SCORE_THRESHOLD"]
        self._base_smiles = env_config["BASE_SMILES"]
        self.mols = [get_mol(s) for s in self.building_blocks]
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError("ACTION_SPACE must be Discrete.")
        if len(self.mols) != self.action_space.n or any(m is None for m in self.mols):
            raise ValueError("BUILDING_BLOCKS must contain one valid SMILES per action.")
        for mol in self.mols:
            validate_reassembly_mol(mol)
        self.env_step = 0
        self.prev_reward = 0.0
        self.base_mol = get_mol(self._base_smiles)
        if self.base_mol is None:
            raise ValueError("BASE_SMILES must be a valid molecule.")
        validate_reassembly_mol(self.base_mol)
        self.mol2vec = Mol2Vec(model_path=env_config["MODEL_PATH"])
        self.action_mask = np.ones(len(self.building_blocks))

        # --- Generative building-block proposer (optional, backward compatible) ---
        # When env_config carries a "PROPOSER" (a VAEProposer or a spec dict, see
        # moldr.generative.proposer.VAEProposer.from_spec), the fixed table is
        # replaced by a learned fragment decoder. The action space stays
        # Discrete(N) with N == proposer.n_blocks == len(building_blocks); only
        # *which* SMILES sit behind each stable index is re-decoded, conditioned
        # on the current base embedding. RECOND_EVERY in {"never","episode","step"}.
        self._proposer = None
        self.recond_every = env_config.get("RECOND_EVERY", "episode")
        if self.recond_every not in {"never", "episode", "step"}:
            raise ValueError("RECOND_EVERY must be 'never', 'episode', or 'step'.")
        proposer_spec = env_config.get("PROPOSER", None)
        if proposer_spec is not None:
            from moldr.generative.proposer import VAEProposer

            self._proposer = VAEProposer.from_spec(proposer_spec)
            if self._proposer.n_blocks != self.action_space.n:
                raise ValueError("PROPOSER.n_blocks must equal the action-space size.")

    def _decode_pool(self, context):
        """Re-decode the N building blocks from the proposer, conditioned on
        ``context`` (the 300-d Mol2Vec embedding of the current base). Keeps N
        fixed; refreshes self.building_blocks / self.mols in place."""
        if self._proposer is None:
            return
        blocks = list(self._proposer.propose(context))
        mols = [get_mol(s) for s in blocks]
        if len(mols) != self.action_space.n or any(m is None for m in mols):
            raise ValueError("PROPOSER must emit one valid molecule per action.")
        for mol in mols:
            validate_reassembly_mol(mol)
        self.building_blocks, self.mols = blocks, mols

    def _observation(self):
        obs = np.asarray(
            self.mol2vec.fit_transform([self.base_mol]),
            dtype=self.observation_space.dtype,
        ).flatten()
        if obs.shape != self.observation_space.shape or not np.isfinite(obs).all():
            raise ValueError("Mol2Vec output must match OBS_SPACE and contain finite values.")
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.base_smiles = self._base_smiles
        self.base_mol = get_mol(self.base_smiles)
        self.env_step = 0
        self.prev_reward = float(self.compute_score([self.base_smiles])[0])
        self.prev_smiles = self.base_smiles
        obs = self._observation()
        if self._proposer is not None and self.recond_every in ("episode", "step"):
            # Rebuild the per-episode pool conditioned on the reset-state base.
            # Within the episode self.mols is then held fixed (matching the env's
            # "built once, constant within an episode" contract) unless
            # recond_every == "step" re-decodes again inside step().
            self._decode_pool(obs)
        return obs, {}

    def render(self, mode="human"):
        from rdkit.Chem import Draw

        mol = self.base_mol
        smiles = get_smiles(mol)
        reward = float(self.compute_score([smiles])[0])
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

    def step(self, action: int) -> Tuple[ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside ACTION_SPACE.")
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
            # A dead end completes the molecule built so far. Preserve its
            # observation and sparse completion reward for the value target.
            infos["score"] = self.prev_reward
            return self._observation(), self.prev_reward * self.final_weight, True, False, infos

        rewards = self.compute_score(gen_smiles)

        # TODO: SELECT NEW NODE BASED ON VF
        idx = np.argmax(rewards)
        # idx = np.random.choice(len(rewards))
        self.base_mol = gen_mols[idx]
        self.base_smiles = get_smiles(self.base_mol)
        obs = self._observation()
        if self._proposer is not None and self.recond_every == "step":
            # Maximally adaptive: re-decode the pool against the new base. Note
            # the index->block map is then only stable across episodes.
            self._decode_pool(obs)
        reward = float(rewards[idx])
        done = self.is_done(reward)
        truncated = not done and self.env_step >= self.max_episode_steps
        self.prev_reward = reward
        infos["score"] = reward
        # A time limit is a truncation, with no completion bonus. PPO can
        # bootstrap from the actual final observation.
        return obs, reward * self.final_weight if done else 0.0, done, truncated, infos

    def compute_score(self, smiles) -> ndarray:
        scores = np.array([self.scoring_function(s) for s in smiles], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("SCORE_FUNCTION must return finite scores.")
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
