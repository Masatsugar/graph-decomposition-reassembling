"""End-to-end reassembly environment for MOLDR.

The original :class:`moldr.env.MolEnvValueMax` let the agent pick only *which*
building block to attach; the environment then enumerated every valid merge,
scored all of them, and greedily kept ``argmax(score)``. That greedy candidate
selection is a hand-coded heuristic (not learned), it leaks the objective into
the transition dynamics, and scoring every candidate at every step dominates
runtime.

:class:`MolEnvE2E` removes that heuristic. Reassembly becomes a two-phase MDP:

* **Phase A - select building block.** The agent picks one building block from
  an (optionally sampled) pool of ``option_pool_size`` blocks.
* **Phase B - select attachment.** The chosen block is merged onto the current
  molecule at every valid site; the agent picks *which* resulting molecule to
  keep, from the enumerated candidates.

Both phases are the *same* kind of decision - "choose one option from a set of
embedded molecules" - so a single pointer-style policy with action masking
handles both (see :mod:`moldr.e2e.model`). Only the chosen candidate is scored,
and the attachment choice is learned end to end.

The observation is a ``Dict``::

    {
        "base":        Box(obs_dim,)        # mol2vec embedding of the current molecule
        "phase":       Box(2,)              # one-hot [select_bb, select_attach]
        "options":     Box(n_options, obs_dim)  # mol2vec embedding of each option
        "action_mask": Box(n_options,)      # 1 for valid options, 0 for padding
    }

The action space is ``Discrete(n_options)`` where ``n_options`` is fixed at
``max(option_pool_size or n_building_blocks, max_attach)``; the per-step
``action_mask`` says how many of those slots are live.
"""

from __future__ import annotations

import copy
from enum import IntEnum
from typing import List, Optional

import gymnasium as gym
import numpy as np
from numpy import ndarray

from moldr.chemutils import get_mol, get_smiles
from moldr.core.molgraph import sanitize_molgraph
from moldr.core.reassemble import merge_edge, merge_node
from moldr.mol2vec.model import Mol2Vec


class Phase(IntEnum):
    SELECT_BB = 0
    SELECT_ATTACH = 1


def reassemble_candidates(base_mol, attach_mol) -> List:
    """Enumerate every valid molecule from merging ``attach_mol`` onto
    ``base_mol`` (node merges + edge merges), de-duplicated and sanitized.

    This is the same enumeration the original environment used, factored out so
    both environments share one code path.
    """
    candidates = list(merge_node(base_mol, attach_mol))
    candidates.extend(merge_edge(base_mol, attach_mol))
    return sanitize_molgraph(candidates)


class MolEnvE2E(gym.Env):
    """Two-phase, pointer-policy reassembly environment (keeps RLlib PPO)."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config):
        self.building_blocks: List[str] = list(env_config["BUILDING_BLOCKS"])
        self.scoring_function = env_config["SCORE_FUNCTION"]
        self.obs_dim: int = int(env_config.get("OBS_DIM", 300))
        # Finite obs bound (see observation_space); embeddings are clipped to it.
        self._obs_bound: float = 1.0e4
        # The fixed-size action space cannot represent an unbounded number of
        # attachments, so MAX_ATTACH always resolves to a positive cap; 0/None
        # mean "use the default" (not "no cap", which the guard used to misread).
        _ma = env_config.get("MAX_ATTACH", 128)
        self.max_attach: int = int(_ma) if _ma and int(_ma) > 0 else 128
        # None -> every building block is an option each step (faithful
        # fixed-table baseline). An int K -> sample K blocks per step (used by
        # the generative pool variant, Phase 3).
        self.option_pool_size: Optional[int] = env_config.get("OPTION_POOL_SIZE", None)
        self.threshold: float = float(env_config.get("SCORE_THRESHOLD", 0.9))
        self.length: int = int(env_config.get("LENGTH", 100))
        self._base_smiles: str = env_config.get("BASE_SMILES", "C")
        self.final_weight: float = float(env_config.get("FINAL_WEIGHT", 1.0))
        # "improvement" -> dense reward = score(new) - score(prev) (default,
        # better for PPO). "sparse" -> reward only on termination (original).
        self.reward_mode: str = env_config.get("REWARD_MODE", "improvement")

        self.mol2vec = Mol2Vec(model_path=env_config["MODEL_PATH"])

        # Optional generative building-block pool (Phase 3). When a PROPOSER is
        # supplied, the fixed table is replaced by a learned fragment decoder
        # whose N blocks become phase-A options; the pointer policy then selects
        # among *generated* building blocks (end-to-end + generative). The pool
        # can be re-decoded conditioned on the current molecule (RECOND_EVERY).
        self._proposer = None
        self.recond_every = env_config.get("RECOND_EVERY", "episode")
        proposer_spec = env_config.get("PROPOSER", None)
        if proposer_spec is not None:
            from moldr.generative.proposer import VAEProposer

            self._proposer = VAEProposer.from_spec(proposer_spec)
            # The proposer owns N; its initial (context-free) library fixes the
            # action semantics and overrides any passed-in table.
            self.building_blocks = list(self._proposer.propose_library(None))

        self.bb_mols = [get_mol(s) for s in self.building_blocks]
        self.n_bb = len(self.building_blocks)
        # Precompute building-block embeddings once (static for the fixed table;
        # refreshed per episode/step when a proposer re-decodes the pool).
        self._bb_embeddings = self._embed([m for m in self.bb_mols])

        n_bb_options = self.option_pool_size or self.n_bb
        self.n_options = int(max(n_bb_options, self.max_attach))

        # Finite bound (not finfo.max) so RLlib's obs statistics never overflow
        # float32; mol2vec embeddings sit well within this range and are clipped
        # to it in _embed() so observations always lie inside the space.
        bound = self._obs_bound
        high = np.full(self.obs_dim, bound, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "base": gym.spaces.Box(-high, high, dtype=np.float32),
                "phase": gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
                "options": gym.spaces.Box(
                    low=-bound,
                    high=bound,
                    shape=(self.n_options, self.obs_dim),
                    dtype=np.float32,
                ),
                "action_mask": gym.spaces.Box(
                    0.0, 1.0, shape=(self.n_options,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(self.n_options)

        # Episode state (populated in reset()).
        self.base_mol = None
        self.base_smiles = None
        self.phase = Phase.SELECT_BB
        self.env_step = 0
        self.prev_score = 0.0
        self._candidates: List = []
        self._candidate_smiles: List[str] = []
        self._bb_pool_idx: Optional[ndarray] = None

    # ------------------------------------------------------------------ utils
    def _embed(self, mols) -> ndarray:
        if len(mols) == 0:
            return np.zeros((0, self.obs_dim), dtype=np.float32)
        vecs = np.asarray(self.mol2vec.fit_transform(mols), dtype=np.float32)
        vecs = np.nan_to_num(vecs, nan=0.0, posinf=self._obs_bound, neginf=-self._obs_bound)
        np.clip(vecs, -self._obs_bound, self._obs_bound, out=vecs)
        return vecs.reshape(len(mols), self.obs_dim)

    def _refresh_bb_pool(self, context: Optional[ndarray] = None):
        """Re-decode the N building blocks from the proposer, conditioned on
        ``context`` (the current base embedding), and refresh their cached
        embeddings. N stays fixed; only which SMILES sit behind each index
        changes. No-op without a proposer."""
        if self._proposer is None:
            return
        self.building_blocks = list(self._proposer.propose(context))
        self.bb_mols = [get_mol(s) for s in self.building_blocks]
        self._bb_embeddings = self._embed(self.bb_mols)

    def _pad_options(self, embeddings: ndarray, count: int):
        options = np.zeros((self.n_options, self.obs_dim), dtype=np.float32)
        mask = np.zeros((self.n_options,), dtype=np.float32)
        # Defense-in-depth: never over-fill the fixed buffer. count should
        # already be <= n_options (max_attach <= n_options and the bb pool is
        # sized to fit), so this only guards against future misconfiguration.
        count = min(count, self.n_options)
        if count > 0:
            options[:count] = embeddings[:count]
            mask[:count] = 1.0
        return options, mask

    def _make_obs(self):
        base_vec = self._embed([self.base_mol])[0]
        phase_onehot = np.zeros(2, dtype=np.float32)
        phase_onehot[int(self.phase)] = 1.0

        if self.phase == Phase.SELECT_BB:
            if self.option_pool_size is None:
                embeddings = self._bb_embeddings
                count = self.n_bb
                self._bb_pool_idx = np.arange(self.n_bb)
            else:
                k = min(self.option_pool_size, self.n_bb)
                self._bb_pool_idx = self.np_random.choice(
                    self.n_bb, size=k, replace=False
                )
                embeddings = self._bb_embeddings[self._bb_pool_idx]
                count = k
        else:
            embeddings = self._embed(self._candidates)
            count = len(self._candidates)

        options, mask = self._pad_options(embeddings, count)
        return {
            "base": base_vec.astype(np.float32),
            "phase": phase_onehot,
            "options": options,
            "action_mask": mask,
        }

    def _resolve_action(self, action: int, valid_count: int) -> int:
        """Map an action to a valid option index. A correctly masked policy
        always selects a live slot; this only guards against buggy/random
        callers so the env never indexes out of range."""
        action = int(action)
        if 0 <= action < valid_count:
            return action
        return action % max(valid_count, 1)

    # ------------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.base_smiles = copy.deepcopy(self._base_smiles)
        self.base_mol = get_mol(self.base_smiles)
        self.phase = Phase.SELECT_BB
        self.env_step = 0
        self.prev_score = float(self.scoring_function(self.base_smiles))
        self._candidates = []
        self._candidate_smiles = []
        if self._proposer is not None and self.recond_every in ("episode", "step"):
            # Re-decode the pool conditioned on the reset-state base. For the
            # "episode" cadence it then stays fixed for the whole episode.
            base_vec = self._embed([self.base_mol])[0]
            self._refresh_bb_pool(base_vec)
        return self._make_obs(), {}

    def step(self, action):
        self.env_step += 1
        if self.phase == Phase.SELECT_BB:
            return self._step_select_bb(action)
        return self._step_select_attach(action)

    def _step_select_bb(self, action):
        pool = self._bb_pool_idx
        idx = self._resolve_action(action, len(pool))
        bb_index = int(pool[idx])
        attach_mol = self.bb_mols[bb_index]

        candidates = reassemble_candidates(self.base_mol, attach_mol)
        if len(candidates) > self.max_attach:
            candidates = candidates[: self.max_attach]

        info = {
            "phase": "select_bb",
            "chosen_bb": bb_index,
            "chosen_bb_smiles": self.building_blocks[bb_index],
            "n_candidates": len(candidates),
            "prev_smiles": self.base_smiles,
        }

        if len(candidates) == 0:
            # Dead end: the chosen block cannot attach. Terminate with the
            # molecule built so far and pay the same completion bonus a terminal
            # attach step would (the molecule is unchanged, so there is no
            # incremental improvement to add). prev_score holds the score of the
            # molecule built so far, so both reward modes reduce to the bonus.
            info["score"] = self.prev_score
            reward = self.prev_score * self.final_weight
            obs = self._make_obs()
            return obs, float(reward), True, False, info

        self._candidates = candidates
        self._candidate_smiles = [get_smiles(m) for m in candidates]
        self.phase = Phase.SELECT_ATTACH
        return self._make_obs(), 0.0, False, False, info

    def _step_select_attach(self, action):
        idx = self._resolve_action(action, len(self._candidates))
        prev_smiles = self.base_smiles  # snapshot BEFORE the transition
        chosen = self._candidates[idx]
        self.base_mol = chosen
        self.base_smiles = self._candidate_smiles[idx]

        score = float(self.scoring_function(self.base_smiles))
        improvement = score - self.prev_score
        done = self._is_done(score)

        if self.reward_mode == "sparse":
            reward = (score * self.final_weight) if done else 0.0
        else:  # "improvement"
            reward = improvement
            if done:
                reward += score * self.final_weight

        info = {
            "phase": "select_attach",
            "chosen_attach": idx,
            "gen_smiles": self._candidate_smiles,
            "prev_smiles": prev_smiles,
            "score": score,
            "improvement": improvement,
        }

        self.prev_score = score
        self.phase = Phase.SELECT_BB
        self._candidates = []
        self._candidate_smiles = []
        if not done and self._proposer is not None and self.recond_every == "step":
            # Maximally adaptive: re-decode the pool against the new base before
            # the next building-block choice.
            self._refresh_bb_pool(self._embed([self.base_mol])[0])
        return self._make_obs(), float(reward), done, False, info

    def _is_done(self, score: float) -> bool:
        if score >= self.threshold:
            return True
        if self.base_mol is not None and self.base_mol.GetNumAtoms() > self.length:
            return True
        return False

    def render(self, mode="human"):
        return self.base_smiles
