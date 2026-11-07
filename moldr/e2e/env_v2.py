"""Versioned E2E observations and shared candidate/scoring services.

The original ``moldr.e2e.env.MolEnvE2E`` deliberately remains the restoration
entry point for old checkpoints. This environment owns the new observation
contract and never silently changes that older contract.
"""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np

from moldr.chemutils import get_mol, get_smiles, validate_reassembly_mol
from moldr.e2e.env import Phase
from moldr.mol2vec.model import Mol2Vec
from moldr.profiling import span


class MolEnvE2EV2(gym.Env):
    """Two decisions per molecular update, with rich or classic policy context."""

    metadata: ClassVar[dict] = {"render_modes": ["human"]}
    observation_version = 2

    def __init__(self, env_config):
        from moldr.chemistry import CandidateGenerator
        from moldr.scoring import ScoringService

        self.env_config = dict(env_config)
        self.building_blocks = list(env_config["BUILDING_BLOCKS"])
        self._base_smiles = str(env_config.get("BASE_SMILES", "C"))
        self.threshold = float(env_config.get("SCORE_THRESHOLD", 0.9))
        self.length = int(env_config.get("LENGTH", 100))
        self.max_episode_steps = int(
            env_config.get("MAX_EPISODE_STEPS", 2 * self.length)
        )
        self.max_attach = int(env_config.get("MAX_ATTACH", 128))
        self.option_pool_size = env_config.get("OPTION_POOL_SIZE")
        if self.option_pool_size is not None:
            self.option_pool_size = int(self.option_pool_size)
            if self.option_pool_size < 1:
                raise ValueError("OPTION_POOL_SIZE must be positive or None")
        if self.max_episode_steps < 1 or self.max_attach < 1 or self.length < 1:
            raise ValueError(
                "MAX_EPISODE_STEPS, MAX_ATTACH and LENGTH must be positive"
            )
        self.reward_mode = env_config.get("REWARD_MODE", "improvement")
        if self.reward_mode not in {"improvement", "sparse"}:
            raise ValueError("REWARD_MODE must be improvement or sparse")
        self.final_weight = float(env_config.get("FINAL_WEIGHT", 1.0))
        self.context_mode = env_config.get("CONTEXT", "rich")
        if self.context_mode not in {"rich", "classic"}:
            raise ValueError("CONTEXT must be rich or classic")
        self.representation = dict(env_config.get("REPRESENTATION", {}))
        self.representation_name = self.representation.get("name", "mol2vec")
        if self.representation_name not in {"mol2vec", "mpnn"}:
            raise ValueError("REPRESENTATION name must be mol2vec or mpnn")
        self.obs_dim = int(env_config.get("OBS_DIM", 300))
        self._obs_bound = 1.0e4
        self.reuse_embeddings = bool(self.representation.get("reuse_embeddings", True))
        self._initial_seed = env_config.get("SEED", env_config.get("seed"))
        self._has_reset = False
        self.candidate_generator = CandidateGenerator(env_config.get("CHEMISTRY"))
        scoring = dict(env_config.get("SCORING", {}))
        scoring.setdefault("objective", env_config.get("SCORE_FUNCTION", "qed"))
        self.scorer = ScoringService(**scoring)
        self.scoring_function = self.scorer.score

        base = self._validate_mol(self._base_smiles, "BASE_SMILES")
        self._proposer = None
        self.recond_every = env_config.get("RECOND_EVERY", "episode")
        if self.recond_every not in {"never", "episode", "step"}:
            raise ValueError("RECOND_EVERY must be never, episode or step")
        if env_config.get("PROPOSER") is not None:
            from moldr.generative.proposer import VAEProposer

            self._proposer = VAEProposer.from_spec(env_config["PROPOSER"])
            self.building_blocks = list(self._proposer.propose_library(None))
        self.conditioning_dim = (
            int(self._proposer.vae.context_dim)
            if self._proposer is not None
            else self.obs_dim
        )
        if not self.building_blocks:
            raise ValueError("BUILDING_BLOCKS must contain at least one molecule")
        self.bb_mols = [
            self._validate_mol(s, "BUILDING_BLOCKS") for s in self.building_blocks
        ]
        self.n_bb = len(self.bb_mols)
        self.n_options = max(
            min(self.option_pool_size or self.n_bb, self.n_bb), self.max_attach
        )
        self.action_space = gym.spaces.Discrete(self.n_options)

        # VAE conditioning always uses its independent, frozen Mol2Vec model.
        self.mol2vec = None
        if self.representation_name == "mol2vec" or self._proposer is not None:
            self.mol2vec = Mol2Vec(model_path=env_config["MODEL_PATH"])
        self._bb_embeddings = (
            self._embed(self.bb_mols) if self.representation_name == "mol2vec" else None
        )
        default_catalog = (
            list(self._proposer.vocab.smiles)
            if self._proposer is not None
            else list(self.building_blocks)
        )
        self.catalog_smiles = list(env_config.get("CATALOG_SMILES", default_catalog))
        # Map chemically equivalent spellings to their stable catalog IDs.
        self._catalog_ids = {}
        catalog_mols = []
        for index, smiles in enumerate(self.catalog_smiles):
            mol = self._validate_mol(smiles, "CATALOG_SMILES")
            catalog_mols.append(mol)
            self._catalog_ids.setdefault(get_smiles(mol), index)
        self._bb_fragment_ids = self._fragment_ids(self.bb_mols)
        largest = max(m.GetNumAtoms() for m in catalog_mols)
        self.max_atoms = int(
            self.representation.get("max_atoms")
            or (max(base.GetNumAtoms(), self.length) + largest)
        )
        self.max_bonds = int(
            self.representation.get("max_bonds")
            or (self.max_atoms * (self.max_atoms - 1) // 2)
        )
        self._graph_cache = {}
        self._structure_cache_size = int(
            self.representation.get("structure_cache_size", 1024)
        )
        if self.representation_name == "mpnn":
            for mol in [base, *catalog_mols]:
                self._graph(mol)
        self.observation_space = self._observation_space()
        self.base_mol = None
        self.base_smiles = None
        self.phase = Phase.SELECT_BB
        self.env_step = 0
        self.prev_score = 0.0
        self._candidates = []
        self._candidate_smiles = []
        self._bb_pool_idx = None
        self._base_embedding = None
        self._candidate_embeddings = None

    @staticmethod
    def _validate_mol(smiles, label):
        mol = get_mol(smiles)
        if mol is None:
            raise ValueError(f"{label} must contain valid molecules")
        validate_reassembly_mol(mol)
        return mol

    def _fragment_ids(self, mols):
        try:
            return np.asarray(
                [self._catalog_ids[get_smiles(m)] for m in mols], dtype=np.int32
            )
        except KeyError as exc:
            raise ValueError(
                "Every proposed fragment must belong to CATALOG_SMILES"
            ) from exc

    def _observation_space(self):
        spaces = {
            "phase": gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
            "action_mask": gym.spaces.Box(
                0.0, 1.0, shape=(self.n_options,), dtype=np.float32
            ),
            "context": gym.spaces.Box(-1.0e10, 1.0e10, shape=(3,), dtype=np.float32),
        }
        if self.representation_name == "mol2vec":
            spaces["base"] = gym.spaces.Box(
                -self._obs_bound,
                self._obs_bound,
                shape=(self.obs_dim,),
                dtype=np.float32,
            )
            spaces["options"] = gym.spaces.Box(
                -self._obs_bound,
                self._obs_bound,
                shape=(self.n_options, self.obs_dim),
                dtype=np.float32,
            )
        else:
            limit = 2**31 - 1
            for prefix, lead in (("base", ()), ("candidate", (self.max_attach,))):
                spaces[f"{prefix}_atoms"] = gym.spaces.Box(
                    -limit, limit, shape=(*lead, self.max_atoms, 8), dtype=np.int32
                )
                spaces[f"{prefix}_bonds"] = gym.spaces.Box(
                    -limit, limit, shape=(*lead, self.max_bonds, 6), dtype=np.int32
                )
                spaces[f"{prefix}_counts"] = gym.spaces.Box(
                    0,
                    max(self.max_atoms, self.max_bonds),
                    shape=(*lead, 2),
                    dtype=np.int32,
                )
            spaces["fragment_ids"] = gym.spaces.Box(
                -1,
                len(self.catalog_smiles) - 1,
                shape=(self.n_options,),
                dtype=np.int32,
            )
        return gym.spaces.Dict(spaces)

    def _embed(self, mols):
        dimension = (
            self.obs_dim
            if self.representation_name == "mol2vec"
            else self.conditioning_dim
        )
        if not mols:
            return np.zeros((0, dimension), dtype=np.float32)
        with span("MolEnvE2EV2.embed", "embedding", {"molecules": len(mols)}):
            vecs = np.asarray(self.mol2vec.fit_transform(mols), dtype=np.float32)
            vecs = np.nan_to_num(
                vecs, nan=0.0, posinf=self._obs_bound, neginf=-self._obs_bound
            )
            return np.clip(vecs, -self._obs_bound, self._obs_bound).reshape(
                len(mols), dimension
            )

    def _graph(self, mol, *, cache=True):
        from moldr.gnn import graph_record

        key = get_smiles(mol)
        record = self._graph_cache.get(key)
        if record is None:
            record = graph_record(mol, self.max_atoms, self.max_bonds)
            if cache and self._structure_cache_size > 0:
                if len(self._graph_cache) >= self._structure_cache_size:
                    self._graph_cache.pop(next(iter(self._graph_cache)))
                self._graph_cache[key] = record
        return record

    def _refresh_bb_pool(self, context: np.ndarray | None = None):
        if self._proposer is None:
            return
        blocks = list(self._proposer.propose(context))
        if len(blocks) != self.n_bb:
            raise ValueError("PROPOSER must preserve the building-block pool size")
        mols = [self._validate_mol(s, "PROPOSER") for s in blocks]
        ids = self._fragment_ids(mols)
        embeddings = (
            self._embed(mols) if self.representation_name == "mol2vec" else None
        )
        self.building_blocks, self.bb_mols = blocks, mols
        self._bb_fragment_ids, self._bb_embeddings = ids, embeddings

    def _choose_pool(self):
        if self.option_pool_size is None:
            self._bb_pool_idx = np.arange(self.n_bb)
        else:
            self._bb_pool_idx = self.np_random.choice(
                self.n_bb, size=min(self.option_pool_size, self.n_bb), replace=False
            )

    def _observation(
        self,
        *,
        base_mol,
        phase,
        score,
        step,
        base_embedding=None,
        candidates=(),
        candidate_embeddings=None,
        pool_idx=None,
        cache=True,
    ):
        phase_onehot = np.zeros(2, dtype=np.float32)
        phase_onehot[int(phase)] = 1.0
        mask = np.zeros(self.n_options, dtype=np.float32)
        if phase == Phase.SELECT_BB:
            pool_idx = self._bb_pool_idx if pool_idx is None else pool_idx
            count = len(pool_idx)
        else:
            count = len(candidates)
        if count > self.n_options:
            raise ValueError("Number of live options exceeds fixed action capacity")
        mask[:count] = 1.0
        obs = {
            "phase": phase_onehot,
            "action_mask": mask,
            "context": np.asarray(
                [score, step, max(0, self.max_episode_steps - step)], dtype=np.float32
            ),
        }
        if self.representation_name == "mol2vec":
            obs["base"] = (
                self._embed([base_mol])[0]
                if base_embedding is None
                else base_embedding.copy()
            )
            options = np.zeros((self.n_options, self.obs_dim), dtype=np.float32)
            if phase == Phase.SELECT_BB:
                options[:count] = self._bb_embeddings[pool_idx]
            elif count:
                options[:count] = (
                    self._embed(list(candidates))
                    if candidate_embeddings is None
                    else candidate_embeddings
                )
            obs["options"] = options
        else:
            with span("MolEnvE2EV2.graph_observation", "gnn"):
                base = self._graph(base_mol, cache=cache)
                obs.update(
                    {
                        f"base_{key}": value.copy()
                        for key, value in base.items()
                        if key in {"atoms", "bonds", "counts"}
                    }
                )
                obs["candidate_atoms"] = np.zeros(
                    (self.max_attach, self.max_atoms, 8), dtype=np.int32
                )
                obs["candidate_bonds"] = np.zeros(
                    (self.max_attach, self.max_bonds, 6), dtype=np.int32
                )
                obs["candidate_counts"] = np.zeros((self.max_attach, 2), dtype=np.int32)
                obs["fragment_ids"] = np.full(self.n_options, -1, dtype=np.int32)
                if phase == Phase.SELECT_BB:
                    obs["fragment_ids"][:count] = self._bb_fragment_ids[pool_idx]
                else:
                    for index, mol in enumerate(candidates):
                        record = self._graph(mol, cache=cache)
                        for key in ("atoms", "bonds", "counts"):
                            obs[f"candidate_{key}"][index] = record[key]
        return obs

    def _make_obs(self):
        return self._observation(
            base_mol=self.base_mol,
            phase=self.phase,
            score=self.prev_score,
            step=self.env_step,
            base_embedding=self._base_embedding if self.reuse_embeddings else None,
            candidates=self._candidates,
            candidate_embeddings=self._candidate_embeddings
            if self.reuse_embeddings
            else None,
        )

    def _score(self, smiles, mol):
        return float(
            self.scorer.score(mol if self.scorer.input_mode == "mol" else smiles)
        )

    def reset(self, *, seed=None, options=None):
        if seed is None and not self._has_reset:
            seed = self._initial_seed
        super().reset(seed=seed)
        self._has_reset = True
        self.base_smiles = self._base_smiles
        self.base_mol = self._validate_mol(self.base_smiles, "BASE_SMILES")
        self.phase = Phase.SELECT_BB
        self.env_step = 0
        self.prev_score = self._score(self.base_smiles, self.base_mol)
        self._candidates, self._candidate_smiles = [], []
        self._candidate_embeddings = None
        self._base_embedding = (
            self._embed([self.base_mol])[0]
            if self.representation_name == "mol2vec" and self.reuse_embeddings
            else None
        )
        if self._proposer is not None and self.recond_every in {"episode", "step"}:
            self._refresh_bb_pool(self._conditioning_embedding())
        self._choose_pool()
        return self._make_obs(), {"observation_version": 2, "score": self.prev_score}

    def _conditioning_embedding(self):
        if self._base_embedding is not None:
            return self._base_embedding.copy()
        return self._embed([self.base_mol])[0]

    def _resolve_action(self, action, valid_count):
        index = int(action)
        if not 0 <= index < valid_count:
            raise ValueError(
                f"Action {index} is masked; {valid_count} options are live"
            )
        return index

    def _done_for(self, score, mol):
        return bool(score >= self.threshold or mol.GetNumAtoms() > self.length)

    def _reward_for(self, score, mol):
        terminated = self._done_for(score, mol)
        reward = 0.0 if self.reward_mode == "sparse" else score - self.prev_score
        if terminated:
            reward += score * self.final_weight
        return float(reward), terminated

    def step(self, action):
        if self.base_mol is None:
            raise RuntimeError("Call reset before step")
        count = (
            len(self._bb_pool_idx)
            if self.phase == Phase.SELECT_BB
            else len(self._candidates)
        )
        index = self._resolve_action(action, count)
        self.env_step += 1
        with span("MolEnvE2EV2.step", "environment"):
            if self.phase == Phase.SELECT_BB:
                transition = self._step_select_bb(index)
            else:
                transition = self._step_select_attach(index)
        obs, reward, terminated, _, info = transition
        truncated = bool(not terminated and self.env_step >= self.max_episode_steps)
        info["final_smiles"] = self.base_smiles
        info["scoring"] = self.scorer.stats
        return obs, reward, terminated, truncated, info

    def _step_select_bb(self, index):
        bb_index = int(self._bb_pool_idx[index])
        generated = self.candidate_generator.generate(
            self.base_mol, self.bb_mols[bb_index]
        )
        retained = generated[: self.max_attach]
        info = {
            "phase": "select_bb",
            "chosen_bb": bb_index,
            "chosen_bb_smiles": self.building_blocks[bb_index],
            "n_candidates": len(retained),
            "n_candidates_total": len(generated),
            "prev_smiles": self.base_smiles,
        }
        if not retained:
            info["score"] = self.prev_score
            return (
                self._make_obs(),
                float(self.prev_score * self.final_weight),
                True,
                False,
                info,
            )
        self._candidates = [candidate.mol for candidate in retained]
        self._candidate_smiles = [candidate.smiles for candidate in retained]
        self._candidate_embeddings = (
            self._embed(self._candidates)
            if self.representation_name == "mol2vec" and self.reuse_embeddings
            else None
        )
        self.phase = Phase.SELECT_ATTACH
        return self._make_obs(), 0.0, False, False, info

    def _step_select_attach(self, index):
        previous = self.base_smiles
        chosen_smiles = self._candidate_smiles[index]
        chosen_mol = self._candidates[index]
        score = self._score(chosen_smiles, chosen_mol)
        reward, terminated = self._reward_for(score, chosen_mol)
        info = {
            "phase": "select_attach",
            "chosen_attach": index,
            "gen_smiles": list(self._candidate_smiles),
            "prev_smiles": previous,
            "score": score,
            "improvement": score - self.prev_score,
        }
        if self.representation_name == "mol2vec" and self.reuse_embeddings:
            self._base_embedding = (
                self._candidate_embeddings[index].copy()
                if self._candidate_embeddings is not None
                else self._embed([chosen_mol])[0]
            )
        self.base_smiles, self.base_mol, self.prev_score = (
            chosen_smiles,
            chosen_mol,
            score,
        )
        self.phase = Phase.SELECT_BB
        self._candidates, self._candidate_smiles, self._candidate_embeddings = (
            [],
            [],
            None,
        )
        if (
            not terminated
            and self._proposer is not None
            and self.recond_every == "step"
        ):
            self._refresh_bb_pool(self._conditioning_embedding())
        self._choose_pool()
        return self._make_obs(), reward, terminated, False, info

    def validate_lookahead(self):
        if self._proposer is not None or self.option_pool_size is not None:
            raise ValueError(
                "value_lookahead requires a fixed table with every fragment presented"
            )
        if not self.scorer.is_registered or not self.scorer.deterministic:
            raise ValueError(
                "value_lookahead requires a registered deterministic objective"
            )

    def peek_attachment(self, action):
        """Score and observe a hypothetical attachment without changing MDP/RNG state.

        Scorer counters and its optional pure-computation cache intentionally
        record this request. Every normal selected transition scores once again.
        """
        self.validate_lookahead()
        if self.phase != Phase.SELECT_ATTACH:
            raise ValueError("peek_attachment requires the attachment phase")
        index = self._resolve_action(action, len(self._candidates))
        mol, smiles = self._candidates[index], self._candidate_smiles[index]
        score = self._score(smiles, mol)
        reward, terminated = self._reward_for(score, mol)
        step = self.env_step + 1
        embedding = (
            self._candidate_embeddings[index]
            if self.reuse_embeddings and self._candidate_embeddings is not None
            else None
        )
        obs = self._observation(
            base_mol=mol,
            phase=Phase.SELECT_BB,
            score=score,
            step=step,
            base_embedding=embedding,
            pool_idx=np.arange(self.n_bb),
            cache=False,
        )
        truncated = bool(not terminated and step >= self.max_episode_steps)
        return (
            obs,
            reward,
            terminated,
            truncated,
            {"score": score, "final_smiles": smiles},
        )

    def render(self, mode="human"):
        return self.base_smiles
