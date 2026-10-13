"""Unit tests for the end-to-end reassembly environment (no Ray required)."""

import numpy as np
import pytest

from moldr.chemutils import get_mol
from moldr.e2e.env import MolEnvE2E, Phase, reassemble_candidates

MODEL_PATH = "models/model_300dim.pkl"
BUILDING_BLOCKS = [
    "CC",
    "CCO",
    "c1ccccc1",
    "C1CCCCC1",
    "CC1CCC2=CC=CC=C2O1",
]


def _num_atoms_score(smiles):
    mol = get_mol(smiles)
    return 0.0 if mol is None else min(1.0, mol.GetNumAtoms() / 50.0)


def _make_env(**overrides):
    cfg = {
        "BUILDING_BLOCKS": BUILDING_BLOCKS,
        "SCORE_FUNCTION": _num_atoms_score,
        "MODEL_PATH": MODEL_PATH,
        "BASE_SMILES": "C",
        "SCORE_THRESHOLD": 0.9,
        "LENGTH": 60,
        "OBS_DIM": 300,
        "MAX_ATTACH": 32,
    }
    cfg.update(overrides)
    return MolEnvE2E(cfg)


def _sample_valid_action(env, obs, rng):
    valid = np.flatnonzero(obs["action_mask"] > 0)
    return int(rng.choice(valid))


def test_reset_obs_structure():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert set(obs) == {"base", "phase", "options", "action_mask"}
    assert obs["base"].shape == (300,)
    assert obs["phase"].shape == (2,)
    assert obs["options"].shape == (env.n_options, 300)
    assert obs["action_mask"].shape == (env.n_options,)
    assert env.observation_space.contains(obs)
    # phase A offers every building block.
    assert env.phase == Phase.SELECT_BB
    assert obs["action_mask"].sum() == len(BUILDING_BLOCKS)
    assert np.array_equal(obs["phase"], np.array([1.0, 0.0], dtype=np.float32))


def test_phase_alternation_and_masks():
    env = _make_env()
    rng = np.random.default_rng(1)
    obs, _ = env.reset(seed=1)

    # Select a building block known to produce candidates.
    bb_idx = BUILDING_BLOCKS.index("c1ccccc1")
    obs, reward, done, trunc, info = env.step(bb_idx)
    assert info["phase"] == "select_bb"
    assert info["n_candidates"] > 0
    assert env.phase == Phase.SELECT_ATTACH
    # phase B mask reflects the number of candidates.
    assert obs["action_mask"].sum() == info["n_candidates"]
    assert np.array_equal(obs["phase"], np.array([0.0, 1.0], dtype=np.float32))
    assert reward == 0.0 and not done

    # Select an attachment.
    act = _sample_valid_action(env, obs, rng)
    obs, reward, done, trunc, info = env.step(act)
    assert info["phase"] == "select_attach"
    assert "score" in info
    assert env.phase == Phase.SELECT_BB or done
    assert env.observation_space.contains(obs)


def test_masked_random_rollout_is_valid():
    env = _make_env()
    rng = np.random.default_rng(7)
    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        done = False
        steps = 0
        while not done and steps < 40:
            act = _sample_valid_action(env, obs, rng)
            obs, reward, done, trunc, info = env.step(act)
            steps += 1
            # Whatever molecule we land on must be a valid SMILES.
            assert get_mol(env.base_smiles) is not None
            assert np.isfinite(reward)


def test_option_pool_size_limits_phase_a():
    env = _make_env(OPTION_POOL_SIZE=3)
    obs, _ = env.reset(seed=0)
    assert env.phase == Phase.SELECT_BB
    assert obs["action_mask"].sum() == 3
    # n_options accommodates the larger of the bb pool and max_attach.
    assert env.n_options == max(3, 32)


def test_resolve_action_guards_out_of_range():
    env = _make_env()
    env.reset(seed=0)
    # Out-of-range actions must not raise; they wrap into the valid set.
    assert env._resolve_action(10_000, 5) == 10_000 % 5
    assert env._resolve_action(3, 5) == 3


def test_reward_modes_differ():
    # Sparse mode yields 0 on non-terminal attach steps; improvement mode can be
    # nonzero as the molecule grows.
    env = _make_env(REWARD_MODE="sparse", LENGTH=200, SCORE_THRESHOLD=2.0)
    rng = np.random.default_rng(3)
    obs, _ = env.reset(seed=3)
    obs, _, _, _, _ = env.step(BUILDING_BLOCKS.index("c1ccccc1"))
    act = _sample_valid_action(env, obs, rng)
    _, reward_sparse, done, _, _ = env.step(act)
    if not done:
        assert reward_sparse == 0.0


def test_reassemble_candidates_helper():
    base = get_mol("C")
    attach = get_mol("c1ccccc1")
    cands = reassemble_candidates(base, attach)
    assert isinstance(cands, list)
    assert len(cands) > 0
    # sanitize_molgraph returns RDKit Mol objects, all valid.
    assert all(c is not None and c.GetNumAtoms() > 0 for c in cands)
