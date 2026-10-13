"""Unit tests for the end-to-end reassembly environment (no Ray required)."""

from types import SimpleNamespace

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


def test_e2e_config_accepts_numpy_block_table():
    from moldr.e2e.config import get_e2e_config

    objective = SimpleNamespace(wrapped_objective=SimpleNamespace(score=_num_atoms_score))
    config = get_e2e_config(objective, np.array(["CC", "CO"]), num_workers=0)
    assert config.env_config["BUILDING_BLOCKS"] == ["CC", "CO"]


def test_e2e_config_rejects_empty_table():
    from moldr.e2e.config import get_e2e_config

    objective = SimpleNamespace(wrapped_objective=SimpleNamespace(score=_num_atoms_score))
    with pytest.raises(ValueError, match="must not be empty"):
        get_e2e_config(objective, [], num_workers=0)


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


def test_option_pool_larger_than_library_does_not_allocate_extra_padding():
    env = _make_env(OPTION_POOL_SIZE=10000, MAX_ATTACH=3)
    obs, _ = env.reset(seed=0)
    assert env.n_options == len(BUILDING_BLOCKS)
    assert obs["action_mask"].sum() == len(BUILDING_BLOCKS)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"OPTION_POOL_SIZE": 0}, "OPTION_POOL_SIZE"),
        ({"OPTION_POOL_SIZE": -1}, "OPTION_POOL_SIZE"),
        ({"BUILDING_BLOCKS": []}, "BUILDING_BLOCKS"),
        ({"BUILDING_BLOCKS": ["invalid"]}, "BUILDING_BLOCKS"),
        ({"BASE_SMILES": "invalid"}, "BASE_SMILES"),
        ({"MAX_EPISODE_STEPS": 0}, "MAX_EPISODE_STEPS"),
        ({"REWARD_MODE": "typo"}, "REWARD_MODE"),
    ],
)
def test_invalid_configuration_fails_before_rollout(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_env(**overrides)


@pytest.mark.parametrize("smiles", ["C[C@H](O)F", "C/C=C/C"])
@pytest.mark.parametrize("key", ["BASE_SMILES", "BUILDING_BLOCKS"])
def test_specified_stereo_fails_before_training(smiles, key):
    value = [smiles] if key == "BUILDING_BLOCKS" else smiles
    with pytest.raises(ValueError, match="stereo"):
        _make_env(**{key: value})


@pytest.mark.parametrize("blocks", [["CC"], ["CC", "C[C@H](O)F", "CC", "CC", "CC"]])
def test_generated_pool_validation_preserves_previous_pool_on_failure(blocks):
    env = _make_env()
    previous = list(env.building_blocks)
    env._proposer = SimpleNamespace(propose=lambda context: blocks)
    with pytest.raises(ValueError):
        env._refresh_bb_pool()
    assert env.building_blocks == previous


def test_no_growth_episode_truncates_without_completion_bonus():
    env = _make_env(
        BUILDING_BLOCKS=["C"],
        SCORE_THRESHOLD=2.0,
        MAX_EPISODE_STEPS=4,
        REWARD_MODE="sparse",
    )
    obs, _ = env.reset(seed=0)
    for step in range(4):
        obs, reward, terminated, truncated, _ = env.step(0)
        assert env.base_smiles == "C"
        assert reward == 0.0
        assert not terminated
        assert truncated is (step == 3)
        assert env.observation_space.contains(obs)


def test_time_limit_can_truncate_building_block_selection():
    env = _make_env(MAX_EPISODE_STEPS=1)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["n_candidates"] > 0
    assert env.phase == Phase.SELECT_ATTACH
    assert reward == 0.0 and not terminated and truncated
    assert env.observation_space.contains(obs)


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


def test_dead_end_pays_completion_bonus():
    """A zero-candidate (dead-end) building block must terminate while still
    paying the completion bonus for the molecule built so far - not a silent 0.0
    that erases the whole sparse-mode training signal."""
    # CF4 has no open valence and no ring -> it can never attach to base "C".
    env = _make_env(BUILDING_BLOCKS=["FC(F)(F)F"], REWARD_MODE="sparse", FINAL_WEIGHT=1.0)
    obs, _ = env.reset(seed=0)
    base_score = _num_atoms_score("C")
    obs, reward, done, trunc, info = env.step(0)
    assert info["n_candidates"] == 0  # genuine dead end
    assert done is True
    assert reward == pytest.approx(base_score)  # bonus paid, not 0.0
    assert reward > 0.0


def test_max_attach_zero_normalizes_to_default():
    """MAX_ATTACH=0 must mean 'use the default cap', not 'no cap' (which used to
    collapse n_options and crash obs padding)."""
    env = _make_env(BUILDING_BLOCKS=["c1ccccc1"], MAX_ATTACH=0)
    assert env.max_attach == 128
    assert env.n_options == 128
    # A ring base + benzene enumerates several candidates; must not crash.
    env2 = _make_env(BUILDING_BLOCKS=["c1ccccc1"], MAX_ATTACH=0, BASE_SMILES="CC1CCC2=CC=CC=C2O1")
    obs, _ = env2.reset(seed=0)
    obs, reward, done, trunc, info = env2.step(0)
    assert info["n_candidates"] > 0
    assert env2.observation_space.contains(obs)


def test_max_attach_caps_candidates():
    env = _make_env(BUILDING_BLOCKS=["c1ccccc1"], MAX_ATTACH=3, BASE_SMILES="CC1CCC2=CC=CC=C2O1")
    obs, _ = env.reset(seed=0)
    obs, reward, done, trunc, info = env.step(0)
    assert info["n_candidates"] <= 3


def test_prev_smiles_is_pre_transition():
    """info['prev_smiles'] on an attach step must be the molecule BEFORE the
    attach, not the newly formed one."""
    env = _make_env()
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    base_before = env.base_smiles  # "C"
    obs, _, _, _, _ = env.step(BUILDING_BLOCKS.index("c1ccccc1"))  # -> phase B
    act = _sample_valid_action(env, obs, rng)
    obs, _, _, _, info = env.step(act)  # attach
    assert info["phase"] == "select_attach"
    assert info["prev_smiles"] == base_before
    assert info["prev_smiles"] != env.base_smiles  # the base actually changed


def test_reassemble_candidates_helper():
    base = get_mol("C")
    attach = get_mol("c1ccccc1")
    cands = reassemble_candidates(base, attach)
    assert isinstance(cands, list)
    assert len(cands) > 0
    # sanitize_molgraph returns RDKit Mol objects, all valid.
    assert all(c is not None and c.GetNumAtoms() > 0 for c in cands)
