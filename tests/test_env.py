"""Environment contracts without launching a Ray cluster or remote workers."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import moldr.env as env_mod
from moldr.chemutils import get_mol, get_smiles


class _Float64Mol2Vec:
    def __init__(self, *args, **kwargs):
        pass

    def fit_transform(self, mols):
        return np.ones((len(mols), 300), dtype=np.float64)


def _atom_score(smiles):
    return get_mol(smiles).GetNumAtoms() / 100.0


def _env_config(**overrides):
    cfg = {
        "BUILDING_BLOCKS": ["CC", "CCO", "CCCC"],
        "OBS_SPACE": gym.spaces.Box(-1e10, 1e10, (300,), dtype=np.float32),
        "SCORE_FUNCTION": _atom_score,
        "SCORE_THRESHOLD": 0.9,
        "FINAL_WEIGHT": 1.0,
        "BASE_SMILES": "C",
        "MODEL_PATH": "models/model_300dim.pkl",
        "LENGTH": 100,
    }
    cfg.update(overrides)
    cfg.setdefault("ACTION_SPACE", gym.spaces.Discrete(len(cfg["BUILDING_BLOCKS"])))
    return cfg


@pytest.fixture
def make_env(monkeypatch):
    monkeypatch.setattr(env_mod, "Mol2Vec", _Float64Mol2Vec)
    return lambda **kwargs: env_mod.MolEnvValueMax(_env_config(**kwargs))


def test_env_real_mol2vec():
    env = env_mod.MolEnvValueMax(_env_config())
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for action in [0, 1, 2]:
        obs, reward, done, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert get_mol(env.base_smiles) is not None
        if done or truncated:
            break


def test_gymnasium_contract(make_env):
    check_env(make_env(), skip_render_check=True)


def test_observations_follow_declared_dtype(make_env):
    env = make_env()
    obs, _ = env.reset()
    assert obs.dtype == np.float32
    obs, *_ = env.step(0)
    assert env.observation_space.contains(obs)


def test_dead_end_preserves_molecule_and_completion_reward(make_env):
    env = make_env(BUILDING_BLOCKS=["FC(F)(F)F"], FINAL_WEIGHT=2.0)
    before, _ = env.reset()
    obs, reward, done, truncated, info = env.step(0)
    assert done and not truncated
    assert reward == pytest.approx(2.0 * _atom_score("C"))
    assert info["gen_smiles"] == []
    assert info["score"] == pytest.approx(_atom_score("C"))
    assert np.array_equal(obs, before)
    assert env.observation_space.contains(obs)


def test_non_growing_episode_truncates_without_completion_bonus(make_env):
    env = make_env(BUILDING_BLOCKS=["C"], MAX_EPISODE_STEPS=2)
    env.reset()
    _, reward, done, truncated, _ = env.step(0)
    assert not done and not truncated and reward == 0.0
    obs, reward, done, truncated, info = env.step(0)
    assert not done and truncated and reward == 0.0
    assert env.observation_space.contains(obs)
    assert info["score"] == pytest.approx(_atom_score("C"))


def test_terminal_reward_uses_final_weight(make_env):
    env = make_env(BUILDING_BLOCKS=["CC"], LENGTH=1, FINAL_WEIGHT=3.0)
    env.reset()
    _, reward, done, truncated, info = env.step(0)
    assert done and not truncated
    assert reward == pytest.approx(info["score"] * 3.0)


def test_score_regression_compares_with_reset_molecule(make_env):
    env = make_env(SCORE_FUNCTION=lambda s: 0.8 if get_mol(s).GetNumAtoms() == 1 else 0.2)
    env.reset()
    _, reward, done, truncated, info = env.step(0)
    assert done and not truncated
    assert info["prev_score"] == pytest.approx(0.8)
    assert reward == pytest.approx(0.2)


@pytest.mark.parametrize("action", [-1, 3, None, [0]])
def test_invalid_action_is_rejected(make_env, action):
    env = make_env()
    env.reset()
    with pytest.raises(ValueError, match="ACTION_SPACE"):
        env.step(action)
    assert env.env_step == 0


@pytest.mark.parametrize("overrides", [
    {"BUILDING_BLOCKS": ["invalid"]},
    {"BUILDING_BLOCKS": ["CC"], "ACTION_SPACE": gym.spaces.Discrete(2)},
    {"BASE_SMILES": "invalid"},
    {"RECOND_EVERY": "sometimes"},
    {"MAX_EPISODE_STEPS": 0},
    {"BUILDING_BLOCKS": ["C[C@H](O)F"]},
    {"BASE_SMILES": "C/C=C/C"},
])
def test_invalid_config_is_rejected(make_env, overrides):
    with pytest.raises(ValueError):
        make_env(**overrides)


def test_proposer_refresh_is_atomic_and_checks_pool_size(make_env):
    env = make_env()
    original = list(env.building_blocks)
    env._proposer = SimpleNamespace(propose=lambda context: ["CC"])
    with pytest.raises(ValueError, match="per action"):
        env._decode_pool(np.zeros(300))
    assert env.building_blocks == original
    assert [get_smiles(m) for m in env.mols] == [get_smiles(get_mol(s)) for s in original]


def test_nonfinite_scores_fail_before_entering_training(make_env):
    env = make_env(SCORE_FUNCTION=lambda s: float("nan"))
    with pytest.raises(ValueError, match="finite scores"):
        env.reset()


def _objective():
    return SimpleNamespace(wrapped_objective=SimpleNamespace(score=_atom_score))


def test_legacy_config_sets_model_used_by_policy():
    from moldr.config import get_default_config_v1

    config = get_default_config_v1(_objective(), ["CC"], "unused", num_workers=0, num_gpus=0)
    config.validate()
    assert not config.enable_rl_module_and_learner
    assert config.model["fcnet_hiddens"] == [256, 128, 128]
    assert config.train_batch_size == 4000
    assert config.num_gpus == 0


@pytest.mark.parametrize("num_gpus,learners,gpus_per_learner", [(0, 0, 0), (1, 0, 1), (2, 2, 1)])
def test_new_stack_config_honors_gpu_budget(num_gpus, learners, gpus_per_learner):
    from moldr.config import get_default_config_v2

    config = get_default_config_v2(_objective(), ["CC"], "unused", num_workers=0, num_gpus=num_gpus)
    config.validate()
    assert config.num_learners == learners
    assert config.num_gpus_per_learner == gpus_per_learner
