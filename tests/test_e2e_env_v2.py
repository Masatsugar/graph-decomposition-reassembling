"""Versioned observations, state reuse, and nonmutating value evaluation."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from moldr.chemutils import get_mol
from moldr.e2e.env import MolEnvE2E
from moldr.e2e.env_v2 import MolEnvE2EV2
from moldr.evaluation import value_lookahead
from moldr.sampling import Sampler


class CountingEmbedding:
    def __init__(self, **kwargs):
        self.calls = []

    def fit_transform(self, mols):
        self.calls.append([m.GetNumAtoms() for m in mols])
        return np.asarray(
            [[m.GetNumAtoms(), m.GetNumBonds(), 0.5] for m in mols], dtype=np.float32
        )


@pytest.fixture(autouse=True)
def embedding_stub(monkeypatch):
    monkeypatch.setattr("moldr.e2e.env_v2.Mol2Vec", CountingEmbedding)
    monkeypatch.setattr("moldr.e2e.env.Mol2Vec", CountingEmbedding)


def atoms_score(smiles):
    return float(get_mol(smiles).GetNumAtoms()) / 50


def make_env(**overrides):
    config = {
        "BUILDING_BLOCKS": ["CC", "CO", "C1CCCCC1"],
        "BASE_SMILES": "C",
        "MODEL_PATH": "unused",
        "OBS_DIM": 3,
        "MAX_ATTACH": 8,
        "MAX_EPISODE_STEPS": 6,
        "LENGTH": 20,
        "SCORE_THRESHOLD": 2.0,
        "SCORING": {"objective": "qed"},
    }
    config.update(overrides)
    return MolEnvE2EV2(config)


def test_rich_observation_and_old_checkpoint_contract():
    config = {
        "BUILDING_BLOCKS": ["CC", "CO"],
        "SCORE_FUNCTION": atoms_score,
        "MODEL_PATH": "unused",
        "BASE_SMILES": "C",
        "OBS_DIM": 3,
        "LENGTH": 20,
        "SCORE_THRESHOLD": 2.0,
        "MAX_ATTACH": 8,
        "MAX_EPISODE_STEPS": 6,
    }
    old = MolEnvE2E(config)
    new = MolEnvE2EV2(config)
    obs_old, _ = old.reset(seed=7)
    obs_new, _ = new.reset(seed=7)
    assert set(obs_old) == {"base", "options", "phase", "action_mask"}
    assert set(obs_new) == {*obs_old, "context"}
    assert new.observation_space.contains(obs_new)
    np.testing.assert_array_equal(
        obs_new["context"], np.array([0.02, 0, 6], np.float32)
    )
    for action in [0, 0, 1, 0]:
        out_old = old.step(action)
        out_new = new.step(action)
        assert old.base_smiles == new.base_smiles
        assert out_old[1:4] == out_new[1:4]
        for key in obs_old:
            np.testing.assert_array_equal(out_old[0][key], out_new[0][key])


def test_base_and_selected_candidate_embeddings_are_reused():
    env = make_env()
    observation, _ = env.reset(seed=0)
    before = len(env.mol2vec.calls)
    env._make_obs()
    assert len(env.mol2vec.calls) == before
    observation, *_ = env.step(0)
    assert len(env.mol2vec.calls) == before + 1
    chosen_vector = observation["options"][0].copy()
    before = len(env.mol2vec.calls)
    observation, *_ = env.step(0)
    assert len(env.mol2vec.calls) == before
    np.testing.assert_array_equal(observation["base"], chosen_vector)


def test_reuse_can_be_disabled():
    env = make_env(REPRESENTATION={"name": "mol2vec", "reuse_embeddings": False})
    env.reset(seed=0)
    before = len(env.mol2vec.calls)
    env._make_obs()
    assert len(env.mol2vec.calls) == before + 1


def test_integer_graph_observation_has_candidate_axis_only_max_attach():
    env = make_env(
        BUILDING_BLOCKS=["CC"] * 20,
        REPRESENTATION={"name": "mpnn", "max_atoms": 24, "max_bonds": 40},
    )
    observation, _ = env.reset(seed=0)
    assert observation["fragment_ids"].shape == (20,)
    assert observation["candidate_atoms"].shape == (8, 24, 8)
    assert observation["candidate_counts"].sum() == 0
    assert observation["base_counts"].tolist() == [1, 0]
    for key in (
        "base_atoms",
        "base_bonds",
        "base_counts",
        "candidate_atoms",
        "candidate_bonds",
        "candidate_counts",
        "fragment_ids",
    ):
        assert observation[key].dtype == np.int32
        assert env.observation_space[key].dtype == np.int32
    assert env.mol2vec is None
    assert env.observation_space.contains(observation)
    observation, *_ = env.step(0)
    assert observation["candidate_counts"][0, 0] > 0
    assert np.all(observation["fragment_ids"] == -1)
    assert env.observation_space.contains(observation)


def test_mpnn_keeps_vae_conditioning_dimension_independent(monkeypatch):
    contexts = []

    def propose(context):
        contexts.append(context.copy())
        return ["CC", "CO"]

    proposer = SimpleNamespace(
        vae=SimpleNamespace(context_dim=3),
        vocab=SimpleNamespace(smiles=["CC", "CO"]),
        propose_library=lambda context: ["CC", "CO"],
        propose=propose,
    )
    monkeypatch.setattr(
        "moldr.generative.proposer.VAEProposer.from_spec", lambda spec: proposer
    )
    env = make_env(
        PROPOSER={"instance": proposer},
        OBS_DIM=128,
        REPRESENTATION={"name": "mpnn", "output": 128},
    )
    observation, _ = env.reset(seed=0)
    assert contexts[0].shape == (3,)
    assert env.observation_space.contains(observation)


def test_graph_overflow_is_an_error():
    with pytest.raises(ValueError, match="capacity"):
        make_env(
            BUILDING_BLOCKS=["C1CCCCC1"],
            REPRESENTATION={"name": "mpnn", "max_atoms": 2, "max_bonds": 4},
        )


def snapshot(env):
    return {
        "base": env.base_smiles,
        "mol": env.base_mol.ToBinary(),
        "phase": env.phase,
        "step": env.env_step,
        "score": env.prev_score,
        "candidates": list(env._candidate_smiles),
        "rng": copy.deepcopy(env.np_random.bit_generator.state),
        "pool": env._bb_pool_idx.copy(),
        "cache_keys": list(env._graph_cache),
    }


@pytest.mark.parametrize("representation", ["mol2vec", "mpnn"])
def test_lookahead_does_not_change_state_and_counts_k_plus_one(representation):
    env = make_env(
        REPRESENTATION={"name": representation},
        MAX_EPISODE_STEPS=2,
        SCORING={"objective": "qed", "cache_size": 16},
    )
    env.reset(seed=0)
    env.step(2)
    state = snapshot(env)
    start = env.scorer.requests
    choice = value_lookahead(env, lambda obs: float(obs["context"][0]) + 0.5, gamma=0.9)
    assert choice.score_requests == len(env._candidates)
    assert all(choice.truncated)
    assert not any(choice.terminated)
    np.testing.assert_allclose(
        choice.returns,
        np.asarray(choice.rewards) + 0.9 * np.asarray(choice.state_values),
    )
    after = snapshot(env)
    for key in state:
        if key == "pool":
            np.testing.assert_array_equal(state[key], after[key])
        else:
            assert state[key] == after[key]
    count = len(env._candidates)
    env.step(choice.action)
    assert env.scorer.requests - start == count + 1
    assert env.scorer.hits >= 1


def test_true_terminal_does_not_bootstrap():
    env = make_env(SCORE_THRESHOLD=0.0)
    env.reset(seed=0)
    env.step(0)

    def value_should_not_be_called(obs):
        raise AssertionError("A true terminal has no bootstrap")

    choice = value_lookahead(env, value_should_not_be_called)
    assert all(choice.terminated)
    assert choice.returns == choice.rewards


@pytest.mark.parametrize(
    "overrides",
    [
        {"OPTION_POOL_SIZE": 2},
        {"SCORING": {"objective": atoms_score}},
    ],
)
def test_lookahead_rejects_unsupported_modes(overrides):
    env = make_env(**overrides)
    with pytest.raises(ValueError, match="value_lookahead"):
        Sampler(env, selector="value_lookahead")


def test_seeded_random_sampler_uses_attachment_mask_and_selected_molecule():
    env = make_env(OPTION_POOL_SIZE=2, MAX_EPISODE_STEPS=4)
    sampler = Sampler(env)
    first = sampler.episode(seed=42, random=True)
    second = sampler.episode(seed=42, random=True)
    assert first.smiles == second.smiles == env.base_smiles
    assert first.steps == 4 and first.truncated
    assert [i["chosen_bb"] for i in first.infos if "chosen_bb" in i] == [
        i["chosen_bb"] for i in second.infos if "chosen_bb" in i
    ]


@pytest.mark.parametrize("mode", ["dead_end", "truncated_bb", "truncated_attach"])
def test_sampler_returns_actual_final_molecule_for_every_ending(mode):
    overrides = {
        "BUILDING_BLOCKS": ["CC"],
        "MAX_EPISODE_STEPS": 1 if mode == "truncated_bb" else 2,
    }
    if mode == "dead_end":
        overrides["BUILDING_BLOCKS"] = ["FC(F)(F)F"]
    env = make_env(**overrides)
    sampler = Sampler(env, SimpleNamespace(compute_single_action=lambda obs: 0))
    result = sampler.episode(seed=1)
    assert result.smiles == env.base_smiles
    if mode in {"dead_end", "truncated_bb"}:
        assert result.smiles == "C"
    else:
        assert result.smiles == "CC"
    assert sampler.sampling((0, 1))["gen_smiles"] == [result.smiles]


def test_sampler_rejects_masked_policy_action():
    env = make_env(BUILDING_BLOCKS=["CC"])
    sampler = Sampler(env, SimpleNamespace(compute_single_action=lambda obs: 7))
    with pytest.raises(ValueError, match="masked"):
        sampler.episode(seed=0)


def test_exploration_seed_is_independent_of_prior_torch_draws():
    import torch

    env = make_env(MAX_EPISODE_STEPS=4)

    def stochastic_action(observation, explore=False):
        assert explore
        return int(torch.multinomial(torch.as_tensor(observation["action_mask"]), 1)[0])

    sampler = Sampler(
        env, SimpleNamespace(compute_single_action=stochastic_action), explore=True
    )
    first = sampler.episode(seed=42)
    torch.rand(100)
    second = sampler.episode(seed=42)
    assert first.smiles == second.smiles
    assert [info.get("chosen_bb") for info in first.infos] == [
        info.get("chosen_bb") for info in second.infos
    ]
    with pytest.raises(ValueError, match="max_steps"):
        Sampler(env, max_steps=0)
