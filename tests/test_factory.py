"""Shared factory contracts and old Python/RLlib entry points, without Ray build."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
from gensim.models import Word2Vec

from moldr.config import get_default_config_v1, get_default_config_v2
from moldr.e2e.config import get_e2e_config
from moldr.e2e.env_v2 import MolEnvE2EV2
from moldr.env import MolEnvValueMax
from moldr.experiment import ExperimentConfig, load_config
from moldr.factory import (
    make_env_config,
    make_environment,
    make_ppo_config,
    resolve_config,
)
from moldr.scoring import ScoringService, register_objective


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "mol2vec.pkl"
    model = Word2Vec(
        sentences=[["UNK", "a", "b"]], min_count=1, vector_size=3, workers=1, seed=0
    )
    model.save(str(path))
    result = ExperimentConfig()
    result.fragments.smiles = ["CC", "CO"]
    result.representation.model_path = str(path)
    result.representation.obs_dim = 3
    result.environment.atom_limit = 5
    result.environment.max_attach = 8
    result.training.num_workers = 0
    result.training.iterations = 0
    return result


def test_new_factory_default_is_versioned_e2e_old_rllib_stack(config):
    from moldr.e2e.action_dist import ACTION_DIST_NAME

    ppo = make_ppo_config(config)
    assert ppo.env == "moldr_e2e_v2"
    assert ppo.model["custom_model"] == "moldr_pointer_v2"
    assert ppo.model["custom_action_dist"] == ACTION_DIST_NAME
    assert not ppo.enable_rl_module_and_learner
    assert not ppo.enable_env_runner_and_connector_v2
    assert ppo._disable_preprocessor_api
    assert ppo.env_config["REWARD_MODE"] == "improvement"
    assert ppo.env_config["SCORING"] == {
        "objective": "QED",
        "input_mode": "smiles",
        "cache_size": 0,
    }
    assert ppo.model["custom_model_config"]["candidate_value"] is False
    assert ppo.validate() is None


def test_explicit_legacy_factory_keeps_original_network_and_reward(config):
    config.environment.mode = "legacy"
    ppo = make_ppo_config(config)
    assert ppo.env == "moldr_env"
    assert ppo.model["custom_model"] is None
    assert ppo.model["custom_action_dist"] is None
    assert ppo.model["fcnet_hiddens"] == [256, 128, 128]
    assert ppo.model["fcnet_activation"] == "relu"
    assert ppo.env_config["REWARD_MODE"] == "sparse"
    assert ppo.env_config["TERMINATE_ON_SCORE_DECREASE"]
    assert isinstance(ppo.env_config["ACTION_SPACE"], gym.spaces.Discrete)
    assert ppo.env_config["OBS_SPACE"].shape == (3,)
    assert not ppo.enable_rl_module_and_learner
    assert ppo.validate() is None


def test_new_legacy_factory_uses_explicit_hidden_width(config):
    config.environment.mode = "legacy"
    config.training.hidden = 16
    ppo = make_ppo_config(config)
    assert ppo.model["fcnet_hiddens"] == [16, 128, 128]
    assert not ppo.enable_rl_module_and_learner


def test_old_python_config_names_and_api_generations_unchanged():
    objective = SimpleNamespace(
        wrapped_objective=SimpleNamespace(score=lambda smiles: 0.5)
    )
    common = {
        "sc": objective,
        "building_blocks_smiles": ["CC", "CO"],
        "model_path": "unused",
        "num_workers": 0,
        "num_gpus": 0,
    }
    v1 = get_default_config_v1(**common)
    v2 = get_default_config_v2(**common)
    old_e2e = get_e2e_config(**common)
    assert v1.env == v2.env == "moldr_env"
    assert not v1.enable_rl_module_and_learner
    assert v2.enable_rl_module_and_learner
    assert old_e2e.env == "moldr_e2e"
    assert old_e2e.model["custom_model"] == "moldr_pointer"
    assert all(ppo.model["custom_action_dist"] is None for ppo in (v1, v2, old_e2e))
    assert "context" not in old_e2e.model["custom_model_config"]
    assert "SCORING" not in v1.env_config


@pytest.mark.parametrize("representation", ["mol2vec", "mpnn"])
@pytest.mark.parametrize("max_attach", [1, 8])
def test_v2_factory_resolves_actual_registered_distribution_and_action_width(
    config, representation, max_attach
):
    import torch
    from ray.rllib.models import ModelCatalog

    from moldr.e2e.action_dist import ACTION_DIST_NAME, StableMaskedTorchCategorical

    config.representation.name = representation
    config.environment.max_attach = max_attach
    ppo = make_ppo_config(config)
    actions = max(len(config.fragments.smiles), max_attach)
    space = gym.spaces.Discrete(actions)
    distribution, width = ModelCatalog.get_action_dist(space, ppo.model, framework="torch")
    assert ppo.model["custom_action_dist"] == ACTION_DIST_NAME
    assert distribution is StableMaskedTorchCategorical
    assert width == actions
    assert distribution.required_model_output_shape(space, ppo.model) == actions
    actual = distribution(torch.zeros(2, width), model=None)
    assert actual.dist.probs.shape == (2, actions)
    assert actual.deterministic_sample().shape == (2,)


def test_v2_registration_leaves_default_legacy_distribution_unchanged(config):
    from ray.rllib.models import ModelCatalog
    from ray.rllib.models.torch.torch_action_dist import TorchCategorical

    make_ppo_config(config)
    config.environment.mode = "legacy"
    legacy = make_ppo_config(config)
    distribution, width = ModelCatalog.get_action_dist(
        gym.spaces.Discrete(2), legacy.model, framework="torch"
    )
    assert legacy.model["custom_action_dist"] is None
    assert distribution is TorchCategorical and width == 2


@pytest.mark.parametrize("mode,cls", [("e2e", MolEnvE2EV2), ("legacy", MolEnvValueMax)])
def test_factory_environment_dispatch_and_actual_reset(config, mode, cls):
    config.environment.mode = mode
    env = make_environment(config)
    assert isinstance(env, cls)
    observation, _ = env.reset(seed=4)
    assert env.observation_space.contains(observation)
    env.close()


@pytest.mark.parametrize("mode,alpha", [("actor", 0.0), ("actor_value", 2.0)])
def test_candidate_value_ablation_forwarded_to_rllib(config, mode, alpha):
    config.selection.mode = mode
    config.selection.candidate_value = True
    config.selection.alpha = 2.0
    config.selection.temperature = 0.5
    config.selection.auxiliary_loss_coeff = 0.25
    model = make_ppo_config(config).model["custom_model_config"]
    assert model["candidate_value"]
    assert model["alpha"] == alpha
    assert model["temperature"] == 0.5
    assert model["auxiliary_loss_coeff"] == 0.25


@pytest.mark.parametrize("trainable", [False, True])
def test_mpnn_shapes_and_training_knobs_forwarded(config, trainable):
    config.representation.name = "mpnn"
    config.representation.trainable = trainable
    config.representation.layers = 2
    config.representation.hidden = 32
    config.representation.output = 16
    config.representation.atom_chunk_size = 48
    config.representation.gradient_checkpointing = True
    config.representation.embedding_cache_size = 0 if trainable else 4
    ppo = make_ppo_config(config)
    model = ppo.model["custom_model_config"]
    assert model["representation"] == "mpnn" and model["trainable"] == trainable
    assert (model["layers"], model["graph_hidden"], model["graph_output"]) == (
        2,
        32,
        16,
    )
    assert model["atom_chunk_size"] == 48 and model["gradient_checkpointing"]
    assert model["catalog_smiles"] == ["CC", "CO"]
    assert model["max_attach"] == 8
    assert ppo.env_config["REPRESENTATION"]["max_atoms"] == 7
    assert ppo.env_config["REPRESENTATION"]["max_bonds"] == 21
    assert config.representation.max_atoms is None


def test_ppo_training_settings_forwarded_without_build(config):
    config.training.num_workers = 2
    config.training.num_gpus = 1.0
    config.training.train_batch_size = 1024
    config.training.minibatch_size = 32
    config.training.num_epochs = 3
    config.training.lr = 0.001
    config.training.gamma = 0.8
    config.training.lambda_ = 0.7
    config.training.entropy_coeff = 0.03
    config.training.rollout_fragment_length = 128
    config.training.sample_timeout_s = 60.0
    config.training.seed = 2
    ppo = make_ppo_config(config)
    assert ppo.num_env_runners == 2 and ppo.num_gpus == 1.0
    assert (ppo.train_batch_size, ppo.minibatch_size, ppo.num_epochs) == (1024, 32, 3)
    assert (ppo.lr, ppo.gamma, ppo.lambda_, ppo.entropy_coeff) == (
        0.001,
        0.8,
        0.7,
        0.03,
    )
    assert ppo.rollout_fragment_length == 128 and ppo.sample_timeout_s == 60.0
    assert ppo.seed == ppo.env_config["SEED"] == 2


def test_chemistry_scoring_and_cache_reuse_config_forwarded(config):
    config.chemistry.map_storage = "string"
    config.chemistry.shared_preparation = False
    config.chemistry.bond_scan = "python"
    config.chemistry.candidate_representation = "mapless"
    config.chemistry.dedup_key = "aromatic"
    config.chemistry.mol_transfer = "direct"
    config.scoring.input_mode = "mol"
    config.scoring.cache_size = 12
    config.representation.reuse_embeddings = False
    env_config = make_env_config(config)
    assert env_config["CHEMISTRY"] == {
        "map_storage": "string",
        "shared_preparation": False,
        "bond_scan": "python",
        "candidate_representation": "mapless",
        "dedup_key": "aromatic",
        "mol_transfer": "direct",
    }
    assert env_config["SCORING"] == {
        "objective": "QED",
        "input_mode": "mol",
        "cache_size": 12,
    }
    assert not env_config["REPRESENTATION"]["reuse_embeddings"]


@pytest.mark.parametrize("objective", ["QED", "qed", "QED_SA", "logp8", "logp-1"])
def test_registered_mol_objective_preflight_and_scoring(config, objective):
    config.scoring.objective = objective
    config.scoring.input_mode = "mol"
    resolved, _, _ = resolve_config(config)
    scorer = ScoringService(resolved.scoring.objective, input_mode="mol")
    assert scorer.is_registered and scorer.deterministic
    assert scorer.score("CCO") == pytest.approx(ScoringService(objective).score("CCO"))


@pytest.mark.parametrize("kind", ["unknown", "missing_mol", "nondeterministic"])
def test_unusable_objectives_fail_in_factory_before_any_ray_build(config, kind):
    if kind == "unknown":
        config.scoring.objective = "missing-objective"
    elif kind == "missing_mol":
        register_objective("factory-no-mol", len)
        config.scoring.objective = "factory-no-mol"
        config.scoring.input_mode = "mol"
    else:
        register_objective("factory-random", len, deterministic=False)
        config.scoring.objective = "factory-random"
        config.scoring.cache_size = 2
    with pytest.raises(ValueError):
        make_ppo_config(config)


def test_lookahead_preflight_requires_full_fixed_table(config):
    config.evaluation.selector = "value_lookahead"
    resolved, _, _ = resolve_config(config)
    assert resolved.evaluation.selector == "value_lookahead"
    config.fragments.pool_size = 1
    with pytest.raises(ValueError, match="full fragment table"):
        resolve_config(config)


def test_resolver_does_not_mutate_original_config(config):
    before = config.to_dict()
    resolved, blocks, catalog = resolve_config(config)
    assert resolved.environment.max_episode_steps == 10
    assert blocks == catalog == ["CC", "CO"]
    assert config.to_dict() == before


def test_no_files_needed_to_load_default_config():
    loaded = load_config()
    assert loaded.environment.mode == "e2e"
    assert loaded.scoring.objective == "QED"


def test_embedding_dimension_mismatch_is_rejected_before_ray(config):
    config.representation.obs_dim = 4
    with pytest.raises(ValueError, match="dimension"):
        resolve_config(config)


@pytest.mark.parametrize("cadence", ["episode", "step"])
def test_untrained_context_proposer_is_rejected_during_factory_preflight(
    config, tmp_path, cadence
):
    from moldr.generative.proposer import VAEProposer
    from moldr.generative.vae import FragmentVAE
    from moldr.generative.vocab import FragmentVocab

    vocab = FragmentVocab.from_smiles_list(["CC", "CO", "C1CC1"])
    model = FragmentVAE(len(vocab), context_dim=3, hidden_dim=8)
    prop = VAEProposer(model, vocab, n_blocks=2, min_ring_fraction=0)
    prop.save(tmp_path / "proposer")
    config.fragments.source = "vae"
    config.fragments.smiles = []
    config.fragments.proposer_dir = str(tmp_path / "proposer")
    config.fragments.recond_every = cadence
    with pytest.raises(ValueError, match="context|conditional"):
        resolve_config(config)
    config.fragments.recond_every = "never"
    resolved, blocks, _ = resolve_config(config)
    assert resolved.fragments.recond_every == "never"
    assert len(blocks) == 2


def test_trained_context_proposer_passes_factory_preflight(config, tmp_path):
    import numpy as np

    from moldr.generative.proposer import VAEProposer
    from moldr.generative.vae import train_fragment_vae
    from moldr.generative.vocab import FragmentVocab

    vocab = FragmentVocab.from_smiles_list(["CC", "CO", "C1CC1"])
    model = train_fragment_vae(
        vocab,
        contexts=np.ones((len(vocab), 3)),
        context_dim=3,
        hidden_dim=8,
        epochs=1,
        seed=0,
    )
    prop = VAEProposer(model, vocab, n_blocks=2, min_ring_fraction=0)
    prop.save(tmp_path / "proposer")
    config.fragments.source = "vae"
    config.fragments.smiles = []
    config.fragments.proposer_dir = str(tmp_path / "proposer")
    config.fragments.recond_every = "episode"
    resolved, blocks, _ = resolve_config(config)
    assert resolved.fragments.recond_every == "episode"
    assert len(blocks) == 2
