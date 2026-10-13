
import pytest
import yaml

from moldr.experiment import ExperimentConfig, config_from_dict, load_config
from moldr.factory import make_environment, resolve_config
from moldr.provenance import load_saved_config, save_run


def test_cli_overrides_only_explicit_values(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("training:\n  seed: 7\n  num_workers: 2\nscoring:\n  objective: logp8\n")
    config = load_config(path, ["training.seed=9"], {"training.num_workers": 1, "scoring.objective": None})
    assert config.training.seed == 9
    assert config.training.num_workers == 1
    assert config.scoring.objective == "logp8"


@pytest.mark.parametrize("override", ["training.seeed=0", "bad.seed=1", "training=1", "training.seed=true",
                                      "chemistry.shared_preparation=1", "training.num_workers=-1",
                                      "training.lr=.nan", "representation.name=graphormer",
                                      "chemistry.mol_transfer=direct", "representation.trainable=true",
                                      "selection.mode=actor_value", "evaluation.selector=value_lookahead"])
def test_invalid_configs_fail_before_build(override):
    base = ExperimentConfig()
    if override == "evaluation.selector=value_lookahead":
        base.fragments.pool_size = 2
    with pytest.raises(ValueError):
        load_config(overrides=[override], base=base)


def test_e2e_and_explicit_legacy_defaults():
    e2e = load_config()
    legacy = load_config(overrides=["environment.mode=legacy"])
    assert e2e.environment.reward == "improvement"
    assert not e2e.environment.terminate_on_score_decrease
    assert legacy.environment.reward == "sparse"
    assert legacy.environment.terminate_on_score_decrease
    from moldr.config import get_default_config_v1 as old
    from moldr.legacy import get_default_config_v1
    assert old is get_default_config_v1


def test_unknown_nested_key():
    with pytest.raises(ValueError, match="training.missing"):
        config_from_dict({"training": {"missing": 5}})


def test_capacity_derived_and_never_truncated():
    config = load_config(overrides=["fragments.smiles=[CC,CO]", "representation.name=mpnn", "environment.atom_limit=5"])
    resolved, _, _ = resolve_config(config)
    assert resolved.representation.max_atoms == 7
    assert resolved.representation.max_bonds == 21
    env = make_environment(resolved)
    observation, _ = env.reset(seed=2)
    assert env.observation_space.contains(observation)
    assert observation["candidate_atoms"].shape[0] == config.environment.max_attach
    config.representation.max_atoms = 2
    with pytest.raises(ValueError, match="derived capacity"):
        resolve_config(config)


def test_mol_scoring_alias_and_legacy_service():
    config = load_config(overrides=["environment.mode=legacy", "fragments.smiles=[CC,CO]", "scoring.objective=qed", "scoring.input_mode=mol"])
    env = make_environment(config)
    env.reset(seed=2)
    env.step(0)
    assert env.scorer.requests > 1
    assert env.scorer.stats["requests"] == env.scorer.stats["computations"]


def test_saved_config_uses_snapshots_and_rejects_shape_changes(tmp_path):
    config = load_config(overrides=["fragments.smiles=[CC,CO]", "representation.name=mpnn", "environment.atom_limit=5"])
    config, blocks, catalog = resolve_config(config)
    directory = save_run(tmp_path / "run", config, blocks=blocks, catalog=catalog)
    checkpoint = directory / "checkpoints" / "iteration1"
    checkpoint.mkdir(parents=True)
    restored, found = load_saved_config(checkpoint, ["evaluation.episodes=2", "training.num_gpus=0"])
    assert found == directory
    assert restored.evaluation.episodes == 2
    with pytest.raises(ValueError, match="not allowed"):
        load_saved_config(checkpoint, ["representation.output=64"])
    with pytest.raises(ValueError):
        load_saved_config(checkpoint, ["environment.mode=legacy"])
    (directory / "fragment_catalog.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        load_saved_config(checkpoint)


def test_yaml_roundtrip(tmp_path):
    config = load_config(overrides=["fragments.smiles=[CC,CO]", "selection.candidate_value=true"])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    assert load_config(path) == config
