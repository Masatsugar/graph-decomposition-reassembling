"""Direct evaluation restores protected settings and saves full provenance."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from moldr import cli, provenance
from moldr.experiment import ExperimentConfig
from moldr.sampling import EpisodeResult, Sampler


def test_explicit_checkpoint_cannot_bypass_saved_model_guard(monkeypatch, tmp_path):
    original = ExperimentConfig().validate()
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(original.to_dict()))
    monkeypatch.setattr(
        provenance,
        "load_saved_config",
        lambda checkpoint: (copy.deepcopy(original), tmp_path),
    )
    altered = copy.deepcopy(original)
    altered.training.hidden = 128
    with pytest.raises(ValueError, match="training.hidden"):
        cli.run_evaluation(altered, checkpoint=tmp_path / "checkpoint")
    assert altered.output.checkpoint is None


@pytest.mark.parametrize("restore_error", [False, True])
def test_evaluation_provenance_and_owned_ray_cleanup(
    monkeypatch, tmp_path, restore_error
):
    import ray

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "algorithm_state.pkl").write_bytes(b"saved-state")
    config = ExperimentConfig()
    config.training.num_workers = 0
    config.training.num_gpus = 0.0
    config.evaluation.episodes = 1
    config.output.directory = str(tmp_path)
    config.output.run_name = "evaluation"
    flags = {"initialized": False, "stopped": False, "closed": False, "shutdown": False}

    def prepare(value):
        assert value.output.checkpoint == str(checkpoint)
        return value

    def save(directory, value, *, blocks, catalog):
        assert blocks == catalog == ["CC"]
        directory = Path(directory)
        directory.mkdir()
        (directory / "resolved_config.yaml").write_text(yaml.safe_dump(value.to_dict()))
        (directory / "manifest.json").write_text(
            json.dumps(
                {"seed": value.training.seed, "assets": [], "versions": {"ray": "stub"}}
            )
        )
        return directory

    def restore(path):
        assert path == str(checkpoint)
        if restore_error:
            raise RuntimeError("restore failed")

    def shutdown():
        flags["initialized"] = False
        flags["shutdown"] = True

    scorer = SimpleNamespace(stats={"requests": 2, "computations": 2, "hits": 0})
    env = SimpleNamespace(scorer=scorer, close=lambda: flags.update(closed=True))
    algo = SimpleNamespace(restore=restore, stop=lambda: flags.update(stopped=True))
    monkeypatch.setattr(cli, "prepare_restored_config", prepare)
    monkeypatch.setattr(
        cli, "resolve_config", lambda value: (value.validate(), ["CC"], ["CC"])
    )
    monkeypatch.setattr(cli, "save_run", save)
    monkeypatch.setattr(cli, "make_environment", lambda value: env)
    monkeypatch.setattr(
        cli, "make_ppo_config", lambda value: SimpleNamespace(build=lambda: algo)
    )
    monkeypatch.setattr(cli, "_init_ray", lambda value: flags.update(initialized=True))
    monkeypatch.setattr(ray, "is_initialized", lambda: flags["initialized"])
    monkeypatch.setattr(ray, "shutdown", shutdown)
    monkeypatch.setattr(
        Sampler,
        "episode",
        lambda self, **kwargs: EpisodeResult(
            "CC", 0.5, 0.1, 2, True, False, scoring=scorer.stats
        ),
    )
    if restore_error:
        with pytest.raises(RuntimeError, match="restore failed"):
            cli.run_evaluation(config, checkpoint=checkpoint)
    else:
        result = cli.run_evaluation(config, checkpoint=checkpoint)
        assert (result / "manifest.json").is_file()
        reference = json.loads((result / "evaluation_checkpoint.json").read_text())
        assert (
            reference["sha256"]["algorithm_state.pkl"]
            == hashlib.sha256(b"saved-state").hexdigest()
        )
        assert json.loads((result / "evaluation.json").read_text())["episodes"] == 1
    assert config.output.checkpoint is None
    assert flags["closed"] and flags["stopped"] and flags["shutdown"]
