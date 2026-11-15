"""A copied checkpoint drops logging configuration without changing training state."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cloudpickle
import numpy as np
import pytest

from scripts.migrate_ray_checkpoint import migrate_checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    source = tmp_path / "original"
    checkpoint = source / "checkpoints" / "iteration_000001"
    policy = checkpoint / "policies" / "default_policy"
    policy.mkdir(parents=True)
    config = {"logger_creator": lambda _: None, "logger_config": None, "seed": 7}
    algorithm = {"config": config, "training_iteration": 1, "counters": {"steps": 64}}
    state = {
        "policy_spec": {"config": config},
        "weights": {"layer": np.arange(12, dtype=np.float32).reshape(3, 4)},
        "_optimizer_variables": [{"step": 1, "momentum": np.ones(4)}],
        "global_timestep": 64,
    }
    for directory, filename, data in (
        (checkpoint, "algorithm_state.pkl", algorithm),
        (policy, "policy_state.pkl", state),
    ):
        with (directory / filename).open("wb") as stream:
            cloudpickle.dump(data, stream)
        (directory / "rllib_checkpoint.json").write_text(
            json.dumps(
                {"ray_version": "2.47.1", "state_file": str(directory / filename)}
            )
        )
    (source / "resolved_config.yaml").write_text("training:\n  seed: 7\n")
    (source / "asset.pkl").write_bytes(b"not a checkpoint; do not deserialize")
    return source


def hashes(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_migration_preserves_original_and_training_state(checkpoint, tmp_path):
    original_hashes = hashes(checkpoint)
    destination = tmp_path / "migrated"
    result = migrate_checkpoint(checkpoint, destination)
    assert hashes(checkpoint) == original_hashes
    assert len(result["changes"]) == 2
    for name in ("resolved_config.yaml", "asset.pkl"):
        assert (destination / name).read_bytes() == (checkpoint / name).read_bytes()
    for path in destination.rglob("*.pkl"):
        if path.name == "asset.pkl":
            continue
        original = cloudpickle.loads(
            (checkpoint / path.relative_to(destination)).read_bytes()
        )
        migrated = cloudpickle.loads(path.read_bytes())
        if path.name == "algorithm_state.pkl":
            config = migrated.pop("config")
            original.pop("config")
            assert migrated == original
        else:
            config = migrated["policy_spec"]["config"]
            assert migrated["global_timestep"] == original["global_timestep"]
            np.testing.assert_array_equal(
                migrated["weights"]["layer"], original["weights"]["layer"]
            )
            assert (
                migrated["_optimizer_variables"][0]["step"]
                == original["_optimizer_variables"][0]["step"]
            )
            np.testing.assert_array_equal(
                migrated["_optimizer_variables"][0]["momentum"],
                original["_optimizer_variables"][0]["momentum"],
            )
        assert config == {"seed": 7}
        metadata = json.loads((path.parent / "rllib_checkpoint.json").read_text())
        assert metadata == {"ray_version": "2.47.1", "state_file": str(path)}


def test_existing_destination_is_never_overwritten(checkpoint, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep").write_text("keep")
    before = hashes(checkpoint)
    with pytest.raises(FileExistsError):
        migrate_checkpoint(checkpoint, destination)
    assert (destination / "keep").read_text() == "keep"
    assert hashes(checkpoint) == before


def test_nested_destination_is_rejected(checkpoint):
    with pytest.raises(ValueError, match="outside"):
        migrate_checkpoint(checkpoint, checkpoint / "nested")
    assert not (checkpoint / "nested").exists()


def test_dangling_destination_symlink_is_not_followed(checkpoint, tmp_path):
    destination = tmp_path / "destination"
    target = tmp_path / "missing-target"
    destination.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        migrate_checkpoint(checkpoint, destination)
    assert destination.is_symlink()
    assert not target.exists()


def test_failed_migration_removes_only_its_copy(checkpoint, tmp_path):
    path = next(checkpoint.rglob("algorithm_state.pkl"))
    path.write_bytes(cloudpickle.dumps({"config": "unexpected configuration"}))
    before = hashes(checkpoint)
    destination = tmp_path / "failed"
    with pytest.raises(TypeError, match="configuration dictionary"):
        migrate_checkpoint(checkpoint, destination)
    assert not destination.exists()
    assert hashes(checkpoint) == before


def test_help_does_not_need_installed_dependencies():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "migrate_ray_checkpoint.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "original Python/Ray environment" in result.stdout
