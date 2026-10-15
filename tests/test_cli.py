"""CLI config/restore dispatch regressions without starting Ray or training."""

import copy
from pathlib import Path

import pytest
import yaml

from moldr import cli
from moldr.experiment import load_config
from moldr.factory import resolve_config
from moldr.provenance import load_saved_config, prepare_restored_config, save_run


@pytest.fixture
def saved_run(tmp_path):
    config = load_config(
        overrides=[
            "fragments.smiles=[CC,CO]",
            "representation.name=mpnn",
            "representation.trainable=true",
            "representation.hidden=8",
            "representation.output=8",
            "selection.mode=actor_value",
            "selection.candidate_value=true",
            "training.hidden=16",
            "training.seed=7",
            "training.num_workers=0",
            "training.train_batch_size=128",
            "training.minibatch_size=32",
            "environment.atom_limit=5",
            "environment.max_attach=4",
            "scoring.objective=logp8",
        ]
    )
    config, blocks, catalog = resolve_config(config)
    directory = save_run(tmp_path / "saved", config, blocks=blocks, catalog=catalog)
    checkpoint = directory / "checkpoints" / "iteration_000001"
    checkpoint.mkdir(parents=True)
    return config, directory, checkpoint


@pytest.fixture
def dispatch(monkeypatch):
    calls = []

    def training(config, *, profile=False):
        calls.append(
            ("profile" if profile else "train", config, config.output.checkpoint)
        )
        return config

    def evaluation(config, checkpoint=None):
        calls.append(("evaluation", config, checkpoint))
        return config

    monkeypatch.setattr(cli, "run_training", training)
    monkeypatch.setattr(cli, "run_evaluation", evaluation)
    return calls


def checkpoint_args(source, checkpoint, tmp_path, overlay=None):
    overlay = copy.deepcopy(overlay or {})
    if source == "flag":
        args = ["--checkpoint", str(checkpoint)]
        if overlay:
            path = tmp_path / "overlay.yaml"
            path.write_text(yaml.safe_dump(overlay))
            args += ["--config", str(path)]
        return args
    if source == "set":
        args = ["--set", f"output.checkpoint={checkpoint}"]
        if overlay:
            path = tmp_path / "overlay.yaml"
            path.write_text(yaml.safe_dump(overlay))
            args += ["--config", str(path)]
        return args
    overlay.setdefault("output", {})["checkpoint"] = str(checkpoint)
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(overlay))
    return ["--config", str(path)]


@pytest.mark.parametrize("source", ["flag", "yaml", "set"])
def test_checkpoint_sources_restore_saved_baseline_before_overrides(
    saved_run, dispatch, tmp_path, source
):
    original, _, checkpoint = saved_run
    arguments = checkpoint_args(
        source, checkpoint, tmp_path, {"training": {"iterations": 2, "seed": 10}}
    )
    result = cli.main(
        ["train", *arguments, "--epochs", "3", "--set", "training.seed=11"]
    )
    assert dispatch == [("train", result, str(checkpoint))]
    assert result.training.iterations == 3 and result.training.seed == 11
    assert result.representation.name == "mpnn"
    assert result.representation.trainable
    assert result.representation.output == original.representation.output == 8
    assert result.representation.max_atoms == original.representation.max_atoms
    assert result.selection.mode == "actor_value" and result.selection.candidate_value
    assert result.training.hidden == 16
    assert result.scoring.objective == "logp8"
    assert result.output.logger == "local"


@pytest.mark.parametrize("source", ["flag", "yaml", "set"])
@pytest.mark.parametrize("command", ["evaluate", "sample", "profile"])
def test_checkpoint_sources_reach_correct_dispatch(
    saved_run, dispatch, tmp_path, source, command
):
    _, _, checkpoint = saved_run
    result = cli.main([command, *checkpoint_args(source, checkpoint, tmp_path)])
    expected = "profile" if command == "profile" else "evaluation"
    assert dispatch == [(expected, result, str(checkpoint))]


@pytest.mark.parametrize("source", ["flag", "yaml", "set"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("representation.output", "16"),
        ("training.hidden", "32"),
        ("environment.max_attach", "8"),
        ("selection.context", "classic"),
    ],
)
def test_restore_rejects_model_changes_for_every_checkpoint_entry(
    saved_run, dispatch, tmp_path, source, field, value
):
    _, _, checkpoint = saved_run
    with pytest.raises(ValueError, match="not allowed"):
        cli.main(
            [
                "train",
                *checkpoint_args(source, checkpoint, tmp_path),
                "--set",
                f"{field}={value}",
            ]
        )
    assert not dispatch


def test_restore_yaml_model_change_is_rejected_before_dispatch(
    saved_run, dispatch, tmp_path
):
    _, _, checkpoint = saved_run
    arguments = checkpoint_args(
        "yaml", checkpoint, tmp_path, {"representation": {"output": 16}}
    )
    with pytest.raises(ValueError, match="not allowed"):
        cli.main(["train", *arguments])
    assert not dispatch


def test_saved_conditional_defaults_are_used_before_validation(
    saved_run, dispatch, tmp_path
):
    _, _, checkpoint = saved_run
    # Against new defaults this override would fail: candidate_value=False.
    # Against the saved checkpoint it is valid and preserves its head layout.
    result = cli.main(
        [
            "train",
            "--set",
            f"output.checkpoint={checkpoint}",
            "--set",
            "selection.mode=actor_value",
        ]
    )
    assert result.selection.candidate_value and result.selection.mode == "actor_value"
    assert dispatch


def test_no_manifest_checkpoint_cannot_implicitly_change_model(dispatch, tmp_path):
    checkpoint = tmp_path / "old_checkpoint"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="legacy loader"):
        cli.main(["train", "--checkpoint", str(checkpoint)])
    assert not dispatch


def test_saved_manifest_integrity_checked_before_dispatch(saved_run, dispatch):
    _, directory, checkpoint = saved_run
    (directory / "fragment_catalog.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        cli.main(["train", "--set", f"output.checkpoint={checkpoint}"])
    assert not dispatch


def test_local_logger_default_and_only_explicit_cli_values(dispatch):
    result = cli.main(["train"])
    assert result.output.logger == "local"
    assert cli.explicit_values(cli.parser().parse_args(["train"])) == {}
    assert dispatch == [("train", result, None)]


def test_yaml_cli_and_set_precedence_is_preserved_for_logger_and_settings(
    dispatch, tmp_path
):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "output:\n  logger: wandb\ntraining:\n  seed: 4\n  num_workers: 2\n"
    )
    from_yaml = cli.main(["train", "--config", str(path)])
    assert from_yaml.output.logger == "wandb" and from_yaml.training.seed == 4
    result = cli.main(
        [
            "train",
            "--config",
            str(path),
            "--logger",
            "tensorboard",
            "--seed",
            "5",
            "--set",
            "output.logger=local",
            "--set",
            "training.seed=6",
        ]
    )
    assert result.output.logger == "local" and result.training.seed == 6
    assert result.training.num_workers == 2


def test_default_training_entry_has_no_required_command(dispatch):
    result = cli.main(["--epochs", "2"], default_command="train")
    assert result.training.iterations == 2
    assert dispatch == [("train", result, None)]


def test_saved_config_partial_yaml_overlay_uses_saved_baseline(saved_run, tmp_path):
    _, _, checkpoint = saved_run
    path = tmp_path / "partial.yaml"
    path.write_text("evaluation:\n  episodes: 2\ntraining:\n  num_gpus: 0\n")
    config, _ = load_saved_config(checkpoint, config_path=path)
    assert config.evaluation.episodes == 2 and config.representation.name == "mpnn"
    assert config.selection.candidate_value


def test_direct_python_runner_config_cannot_bypass_restore_shape_checks(saved_run):
    _, _, checkpoint = saved_run
    config, _ = load_saved_config(checkpoint)
    config.output.checkpoint = str(checkpoint)
    prepare_restored_config(config)
    config.representation.output = 16
    with pytest.raises(ValueError, match="not allowed"):
        prepare_restored_config(config)


@pytest.mark.parametrize("command", ["train", "evaluate", "sample", "profile"])
def test_sweep_argument_is_not_silently_ignored(dispatch, command, tmp_path):
    with pytest.raises(ValueError, match="only supported"):
        cli.main([command, "--sweep", str(tmp_path / "sweep.yaml")])
    assert not dispatch


def test_sweep_execution_must_be_explicit_and_serial(dispatch, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli,
        "run_sweep",
        lambda config, path, dry_run=False: calls.append((config, path, dry_run)),
    )
    path = tmp_path / "sweep.yaml"
    cli.main(["sweep", "--sweep", str(path), "--dry-run"])
    assert len(calls) == 1 and calls[0][1:] == (Path(path), True)
    assert not dispatch


@pytest.mark.parametrize("memory_mib", [None, 1, 16384])
def test_ray_initialization_omits_auto_memory_or_converts_explicit_mib(monkeypatch, memory_mib):
    import ray
    import torch

    calls = []
    monkeypatch.setattr(ray, "init", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    config = load_config(overrides=["training.num_gpus=0"])
    config.training.object_store_memory_mib = memory_mib
    cli._init_ray(config)
    assert len(calls) == 1
    kwargs = calls[0]
    if memory_mib is None:
        assert "object_store_memory" not in kwargs
    else:
        assert kwargs["object_store_memory"] == memory_mib * 2**20
        assert type(kwargs["object_store_memory"]) is int
    assert kwargs["num_cpus"] == 10 and kwargs["num_gpus"] == 0
    assert kwargs["runtime_env"]["worker_process_setup_hook"] is cli._worker_setup


def test_saved_checkpoint_allows_object_store_resource_override(saved_run, dispatch):
    _, _, checkpoint = saved_run
    config = cli.main([
        "train", "--checkpoint", str(checkpoint),
        "--set", "training.object_store_memory_mib=16384",
    ])
    assert config.training.object_store_memory_mib == 16384
    prepare_restored_config(config)
    assert config.training.object_store_memory_mib == 16384
    assert dispatch == [("train", config, str(checkpoint))]
