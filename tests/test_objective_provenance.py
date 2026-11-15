"""Objective snapshots keep evaluator code and purity across fresh processes."""

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem

from moldr.experiment import ExperimentConfig
from moldr.provenance import (
    _serialize_objective,
    load_saved_config,
    save_run,
    sha256_file,
)
from moldr.scoring import (
    BUILTIN_OBJECTIVES,
    OBJECTIVE_REGISTRY,
    Objective,
    ScoringService,
    get_objective,
    register_objective,
)


def config_for(name):
    config = ExperimentConfig()
    config.fragments.smiles = ["CC", "CO"]
    config.representation.name = "mpnn"
    config.environment.atom_limit = 5
    config.scoring.objective = name
    return config.validate()


def register(monkeypatch, name, smiles_function, mol_function=None):
    # Restore both additions and replacements after each test.
    monkeypatch.setitem(OBJECTIVE_REGISTRY, name, OBJECTIVE_REGISTRY.get(name))
    return register_objective(name, smiles_function, mol_function)


def save(tmp_path, config):
    directory = save_run(
        tmp_path / "run", config, blocks=["CC", "CO"], catalog=["CC", "CO"]
    )
    checkpoint = directory / "checkpoints" / "iteration1"
    checkpoint.mkdir(parents=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    return checkpoint, directory, manifest


def fresh_python(code, *arguments, extra_pythonpath=None):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    if extra_pythonpath is not None:
        env["PYTHONPATH"] = (
            str(extra_pythonpath) + os.pathsep + env.get("PYTHONPATH", "")
        )
    result = subprocess.run(
        [sys.executable, "-c", code, *map(str, arguments)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


RESTORE_SCRIPT = """
import json, sys
from moldr.provenance import load_saved_config
from moldr.factory import make_env_config
from moldr.scoring import OBJECTIVE_REGISTRY, Objective, ScoringService, get_objective
from ray import cloudpickle

checkpoint, name = sys.argv[1:3]
before = name in OBJECTIVE_REGISTRY
config, _ = load_saved_config(checkpoint)
metadata = make_env_config(config)["SCORING"]
serialized = cloudpickle.dumps(metadata)
# Simulate the fresh rollout worker: it never called driver registration.
OBJECTIVE_REGISTRY.pop(name, None)
worker_metadata = cloudpickle.loads(serialized)
smiles = ScoringService(**dict(worker_metadata, input_mode="smiles"))
mol_metadata = dict(worker_metadata, input_mode="mol")
mol = ScoringService(**mol_metadata)
cache_metadata = dict(worker_metadata, cache_size=4)
cache = ScoringService(**cache_metadata)
scores = [smiles.score("CCO"), mol.score("CCO"), cache.score("CCO"), cache.score("CCO")]
print(json.dumps({"registered_before_restore": before, "metadata_type": isinstance(worker_metadata["objective"], Objective),
                  "registered_worker": smiles.is_registered, "deterministic": smiles.deterministic,
                  "scores": scores, "cache": cache.stats}))
"""


def test_closure_snapshot_restores_in_independent_process_and_fresh_worker(
    tmp_path, monkeypatch
):
    scale, bias = 0.37, 0.125
    name = "persist-closure"
    objective = register(
        monkeypatch,
        name,
        lambda smiles: bias + scale * Chem.MolFromSmiles(smiles).GetNumAtoms(),
        lambda mol: bias + scale * mol.GetNumAtoms(),
    )
    checkpoint, directory, manifest = save(tmp_path, config_for(name))
    asset = next(
        asset
        for asset in manifest["assets"]
        if asset["snapshot"] == "assets/objective.pkl"
    )
    assert asset["sha256"] == sha256_file(directory / asset["snapshot"])
    result = fresh_python(RESTORE_SCRIPT, checkpoint, name)
    assert not result["registered_before_restore"]
    assert (
        result["metadata_type"]
        and result["registered_worker"]
        and result["deterministic"]
    )
    assert result["scores"] == pytest.approx([1.235] * 4)
    assert result["cache"] == {"requests": 2, "computations": 1, "hits": 1}
    assert ScoringService(objective).score("CCO") == pytest.approx(1.235)


@pytest.mark.parametrize("change", ["change", "delete"])
def test_importable_user_module_and_helper_are_frozen(tmp_path, monkeypatch, change):
    from ray import cloudpickle

    module_name = "user_score_module"
    source = tmp_path / f"{module_name}.py"
    helper_source = tmp_path / "user_score_helper.py"
    helper_source.write_text(
        "OFFSET=0.125\ndef compute(mol):\n    return OFFSET + 0.37 * mol.GetNumAtoms()\n"
    )
    helper_spec = importlib.util.spec_from_file_location(
        "user_score_helper", helper_source
    )
    helper_module = importlib.util.module_from_spec(helper_spec)
    monkeypatch.setitem(sys.modules, "user_score_helper", helper_module)
    helper_spec.loader.exec_module(helper_module)
    source.write_text(
        "from rdkit import Chem\nimport user_score_helper\ndef helper(mol):\n    return user_score_helper.compute(mol)\ndef score(smiles):\n    return helper(Chem.MolFromSmiles(smiles))\ndef score_mol(mol):\n    return helper(mol)\n"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    name = "persist-module"
    register(monkeypatch, name, module.score, module.score_mol)
    before = cloudpickle.list_registry_pickle_by_value()
    checkpoint, _, manifest = save(tmp_path, config_for(name))
    assert cloudpickle.list_registry_pickle_by_value() == before
    asset = next(
        asset
        for asset in manifest["assets"]
        if asset["snapshot"] == "assets/objective.pkl"
    )
    assert module_name in asset["modules_by_value"]
    assert "user_score_helper" in asset["modules_by_value"]
    if change == "change":
        source.write_text(
            "def helper(mol):\n    return -999.0\ndef score(smiles):\n    return -999.0\ndef score_mol(mol):\n    return -999.0\n"
        )
        helper_source.write_text("def compute(mol):\n    return -999.0\n")
    else:
        source.unlink()
        helper_source.unlink()
    result = fresh_python(RESTORE_SCRIPT, checkpoint, name, extra_pythonpath=tmp_path)
    assert result["scores"] == pytest.approx([1.235] * 4)
    assert result["metadata_type"] and result["registered_worker"]


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("already_registered", [False, True])
def test_temporary_cloudpickle_registration_restores_prior_state(
    tmp_path, monkeypatch, failure, already_registered
):
    from ray import cloudpickle

    source = tmp_path / "temporary_objective_module.py"
    source.write_text("def score(smiles):\n    return 0.5\n")
    spec = importlib.util.spec_from_file_location("temporary_objective_module", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec.loader.exec_module(module)
    objective = Objective("temporary", module.score)
    if already_registered:
        cloudpickle.register_pickle_by_value(module)
    before = cloudpickle.list_registry_pickle_by_value()
    try:
        if failure:

            def failed_pickle(value):
                raise RuntimeError("pickle failed")

            monkeypatch.setattr(cloudpickle, "dumps", failed_pickle)
            with pytest.raises(RuntimeError, match="pickle failed"):
                _serialize_objective(objective)
        else:
            payload, _, _ = _serialize_objective(objective)
            assert cloudpickle.loads(payload).smiles_function("C") == 0.5
        assert cloudpickle.list_registry_pickle_by_value() == before
    finally:
        if already_registered:
            cloudpickle.unregister_pickle_by_value(module)


@pytest.mark.parametrize("tamper", ["change", "delete"])
def test_objective_hash_validation_precedes_deserialization(
    tmp_path, monkeypatch, tamper
):
    from ray import cloudpickle

    name = "persist-hash"
    original = register(monkeypatch, name, lambda smiles: 0.5)
    checkpoint, directory, _ = save(tmp_path, config_for(name))
    snapshot = directory / "assets/objective.pkl"
    if tamper == "change":
        snapshot.write_bytes(snapshot.read_bytes() + b"tampered")
    else:
        snapshot.unlink()

    def forbidden_deserialization(value):
        raise AssertionError("Objective was unpickled before hash verification")

    monkeypatch.setattr(cloudpickle, "loads", forbidden_deserialization)
    with pytest.raises(ValueError, match="missing or changed"):
        load_saved_config(checkpoint)
    assert get_objective(name) is original


@pytest.mark.parametrize("name", ["QED", "logp8"])
def test_builtin_identity_does_not_create_custom_objective_asset(tmp_path, name):
    _, directory, manifest = save(tmp_path, config_for(name))
    assert not (directory / "assets/objective.pkl").exists()
    assert all(
        asset["snapshot"] != "assets/objective.pkl" for asset in manifest["assets"]
    )
    assert get_objective(name) is BUILTIN_OBJECTIVES[get_objective(name).name]


def test_builtin_runtime_replacement_is_snapshotted_and_restored(tmp_path, monkeypatch):
    name = "qed"
    register(monkeypatch, name, lambda smiles: 0.625, lambda mol: 0.625)
    checkpoint, directory, _ = save(tmp_path, config_for("QED"))
    assert (directory / "assets/objective.pkl").is_file()
    result = fresh_python(RESTORE_SCRIPT, checkpoint, name)
    assert result["registered_before_restore"]  # the fresh process has original QED
    assert result["scores"] == [0.625] * 4
    assert result["metadata_type"]


def test_direct_library_function_keeps_small_reference_payload():
    objective = Objective("library", np.sum)
    payload, modules, libraries = _serialize_objective(objective)
    assert not modules and "numpy" in libraries
    assert len(payload) < 8192
