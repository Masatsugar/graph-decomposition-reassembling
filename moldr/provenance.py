"""Local experiment manifests and immutable input asset snapshots."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
from functools import partial
from pathlib import Path
from types import CodeType, ModuleType

import yaml

from moldr.experiment import config_diff, load_config


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _serialize_objective(objective):
    """Freeze user evaluator definitions while retaining library references.

    Importable user functions otherwise pickle by reference, so changing or
    deleting their source would change a restored score despite a valid asset
    hash. Standard/third-party libraries remain references; the manifest lists
    those modules alongside the experiment runtime versions. Serializing their
    entire modules would be large and unreliable.
    Temporary cloudpickle registrations always restore the caller's state.
    """
    from ray import cloudpickle

    library_roots = {
        Path(sysconfig.get_path(key)).resolve()
        for key in ("stdlib", "purelib", "platlib")
    }
    by_value, library_modules = {}, set()
    pending = [objective.smiles_function, objective.mol_function]
    visited = set()
    while pending:
        function = pending.pop()
        if id(function) in visited:
            continue
        visited.add(id(function))
        while isinstance(function, partial):
            function = function.func
        module = function if isinstance(function, ModuleType) else inspect.getmodule(function)
        if module is None:
            continue
        origin = getattr(module, "__file__", None)
        origins = [origin] if origin else list(getattr(module, "__path__", ()))
        is_library = module.__name__.split(".", 1)[0] in sys.stdlib_module_names or any(
            Path(path).resolve().is_relative_to(root)
            for path in origins for root in library_roots
        )
        if module.__name__ == "moldr.scoring" or is_library:
            # In particular, keep Objective's defining class importable, even
            # when a user registers a helper from the scoring infrastructure.
            library_modules.add(module.__name__)
            continue
        if module.__name__ != "__main__":
            by_value[module.__name__] = module
        # Follow only the evaluator's named user functions/module references,
        # including same-file helper functions and explicit helper modules.
        # This deliberately does not collect an entire package dependency tree.
        if inspect.isfunction(function):
            codes, names = [function.__code__], set()
            while codes:
                code = codes.pop()
                names.update(code.co_names)
                codes.extend(value for value in code.co_consts if isinstance(value, CodeType))
            dependencies = [function.__globals__.get(name) for name in names]
            dependencies.extend(function.__defaults__ or ())
            dependencies.extend((function.__kwdefaults__ or {}).values())
            dependencies.extend(cell.cell_contents for cell in (function.__closure__ or ()))
            pending.extend(value for value in dependencies if inspect.isfunction(value) or isinstance(value, ModuleType))
    previously_registered = cloudpickle.list_registry_pickle_by_value()
    added = []
    try:
        for name, module in by_value.items():
            if name not in previously_registered:
                cloudpickle.register_pickle_by_value(module)
                added.append(module)
        payload = cloudpickle.dumps(objective)
    finally:
        for module in reversed(added):
            cloudpickle.unregister_pickle_by_value(module)
    return payload, sorted(by_value), sorted(library_modules)


def save_run(directory, config, *, blocks, catalog):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "resolved_config.yaml").exists():
        raise ValueError(f"Refusing to overwrite an existing experiment: {directory}")
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    assets = []

    def snapshot(source, destination):
        source, destination = Path(source).resolve(), directory / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assets.append({"source": str(source), "snapshot": str(destination.relative_to(directory)), "sha256": sha256_file(destination)})

    if config.representation.name == "mol2vec" or config.fragments.source == "vae":
        snapshot(config.representation.model_path, "assets/mol2vec/" + Path(config.representation.model_path).name)
        model = Path(config.representation.model_path)
        for sidecar in sorted(model.parent.glob(model.name + ".*")):
            if sidecar.is_file():
                snapshot(sidecar, "assets/mol2vec/" + sidecar.name)
    if config.fragments.source == "fixed" and not config.fragments.smiles:
        snapshot(config.fragments.path, "assets/fragments/" + Path(config.fragments.path).name)
    if config.fragments.source == "vae":
        for source in sorted(Path(config.fragments.proposer_dir).rglob("*")):
            if source.is_file():
                snapshot(source, Path("assets/proposer") / source.relative_to(config.fragments.proposer_dir))
    from moldr.scoring import BUILTIN_OBJECTIVES, get_objective

    objective = get_objective(config.scoring.objective)
    if BUILTIN_OBJECTIVES.get(objective.name) is not objective:
        path = directory / "assets/objective.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload, modules, libraries = _serialize_objective(objective)
        path.write_bytes(payload)
        assets.append({"source": f"registered-objective:{objective.name}", "snapshot": "assets/objective.pkl", "sha256": sha256_file(path),
                       "modules_by_value": modules, "library_modules": libraries})
    write_json(directory / "fragment_catalog.json", {"blocks": blocks, "catalog": catalog})
    versions = {}
    for name in ("torch", "ray", "rdkit", "numpy", "scipy", "pyarrow", "guacamol", "gensim", "PyYAML", "matplotlib"):
        versions[name] = importlib.metadata.version(name)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout
    source_hashes = {str(path): sha256_file(path) for base in ("moldr", "scripts") for path in sorted(Path(base).rglob("*.py"))}
    metadata = {"revision": revision, "working_tree_status": status, "source_sha256": source_hashes,
                "versions": versions, "python": platform.python_version(), "platform": platform.platform(),
                "seed": config.training.seed, "assets": assets,
                "catalog_sha256": sha256_file(directory / "fragment_catalog.json"),
                "resolved_config_sha256": sha256_file(directory / "resolved_config.yaml")}
    write_json(directory / "manifest.json", metadata)
    return directory


_MODEL_FIELDS = ["environment.mode", "environment.max_attach", "fragments.source", "fragments.pool_size",
                 "representation.name", "representation.trainable", "representation.obs_dim", "representation.layers",
                 "representation.hidden", "representation.output", "representation.max_atoms", "representation.max_bonds",
                 "selection.mode", "selection.context", "selection.candidate_value", "training.hidden"]


def find_run(checkpoint):
    path = Path(checkpoint).resolve()
    for parent in (path, *path.parents):
        if (parent / "resolved_config.yaml").is_file() and (parent / "manifest.json").is_file():
            return parent
    raise ValueError("New checkpoints require resolved_config.yaml and manifest.json; use the legacy loader for old checkpoints")


def load_saved_config(checkpoint, overrides=(), cli_values=None, *, config_path=None):
    directory = find_run(checkpoint)
    manifest = json.loads((directory / "manifest.json").read_text())
    for filename, key in (("fragment_catalog.json", "catalog_sha256"), ("resolved_config.yaml", "resolved_config_sha256")):
        if sha256_file(directory / filename) != manifest[key]:
            raise ValueError(f"Saved experiment data changed: {filename}")
    for asset in manifest["assets"]:
        snapshot = directory / asset["snapshot"]
        if not snapshot.is_file() or sha256_file(snapshot) != asset["sha256"]:
            raise ValueError(f"Saved asset missing or changed: {snapshot}")
    for asset in manifest["assets"]:
        if asset["snapshot"] == "assets/objective.pkl":
            from ray import cloudpickle

            from moldr.scoring import Objective, register_objective

            objective = cloudpickle.loads((directory / asset["snapshot"]).read_bytes())
            if not isinstance(objective, Objective):
                raise ValueError("Saved objective metadata is invalid")
            register_objective(objective.name, objective.smiles_function, objective.mol_function, objective.deterministic)
    saved = load_config(directory / "resolved_config.yaml")
    updated = load_config(config_path, overrides=overrides, cli_values=cli_values, base=saved)
    changes = config_diff(saved, updated)
    for field in _MODEL_FIELDS:
        if field in changes:
            raise ValueError(f"Checkpoint model/environment change is not allowed: {field}")
    if any(field in changes for field in ("fragments.path", "fragments.smiles", "fragments.proposer_dir", "representation.model_path")):
        raise ValueError("Checkpoint vocabulary and model assets cannot be overridden")
    # Always use the saved assets; a changed original file must not affect a
    # restored policy. Paths remain valid when the whole run is moved.
    for asset in manifest["assets"]:
        snapshot = str(directory / asset["snapshot"])
        if asset["snapshot"].startswith("assets/mol2vec/") and Path(asset["source"]).name == Path(saved.representation.model_path).name:
            updated.representation.model_path = snapshot
        elif asset["snapshot"].startswith("assets/fragments/"):
            updated.fragments.path = snapshot
    if updated.fragments.source == "vae":
        updated.fragments.proposer_dir = str(directory / "assets/proposer")
    return updated, directory


def prepare_restored_config(config):
    """Apply the same restoration checks to direct Python runner calls."""
    if not config.output.checkpoint:
        return config
    saved, directory = load_saved_config(config.output.checkpoint)
    original = load_config(directory / "resolved_config.yaml")
    changes = config_diff(original, config)
    for field in _MODEL_FIELDS:
        if field in changes:
            raise ValueError(f"Checkpoint model/environment change is not allowed: {field}")
    for section, key in (("fragments", "path"), ("fragments", "smiles"), ("fragments", "proposer_dir"), ("representation", "model_path")):
        requested = getattr(getattr(config, section), key)
        if requested not in (getattr(getattr(original, section), key), getattr(getattr(saved, section), key)):
            raise ValueError("Checkpoint vocabulary and model assets cannot be overridden")
        setattr(getattr(config, section), key, getattr(getattr(saved, section), key))
    return config


def checkpoint_path(path):
    path = Path(path).resolve()
    if (path / "last_checkpoint.json").is_file():
        return str(path / json.loads((path / "last_checkpoint.json").read_text())["path"])
    return str(path)
