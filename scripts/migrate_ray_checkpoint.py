"""Copy a trusted Ray checkpoint and remove obsolete logger configuration.

Run this with the original Python/Ray environment that can load the checkpoint.
Checkpoint pickle files execute Python code: use only artifacts you trust.
The source can be a complete run directory or a single checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def migrate_checkpoint(source: Path, destination: Path) -> dict:
    """Copy a checkpoint without changing its trained state or original files."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().absolute()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination = destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Destination must be outside the source directory")
    if not any(source.rglob("algorithm_state.pkl")):
        raise ValueError("Source contains no Ray algorithm_state.pkl checkpoint")

    # Delay this import so --help works without the original Ray environment.
    import cloudpickle

    shutil.copytree(source, destination)
    changes = []
    try:
        for path in sorted(destination.rglob("*.pkl")):
            if path.name not in {"algorithm_state.pkl", "policy_state.pkl"}:
                continue
            with path.open("rb") as stream:
                state = cloudpickle.load(stream)
            if path.name == "algorithm_state.pkl":
                config = state["config"]
            else:
                config = state["policy_spec"]["config"]
            if not isinstance(config, dict):
                raise TypeError(f"Expected a saved configuration dictionary: {path}")
            removed = []
            for key in ("logger_creator", "logger_config"):
                if key in config:
                    del config[key]
                    removed.append(key)
            if removed:
                with path.open("wb") as stream:
                    cloudpickle.dump(state, stream)
                changes.append(
                    {"file": str(path.relative_to(destination)), "removed": removed}
                )

        # Ray records absolute state-file paths in some checkpoint versions.
        for path in sorted(destination.rglob("rllib_checkpoint.json")):
            metadata = json.loads(path.read_text())
            previous = metadata.get("state_file")
            if previous:
                copied_state = path.parent / Path(previous).name
                if not copied_state.is_file():
                    raise ValueError(
                        f"Checkpoint state file is missing: {copied_state}"
                    )
                metadata["state_file"] = str(copied_state)
                path.write_text(json.dumps(metadata, indent=2) + "\n")
    except BaseException:
        # Only this newly created copy is removed; the source is never opened
        # for writing and a pre-existing destination is rejected above.
        shutil.rmtree(destination)
        raise
    return {"source": str(source), "destination": str(destination), "changes": changes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Trusted original run or checkpoint")
    parser.add_argument("destination", type=Path, help="New directory; must not exist")
    args = parser.parse_args(argv)
    print(json.dumps(migrate_checkpoint(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
