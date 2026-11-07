"""Run a small E2E demo, train a CPU policy, or sample a saved policy.

Examples (from the repository root)::

    uv run python examples/tutorial.py demo
    uv run python examples/tutorial.py train --iterations 1
    uv run python examples/tutorial.py sample --checkpoint PATH_PRINTED_BY_TRAIN

See docs/tutorial.md for score design, reward shaping, and saved artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The 13 fragments mined from eight small molecules in quickstart.ipynb.
# Replace these or pass --fragments your_blocks.smi / your_blocks.csv.
TOY_FRAGMENTS = [
    "C=O", "CC", "CC(=O)O", "CC=O", "CCC", "CCCO", "CCN",
    "CCO", "CCOC", "CN", "CO", "COC", "O=CO",
]
PREVIEW_SMILES = ["C", "CCO", "CC(=O)O", "c1ccccc1", "CCOC(=O)C", "CCCCCCCC"]


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest="command", required=True)
    for command in ("demo", "train"):
        sub = commands.add_parser(command, help="Random valid actions" if command == "demo" else "Train CPU PPO")
        sub.add_argument("--objective", choices=["custom", "QED"], default="custom")
        sub.add_argument("--target-atoms", type=float, default=8.0, help="Custom score's preferred heavy-atom count")
        sub.add_argument("--size-width", type=float, default=4.0, help="Custom score's size tolerance (> 0)")
        sub.add_argument("--qed-weight", type=float, default=0.5, help="Custom score's QED weight (0..1)")
        sub.add_argument("--reward", choices=["improvement", "sparse"], default="improvement")
        sub.add_argument("--final-weight", type=float, default=1.0, help="Multiplier for true-termination score bonus")
        sub.add_argument("--fragments", type=Path, help="One SMILES per line, or CSV with index then SMILES")
        sub.add_argument("--base-smiles", default="C")
        sub.add_argument("--atom-limit", type=int, default=12)
        sub.add_argument("--max-steps", type=int, default=8, help="Maximum decisions, including both E2E phases")
        sub.add_argument("--seed", type=int, default=7)
        sub.add_argument("--output", type=Path, default=ROOT / "outputs/tutorial")
        if command == "train":
            sub.add_argument("--iterations", type=int, default=1, help="PPO iterations; one is a connection check")
    sample = commands.add_parser("sample", help="Restore saved configuration, objective, and policy")
    sample.add_argument("--checkpoint", type=Path, required=True, help="Run directory printed by train")
    sample.add_argument("--episodes", type=int, default=3)
    sample.add_argument("--seed", type=int, help="Default: use the training run's saved seed")
    sample.add_argument("--output", type=Path, help="Default: use the saved output parent")
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_config(args):
    from moldr.experiment import load_config
    from moldr.factory import resolve_config

    objective_name = "QED"
    if args.objective == "custom":
        # This import and registration are intentionally absent from sampling.
        from tutorial_objective import register_tutorial_objective

        objective_name = register_tutorial_objective(
            args.target_atoms, args.size_width, args.qed_weight
        ).name
    config = load_config(cli_values={
        "environment.mode": "e2e", "environment.base_smiles": args.base_smiles,
        "environment.atom_limit": args.atom_limit,
        "environment.max_episode_steps": args.max_steps,
        "environment.max_attach": 32, "environment.score_threshold": 1.1,
        "environment.reward": args.reward, "environment.final_weight": args.final_weight,
        "fragments.smiles": [] if args.fragments else TOY_FRAGMENTS,
        "fragments.path": str(args.fragments) if args.fragments else "unused-inline-fragments",
        "representation.model_path": str(ROOT / "models/model_300dim.pkl"),
        "scoring.objective": objective_name,
        "training.seed": args.seed, "training.iterations": getattr(args, "iterations", 1),
        "training.num_workers": 0, "training.num_cpus": 2, "training.num_gpus": 0.0,
        "training.train_batch_size": 64, "training.minibatch_size": 32,
        "training.num_epochs": 1, "training.hidden": 32,
        "output.directory": str(args.output), "output.logger": "local",
    })
    return resolve_config(config)


def preview_scores(config):
    from rdkit import Chem

    from moldr.scoring import get_objective

    objective = get_objective(config.scoring.objective)
    rows = []
    print(f"Objective: {objective.name} (higher is better)")
    for smiles in PREVIEW_SMILES:
        mol = Chem.MolFromSmiles(smiles)
        score = float(objective.smiles_function(smiles))
        repeat = float(objective.smiles_function(smiles))
        mol_score = float(objective.mol_function(mol))
        if not math.isfinite(score) or score != repeat or not math.isclose(score, mol_score):
            raise ValueError(f"Objective preflight failed for {smiles}: {score}, {repeat}, {mol_score}")
        rows.append({"smiles": smiles, "heavy_atoms": mol.GetNumHeavyAtoms(), "score": score})
        print(f"  {smiles:16s} atoms={mol.GetNumHeavyAtoms():2d} score={score:.6f}")
    return rows


def run_demo(config, blocks, catalog):
    import numpy as np
    from rdkit import rdBase

    from moldr.factory import make_environment
    from moldr.provenance import save_run, write_json

    stamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d_%H-%M-%S-%f")
    directory = save_run(Path(config.output.directory) / f"demo_{stamp}_seed{config.training.seed}",
                         config, blocks=blocks, catalog=catalog)
    env = make_environment(config)
    rng = np.random.default_rng(config.training.seed)
    trajectory = []
    try:
        obs, _ = env.reset(seed=config.training.seed)
        for step in range(1, config.environment.max_episode_steps + 1):
            valid_actions = np.flatnonzero(obs["action_mask"] > 0)
            if not valid_actions.size:
                raise RuntimeError("No live action before episode completion")
            action = int(rng.choice(valid_actions))
            with rdBase.BlockLogs():
                obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append({
                "step": step, "phase": info["phase"], "action": action,
                "smiles": info["final_smiles"], "score": float(env.prev_score),
                "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
            })
            print(f"{step:2d} {info['phase']:13s} score={env.prev_score:.6f} "
                  f"reward={reward:+.6f} {info['final_smiles']}")
            if terminated or truncated:
                break
        if not (terminated or truncated):
            raise RuntimeError("Episode did not finish within the decision limit")
        molecule = {
            "policy": "uniform_valid_actions", "seed": config.training.seed,
            "smiles": env.base_smiles, "score": float(env.prev_score),
            "reward": sum(row["reward"] for row in trajectory), "steps": len(trajectory),
            "terminated": bool(terminated), "truncated": bool(truncated),
        }
        write_csv(directory / "trajectory.csv", trajectory)
        write_json(directory / "molecules.json", [molecule])
        print(f"Episode complete: steps={len(trajectory)}, terminated={terminated}, truncated={truncated}")
    finally:
        env.close()
    return directory


def main(argv=None):
    args = parser().parse_args(argv)
    # Resolve user paths before switching to the repository for asset/source metadata.
    for field in ("output", "fragments", "checkpoint"):
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    os.chdir(ROOT)
    if args.command == "sample":
        from moldr.cli import main as moldr_cli

        command = ["sample", "--checkpoint", str(args.checkpoint), "--episodes", str(args.episodes)]
        if args.seed is not None:
            command += ["--seed", str(args.seed)]
        if args.output is not None:
            command += ["--output", str(args.output)]
        directory = moldr_cli(command)
        rows = json.loads((directory / "molecules.json").read_text())
        columns = ("seed", "smiles", "score", "reward", "steps", "terminated", "truncated")
        write_csv(directory / "molecules.csv", [{key: row[key] for key in columns} for row in rows])
        for row in rows:
            print(f"score={row['score']:.6f} {row['smiles']}")
    else:
        config, blocks, catalog = make_config(args)
        scores = preview_scores(config)
        if args.command == "demo":
            directory = run_demo(config, blocks, catalog)
        else:
            from moldr.cli import run_training

            directory = run_training(config)
        write_csv(directory / "objective_preview.csv", scores)
        settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        (directory / "tutorial_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"\nResults: {directory}")
    if args.command == "train":
        import shlex

        print("Next:", shlex.join([sys.executable, str(Path(__file__).resolve()),
                                  "sample", "--checkpoint", str(directory), "--episodes", "3"]))
    return directory


if __name__ == "__main__":
    main()
