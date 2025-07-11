import argparse
import os
import numbers

from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm

import ray
from rdkit import RDLogger

CODE_ROOT = Path(__file__).resolve().parent

def _silence_rdkit():
    RDLogger.DisableLog('rdApp.warning')


# Initialize Ray
ray.init(
    ignore_reinit_error=True,
    runtime_env={
        "worker_process_setup_hook": _silence_rdkit,
        "env_vars": {"PYTHONWARNINGS": "ignore::DeprecationWarning"},
        "excludes": ["**/.git/**", "**/__pycache__/**", "**/.pytest_cache/**"],
    },
)

from ray.rllib.algorithms.ppo import PPO
from ray.air.integrations.wandb import setup_wandb
from ray.tune.utils import flatten_dict
from guacamol.standard_benchmarks import (
    hard_osimertinib,
    logP_benchmark,
    qed_benchmark,
    ranolazine_mpo,
)

from moldr.config import get_default_config_v1
from moldr.utils import custom_log_creator, save

from moldr.objective import rediscovery, qed_sa

sc_list = [
    logP_benchmark(8.0),
    logP_benchmark(-1.0),
    qed_benchmark(),
    qed_sa(),
    rediscovery("Celecoxib rediscovery"),
    rediscovery("Troglitazone rediscovery"),
    rediscovery("Aripiprazole similarity"),
    hard_osimertinib(),
    ranolazine_mpo(),
]


def clean_for_wandb(d: dict, sep: str = "/"):
    flat = flatten_dict(d, delimiter=sep)
    return {k: v for k, v in flat.items() if isinstance(v, numbers.Number)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test arguments.")
    parser.add_argument(
        "--dataset",
        default="GuacaMol",
        type=str,
        help="GuacaMol or ZINC",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="The maximum number of training iterations",
    )
    parser.add_argument(
        "--minsup",
        default=10000,
        type=int,
        help="The maximum number of support for subgraph mining",
    )
    parser.add_argument(
        "--jt",
        action="store_true",
        help="Use junction tree as building blocks",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="the number of workers",
    )
    parser.add_argument(
        "--num_gpus",
        default=1,
        type=int,
        help="the number of gpus",
    )
    parser.add_argument(
        "--logger",
        default="wandb",
        type=str,
        help="logging method, e.g., wandb or tensorboard",
    )
    parser.add_argument(
        "--project_name",
        default="moldr",
        type=str,
        help="Project name for logging",
    )

    print("[INFO] Base path:", os.getcwd())
    # DEFAULT MODEL PATH
    model_path = "models/model_300dim.pkl"
    base_dir = Path(f"{os.getcwd()}/data/building_blocks")
    print("[INFO] Base dir:", base_dir)

    args = parser.parse_args()

    # DATASET PATH
    lower_name = args.dataset.lower()
    base_path = base_dir / lower_name
    minsup = args.minsup

    mining_method = ""
    smi_file = base_dir / "test"

    if lower_name == "zinc" and args.jt:
        smi_file = base_path / "gspan_jt.datavocab.csv"
    elif lower_name in {"zinc", "guacamol"}:
        mining_method = f"gSpan_s{minsup}"
        smi_file = base_path / f"{lower_name}_s{minsup}_l2.csv"

    print("[INFO] SMILES path:", smi_file)
    if smi_file.exists():
        print("Load:", smi_file)
        building_blocks_smiles = pd.read_csv(smi_file, index_col=0).values.flatten()
    else:
        raise ValueError(
            "Firstly decompose molecules into building blocks or select the path of building block."
        )

    building_blocks_smiles = [smi for smi in building_blocks_smiles if len(smi) > 1]
    print(
        f"[INFO] Dataset: {args.dataset}, minsup: {args.minsup}, nums: {len(building_blocks_smiles)}"
    )

    for objective in sc_list:
        target = objective.name
        print("[INFO] TARGET:", target)
        run_time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

        # Get configuration
        config = get_default_config_v1(
            objective,
            building_blocks_smiles,
            model_path=model_path,
            num_workers=args.num_workers,
            num_gpus=args.num_gpus,
        )

        # Add custom logger
        wb_run = None
        if args.logger == "wandb":
            wb_run = setup_wandb(
                project=args.project_name,
                group=f"{target}_{run_time}",
                name=f"{target}_{run_time}",
                config={"task": target},
            )
        else:
            config.debugging(
                log_level="INFO",
                logger_creator=custom_log_creator(
                    f"ray_results/PPO_MolEnvValueMax_{args.dataset}_{mining_method}",
                    target,
                ),
            )

        # Create PPO algorithm instance
        algo = config.build()
        algo_name = algo.__class__.__name__

        # Save env_config separately as it contains the actual environment parameters
        env_config = config.env_config
        save(
            f"outputs/{algo_name}/models/{target}/{run_time}/env_config.pkl", env_config
        )

        # Training loop
        for i in tqdm(range(1, args.epochs + 1)):
            result = algo.train()
            if args.logger == "wandb":
                flatten_result = clean_for_wandb(result)
                wb_run.log(flatten_result, step=i)

            if i % 20 == 0:
                checkpoint_dir = algo.save(
                    f"outputs/{algo_name}/models/{target}/{run_time}"
                )
                print(f"Checkpoint saved at: {checkpoint_dir}")

        # Final save
        final_checkpoint = algo.save(f"outputs/{algo_name}/models/{target}/{run_time}/")
        print(f"Final checkpoint saved at: {final_checkpoint}")
        algo.stop()
        wb_run.finish()

    ray.shutdown()
