import argparse
import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID

from guacamol.standard_benchmarks import (
    hard_osimertinib,
    logP_benchmark,
    qed_benchmark,
    ranolazine_mpo,
)

from moldr.chemutils import get_mol
from moldr.config import get_default_config
from moldr.env import MolEnvValueMax
from moldr.objective import qed_sa, rediscovery
from moldr.utils import custom_log_creator, load, save

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

    print(os.getcwd())
    model_path = "models/model_300dim.pkl"
    base_dir = Path("outputs/building_blocks")
    args = parser.parse_args()

    if args.dataset == "ZINC":
        if args.jt:
            smi_file = base_dir / f"{args.dataset}/gspan_jt.datavocab.csv"
        else:
            smi_file = base_dir / f"{args.dataset}/zinc_s{args.minsup}_l2.csv"
    elif args.dataset == "GuacaMol":
        mining_method = f"gSpan_s{args.minsup}"
        smi_file = base_dir / f"{args.dataset}/guacamol_s{args.minsup}_l2.csv"
    else:
        smi_file = base_dir / "test"
        mining_method = ""

    if smi_file.exists():
        print("Load:", smi_file)
        building_blocks_smiles = pd.read_csv(smi_file, index_col=0).values.flatten()
    else:
        raise ValueError("Firstly decompose molecules into building blocks.")

    building_blocks_smiles = [smi for smi in building_blocks_smiles if len(smi) > 1]
    print(
        f"Dataset: {args.dataset}, minsup: {args.minsup}, nums: {len(building_blocks_smiles)}"
    )

    for objective in sc_list:
        target = objective.name
        print("TARGET:", target)
        time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

        # Initialize Ray
        ray.init(ignore_reinit_error=True)

        # Get configuration
        config = get_default_config(
            MolEnvValueMax,
            objective,
            building_blocks_smiles,
            model_path=model_path,
            num_workers=args.num_workers,
            num_gpus=args.num_gpus,
        )

        # Add custom logger
        config.debugging(
            log_level="INFO",
            logger_creator=custom_log_creator(
                f"ray_results/PPO_MolEnvValueMax_{args.dataset}_{mining_method}", target
            ),
        )

        # Save env_config separately as it contains the actual environment parameters
        env_config = config.env_config
        save(f"outputs/PPO/models/{target}/{time}/env_config.pkl", env_config)

        # Create PPO algorithm instance
        algo = config.build()

        # Training loop
        for i in tqdm(range(1, args.epochs + 1)):
            result = algo.train()

            if i % 20 == 0:
                checkpoint_dir = algo.save(f"outputs/PPO/models/{target}/{time}")
                print(f"Checkpoint saved at: {checkpoint_dir}")

        # Final save
        final_checkpoint = algo.save(f"outputs/PPO/models/{target}/{time}/")
        print(f"Final checkpoint saved at: {final_checkpoint}")

        algo.stop()
        ray.shutdown()
