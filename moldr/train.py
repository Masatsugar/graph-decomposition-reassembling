import argparse
import numbers
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import ray
from guacamol.standard_benchmarks import (
    hard_osimertinib,
    logP_benchmark,
    qed_benchmark,
    ranolazine_mpo,
)
from ray.air.integrations.wandb import setup_wandb
from ray.tune.utils import flatten_dict
from rdkit import RDLogger
from tqdm import tqdm

from moldr.config import get_default_config_v1
from moldr.objective import qed_sa, rediscovery
from moldr.utils import custom_log_creator, save


def _silence_rdkit():
    RDLogger.DisableLog("rdApp.warning")


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


def main(argv=None):
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

    parser.add_argument("--objective", choices=[sc.name for sc in sc_list])
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--length", type=int, default=100)

    print("[INFO] Base path:", os.getcwd())
    # DEFAULT MODEL PATH
    model_path = "models/model_300dim.pkl"
    base_dir = Path(f"{os.getcwd()}/data/building_blocks")
    print("[INFO] Base dir:", base_dir)

    args = parser.parse_args(argv)

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
        filename = (
            f"guacamol_s{minsup}_l2.csv"
            if lower_name == "guacamol"
            else f"s{minsup}_l2.csv"
        )
        smi_file = base_path / filename

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

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={
            "worker_process_setup_hook": _silence_rdkit,
            "env_vars": {"PYTHONWARNINGS": "ignore::DeprecationWarning"},
        },
    )
    try:
        objectives = [
            sc for sc in sc_list if args.objective is None or sc.name == args.objective
        ]
        for objective in objectives:
            target = objective.name
            print("[INFO] TARGET:", target)
            run_time = datetime.now(ZoneInfo("Asia/Tokyo")).strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            # Get configuration
            config = get_default_config_v1(
                objective,
                building_blocks_smiles,
                model_path=model_path,
                num_workers=args.num_workers,
                num_gpus=args.num_gpus,
                length=args.length,
            ).training(
                train_batch_size=args.train_batch_size,
                minibatch_size=args.minibatch_size,
                num_epochs=args.num_epochs,
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
            try:
                algo_name = algo.__class__.__name__

                # Save env_config separately as it contains the actual environment parameters
                env_config = config.env_config
                save(
                    f"outputs/{algo_name}/models/{target}/{run_time}/env_config.pkl",
                    env_config,
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
                        print(f"Checkpoint saved at: {checkpoint_dir.checkpoint.path}")

                # Final save
                final_checkpoint = algo.save(
                    f"outputs/{algo_name}/models/{target}/{run_time}/"
                )
                print(f"Final checkpoint saved at: {final_checkpoint.checkpoint.path}")
            finally:
                algo.stop()
                if wb_run is not None:
                    wb_run.finish()

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
