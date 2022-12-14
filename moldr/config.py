import gym
import numpy as np
import ray
from mi_collections.chemutils import get_mol
from ray.rllib.agents.ppo import PPOTrainer, ppo


def get_default_config(
    env,
    sc,
    building_blocks_smiles,
    model_path,
    base_smiles="C",
    num_workers=40,
    num_gpus=2,
    score_threshold=1.0,
    length=100,
):
    def check_valid_smiles(smiles):
        mol = get_mol(smiles)
        if mol is None:
            raise ValueError("INVALID SMILES.")
        else:
            return base_smiles

    base_smiles = check_valid_smiles(base_smiles)
    high = np.array([np.finfo(np.float32).max for i in range(300)])
    observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
    scoring_function = sc.wrapped_objective.score

    config = ppo.DEFAULT_CONFIG
    # config = apex.APEX_DEFAULT_CONFIG
    config.update(
        {
            "env": env,
            "ACTION_SPACE": gym.spaces.Discrete(len(building_blocks_smiles)),
            "OBS_SPACE": observation_space,
            "BUILDING_BLOCKS": building_blocks_smiles,
            "SCORE_FUNCTION": scoring_function,
            "SCORE_THRESHOLD": score_threshold,  # 12. for maximizing penalized logp for our paper
            "FINAL_WEIGHT": 1.0,  # Weighted reward for final step.
            "BASE_SMILES": base_smiles,  # Starting point of node
            "MODEL_PATH": model_path,  # MOL2VEC PATH
            "LENGTH": length,  # Max nodes of molecules
            "model": {
                "fcnet_hiddens": [256, 128, 128],  # [256, 128] # old version
                "fcnet_activation": "relu",
                "max_seq_len": 100,
            },
            "framework": "torch",
            # Set up a separate evaluation worker set for the
            # `trainer.evaluate()` call after training (see below).
            "num_workers": num_workers,
            "num_gpus": num_gpus,
            # "train_batch_size": 2000,
            # "sgd_minibatch_size": 20,
            #              "use_lstm": True,
            #             # Max seq len for training the LSTM, defaults to 20.
            #             "max_seq_len": 20,
            #             # Size of the LSTM cell.
            #             "lstm_cell_size": 256,
            #             # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            #             "lstm_use_prev_action": True,
            #             # Whether to feed r_{t-1} to LSTM.
            #             "lstm_use_prev_reward": False,
            #             # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
            #             "_time_major": False,
            # "callbacks": MolEnvCallbacks,
            "evaluation_num_workers": 1,
            # Only for evaluation runs, render the env.
            "evaluation_config": {
                "render_env": False,
            },
        }
    )
    return config
