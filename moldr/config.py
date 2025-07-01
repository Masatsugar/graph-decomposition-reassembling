import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from moldr.chemutils import get_mol


def get_default_config(
    env,
    sc,
    building_blocks_smiles,
    model_path,
    base_smiles="C",
    num_workers=40,
    num_gpus=2,
    score_threshold=0.9,
    length=100,
):
    def check_valid_smiles(smiles):
        mol = get_mol(smiles)
        if mol is None:
            raise ValueError("INVALID SMILES.")
        else:
            return base_smiles

    base_smiles = check_valid_smiles(base_smiles)
    high = np.array([np.finfo(np.float32).max for _ in range(300)])
    observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
    scoring_function = sc.wrapped_objective.score

    # Environment configuration dictionary
    env_config = {
        "ACTION_SPACE": gym.spaces.Discrete(len(building_blocks_smiles)),
        "OBS_SPACE": observation_space,
        "BUILDING_BLOCKS": building_blocks_smiles,
        "SCORE_FUNCTION": scoring_function,
        "SCORE_THRESHOLD": score_threshold,
        "FINAL_WEIGHT": 1.0,
        "BASE_SMILES": base_smiles,
        "MODEL_PATH": model_path,
        "LENGTH": length,
    }

    # Create PPO configuration using the new API
    config = (
        PPOConfig()
        .environment(env=env, env_config=env_config)
        .framework("torch")
        .resources(
            num_gpus=num_gpus,
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            sample_timeout_s=360.0,
        )
        .learners(
            num_learners=1,  # Need to set to 0 for RLModule due to bug.
            num_gpus_per_learner=1,
        )
        # .rl_module(
        #     model_config={
        #         "fcnet_hiddens": [256, 128, 128],
        #         "fcnet_activation": "relu",
        #         "max_seq_len": 100,
        #     }
        # )
        .training(
            train_batch_size_per_learner=4000,
            minibatch_size=128,
            num_epochs=30,
            lr=5e-5,
            lambda_=0.95,
            gamma=0.99,
            vf_loss_coeff=1.0,
            clip_param=0.3,
            grad_clip=None,
            entropy_coeff=0.1,
        )
        # .evaluation(
        #     evaluation_interval=10,
        #     evaluation_num_env_runners=1,
        #     evaluation_config={
        #         "render_env": False,
        #     },
        # )
    )

    return config
