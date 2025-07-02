import os
import torch
import numpy as np
import gymnasium as gym

from guacamol.standard_benchmarks import logP_benchmark

from moldr.env import MolEnvValueMax
from moldr.config import get_default_config


def test_env():
    objective = logP_benchmark(8.0)
    building_blocks_smiles = ["CC", "CCO", "CCCC"]
    config = get_default_config(
        objective,
        building_blocks_smiles,
        model_path="models/model_300dim.pkl",
    )
    algo = config.build()
    # env = MolEnvValueMax(config)
    env = algo.env_creator(config.env_config)
    obs, info = env.reset()
    assert len(obs) == 300  # mol2vec dim

    rewards = 0.0
    while True:
        action = np.random.choice(len(building_blocks_smiles))
        obs, reward, done, truncated, info = env.step(action)
        rewards += reward
        if done or truncated:
            break

    print(info)


def run_env():
    cwd = os.getcwd()
    building_blocks_smiles = ["CC", "CCC", "CO", "CN", "CF"]
    objective = logP_benchmark(8.0)
    config = get_default_config(
        MolEnvValueMax,
        objective,
        building_blocks_smiles,
        model_path=os.path.join(cwd, "models/model_300dim.pkl"),
        num_workers=8,
        num_gpus=1,
    )
    rl_module = RLModule.from_checkpoint(checkpoint_path)

    env = MolEnvValueMax(config.env_config)

    done = False
    obs, info = env.reset()
    episode_return = 0.0
    while not done:
        # Uncomment this line to render the env.
        # env.render()

        # Compute the next action from a batch (B=1) of observations.
        obs_batch = torch.from_numpy(obs).unsqueeze(0)  # add batch B=1 dimension
        model_outputs = rl_module.forward_inference({"obs": obs_batch})

        # Extract the action distribution parameters from the output and dissolve batch dim.
        action_dist_params = model_outputs["action_dist_inputs"][0].numpy()

        if isinstance(env.action_space, gym.spaces.Box):
            # We have continuous actions -> take the mean (max likelihood).
            greedy_action = np.clip(
                action_dist_params[
                    0:1
                ],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
                a_min=env.action_space.low[0],
                a_max=env.action_space.high[0],
            )
        elif isinstance(env.action_space, gym.spaces.Discrete):
            # For discrete actions, you should take the argmax over the logits:
            greedy_action = np.argmax(action_dist_params)
        else:
            raise TypeError(f"Unsupported action space: {env.action_space}")

        # Send the action to the environment for the next step.
        obs, reward, terminated, truncated, info = env.step(greedy_action)

        # Perform env-loop bookkeeping.
        episode_return += reward
        done = terminated or truncated

    print(f"Reached episode return of {episode_return}.")
    print(info)


if __name__ == "__main__":
    test_env()
