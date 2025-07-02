import os
import multiprocessing
import timeit

import torch

import numpy as np
import pandas as pd
import ray

from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.standard_benchmarks import logP_benchmark

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO

from moldr.env import MolEnvValueMax
from moldr.utils import load

# import torch.multiprocessing as multiprocessing
register_env("moldr_env", lambda env_config: MolEnvValueMax(env_config))


class Sampler:
    def __init__(self, env, algo, threshold=0.9, n_jobs=16):
        self.env = env
        self.n_jobs = n_jobs
        self.threshold = threshold
        self.n_building_blocks = len(self.env.building_blocks)
        self.algo = algo
        self.env.reset()

    def sampling(self, y):
        x, seed = y
        np.random.seed(seed)
        obs, info = self.env.reset()
        done = False
        prev_info = None
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = self.algo.compute_single_action(obs_tensor)
            obs, reward, done, truncated, info = self.env.step(action)
            if reward > self.threshold:
                return prev_info
            prev_info = info
        return info

    def random(self, y):
        x, seed = y
        np.random.seed(seed)
        obs, info = self.env.reset()
        done = False
        # prev_info = None
        while not done:
            action = np.random.choice(self.n_building_blocks)
            obs, reward, done, truncated, info = self.env.step(action)

        if info["gen_smiles"]:
            return np.random.choice(info["gen_smiles"])
        else:
            return info["prev_smiles"]

    def _generate(self, number_samples):
        with multiprocessing.Pool(self.n_jobs) as p:
            smiles = list(
                p.map(
                    self.random,
                    zip(
                        range(number_samples),
                        range(number_samples),
                        # np.random.randint(0, 2**32 - 1, number_samples),
                    ),
                )
            )
        return smiles

    def generate(self, number_samples):
        smiles = Parallel(n_jobs=self.n_jobs)(
            delayed(self.random)(x)
            for x in zip(range(number_samples), range(number_samples))
        )
        return smiles

    def generate_single(self, number_samples):
        smiles = []
        for i in tqdm(range(number_samples)):
            s = self.sampling((i, i))
            smiles.append(s)
        return smiles

    def transition(self):
        obs, info = self.env.reset()
        done = False
        rewards = 0.0
        info_list = []
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = self.algo.compute_single_action(obs_tensor)
            obs, reward, done, truncated, info = self.env.step(action)
            rewards += reward
            info_list.append(info)

        final_scores = self.env.compute_score(info_list[-1]["gen_smiles"])
        final_smiles = info_list[-1]["gen_smiles"][np.argmax(final_scores)]
        rewards += np.max(final_scores)
        info_list.append({"prev_smiles": final_smiles})
        print("total reward:", rewards, "final reward:", np.max(final_scores))
        return info_list


if __name__ == "__main__":
    objective = logP_benchmark(8.0)
    checkpoint_dir = (
        Path.cwd()
        / "outputs"
        / "PPO"
        / "models"
        / objective.name
        / "2025-07-02_23-02-29"
    ).resolve()
    env_config = load(checkpoint_dir / "env_config.pkl")
    algo = PPO.from_checkpoint(checkpoint_dir)

    # Old way to get the RL module
    # rl_module = algo.get_policy("default_policy")
    # print(rl_module)

    env = algo.env_creator(env_config)
    sampler = Sampler(env, algo, threshold=1.0, n_jobs=32)
    start = timeit.default_timer()
    print("Generating...")
    results = sampler.generate_single(100)
    stop = timeit.default_timer()
    print("Time: ", stop - start)

    gen_smiles = []
    for s in results:
        if s is not None:
            gen_smiles.extend(s["gen_smiles"])

    gen_smiles = np.unique(gen_smiles)
    scores = [env.scoring_function(s) for s in tqdm(gen_smiles)]
    _result = pd.DataFrame({"smiles": gen_smiles, "score": scores}).sort_values(
        "score", ascending=False
    )
    # gen_scores = list([env.scoring_function(s) for s in tqdm(gen_smiles)])
    top100 = _result.iloc[0:100, :]
    print("Top 100 Score mean:", top100.score.mean())
    os.makedirs("outputs/results", exist_ok=True)
    _result.to_csv(f"outputs/results/{objective.name}.csv")
