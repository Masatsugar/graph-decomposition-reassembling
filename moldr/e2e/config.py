"""RLlib PPO configuration for the end-to-end reassembly environment.

Mirrors :func:`moldr.config.get_default_config_v1` (old API stack / ModelV2),
but wires up the two-phase :class:`~moldr.e2e.env.MolEnvE2E` with the
action-masking :class:`~moldr.e2e.model.MoldrPointerModel`.
"""

from __future__ import annotations

from typing import List, Optional

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from moldr.chemutils import get_mol
from moldr.e2e.env import MolEnvE2E
from moldr.e2e.model import MODEL_NAME, register_pointer_model

ENV_NAME = "moldr_e2e"


def e2e_env_creator(env_config):
    return MolEnvE2E(env_config)


def get_e2e_config(
    sc,
    building_blocks_smiles: Optional[List[str]] = None,
    model_path: str = "models/model_300dim.pkl",
    base_smiles: str = "C",
    num_workers: int = 8,
    num_gpus: int = 0,
    score_threshold: float = 0.9,
    length: int = 100,
    obs_dim: int = 300,
    max_attach: int = 128,
    option_pool_size: Optional[int] = None,
    reward_mode: str = "improvement",
    hidden: int = 256,
    proposer=None,
    recond_every: str = "episode",
    max_episode_steps: int | None = None,
):
    """Build a PPOConfig for the end-to-end reassembly environment.

    ``option_pool_size`` controls phase A: ``None`` offers every building block
    each step (faithful fixed-table baseline); an int ``K`` samples K blocks.
    ``sc`` is a GuacaMol benchmark whose ``wrapped_objective.score`` is the
    scoring function.

    Generative variant: pass ``proposer`` (a :class:`VAEProposer` or a
    ``{"dir": ...}`` spec). Phase A then offers *generated* building blocks,
    re-decoded per ``recond_every`` in {"never", "episode", "step"}. The action
    space stays ``Discrete`` of the proposer's N, so PPO is unchanged.

    ``max_episode_steps`` bounds decisions even when an attachment adds no
    atoms. By default it is twice ``length``; hitting it truncates the episode
    so PPO can bootstrap the value of the unfinished molecule.
    """
    register_env(ENV_NAME, e2e_env_creator)
    register_pointer_model()

    if get_mol(base_smiles) is None:
        raise ValueError("INVALID BASE SMILES.")

    scoring_function = sc.wrapped_objective.score

    if proposer is None and building_blocks_smiles is None:
        raise ValueError("Pass either building_blocks_smiles or a proposer.")
    blocks = list(building_blocks_smiles) if building_blocks_smiles is not None else ["C"]
    if proposer is None and not blocks:
        raise ValueError("BUILDING_BLOCKS must not be empty.")

    # The env derives the pool from the proposer when one is given; otherwise it
    # uses the fixed table. A placeholder table keeps the env_config schema
    # uniform when a proposer is supplied.
    env_config = {
        "BUILDING_BLOCKS": blocks,
        "SCORE_FUNCTION": scoring_function,
        "MODEL_PATH": model_path,
        "BASE_SMILES": base_smiles,
        "SCORE_THRESHOLD": score_threshold,
        "LENGTH": length,
        "MAX_EPISODE_STEPS": (
            max_episode_steps if max_episode_steps is not None else 2 * length
        ),
        "OBS_DIM": obs_dim,
        "MAX_ATTACH": max_attach,
        "OPTION_POOL_SIZE": option_pool_size,
        "REWARD_MODE": reward_mode,
        "FINAL_WEIGHT": 1.0,
    }
    if proposer is not None:
        env_config["PROPOSER"] = proposer
        env_config["RECOND_EVERY"] = recond_every

    config = (
        PPOConfig()
        .environment(ENV_NAME, env_config=env_config)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        # The pointer model consumes the Dict directly. Flattening the padded
        # option matrix copies potentially millions of elements per decision,
        # then ModelV2 immediately reconstructs the same Dict again.
        .experimental(_disable_preprocessor_api=True)
        .resources(num_gpus=num_gpus)
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            sample_timeout_s=360.0,
        )
        .training(
            model={
                "custom_model": MODEL_NAME,
                "custom_model_config": {"obs_dim": obs_dim, "hidden": hidden},
            },
            train_batch_size=4000,
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
    )
    return config
