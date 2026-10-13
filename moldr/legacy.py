"""Stable entry points for the original environment and Python factories."""

from moldr.config import (
    get_default_config_v1,
    get_default_config_v2,
    get_generative_config_v1,
)
from moldr.env import MolEnvValueMax

__all__ = ["MolEnvValueMax", "get_default_config_v1", "get_default_config_v2", "get_generative_config_v1"]


def restore_checkpoint(path):
    """Restore an old legacy or E2E checkpoint with its original architecture."""
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.tune.registry import register_env

    from moldr.config import env_creator
    from moldr.e2e.config import ENV_NAME, e2e_env_creator
    from moldr.e2e.model import register_pointer_model

    register_env("moldr_env", env_creator)
    register_env(ENV_NAME, e2e_env_creator)
    register_pointer_model()
    return Algorithm.from_checkpoint(path)


__all__.append("restore_checkpoint")
