"""One factory for all experiment entry points, on RLlib's existing API stack."""

from __future__ import annotations

import copy
from dataclasses import asdict
from functools import lru_cache
from numbers import Integral
from pathlib import Path

import numpy as np

from moldr.experiment import ExperimentConfig

ENV_NAME_V2 = "moldr_e2e_v2"


@lru_cache(maxsize=8)
def _mol2vec_metadata(path, modified_ns, size):
    from gensim.models import Word2Vec

    model = Word2Vec.load(path)
    radius = getattr(model, "moldr_radius", None)
    if radius is not None:
        if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 0:
            raise ValueError(
                "Mol2Vec moldr_radius metadata must be a nonnegative integer"
            )
        radius = int(radius)
    else:
        bundled = Path(__file__).resolve().parents[1] / "models/model_300dim.pkl"
        if bundled.is_file() and Path(path).samefile(bundled):
            radius = 1
    # Token absence in a few probe molecules cannot establish the training
    # radius of an arbitrary model. Custom models use the configured radius
    # unless they declare it explicitly.
    return model.wv.vector_size, radius


@lru_cache(maxsize=8)
def _embedding_dimension(path, modified_ns, size):
    return _mol2vec_metadata(path, modified_ns, size)[0]


@lru_cache(maxsize=8)
def _load_fixed_file(resolved_path, modified_ns, size):
    import pandas as pd

    from moldr.chemutils import get_mol, validate_reassembly_mol

    path = Path(resolved_path)
    if path.suffix == ".csv":
        raw_blocks = pd.read_csv(path, index_col=0).iloc[:, 0].tolist()
    else:
        raw_blocks = [
            line.split()[0] for line in path.read_text().splitlines() if line.strip()
        ]
    if not raw_blocks:
        raise ValueError("Fragment table must not be empty")
    max_atoms = 0
    for smiles in raw_blocks:
        mol = get_mol(smiles)
        if mol is None:
            raise ValueError(
                "Initial molecule and fragment catalog must contain valid SMILES"
            )
        validate_reassembly_mol(mol)
        max_atoms = max(max_atoms, mol.GetNumAtoms())
    # Entries are policy action indices: repeats and ordering must survive
    # restoration of an existing checkpoint.
    return tuple(raw_blocks), max_atoms


def load_fragments(config):
    if config.fragments.source == "vae":
        from moldr.chemutils import get_mol, validate_reassembly_mol
        from moldr.generative.proposer import VAEProposer

        proposer = VAEProposer.from_spec({"dir": config.fragments.proposer_dir})
        if config.fragments.recond_every != "never" and not proposer.context_trained:
            raise ValueError(
                "A proposer trained without contexts requires fragments.recond_every=never"
            )
        if config.fragments.recond_every == "never":
            blocks = list(proposer.propose_library(None))
        else:
            if not Path(config.representation.model_path).is_file():
                raise ValueError(
                    f"Mol2Vec asset is missing: {config.representation.model_path}"
                )
            base_mol = get_mol(config.environment.base_smiles)
            if base_mol is None:
                raise ValueError(
                    "Initial molecule and fragment catalog must contain valid SMILES"
                )
            validate_reassembly_mol(base_mol)
            from moldr.mol2vec.model import Mol2Vec

            radius = (
                config.representation.radius
                if config.representation.name == "mol2vec"
                else 1
            )
            vec = np.asarray(
                Mol2Vec(
                    model_path=config.representation.model_path, radius=radius
                ).fit_transform([base_mol]),
                dtype=np.float32,
            )
            vec = np.clip(
                np.nan_to_num(vec, nan=0.0, posinf=1.0e4, neginf=-1.0e4),
                -1.0e4,
                1.0e4,
            ).reshape(-1)
            blocks = list(proposer.propose(vec))
        return blocks, list(proposer.vocab.smiles)
    if config.fragments.smiles:
        blocks = list(config.fragments.smiles)
    else:
        path = Path(config.fragments.path).resolve()
        stat = path.stat()
        blocks_tuple, _ = _load_fixed_file(str(path), stat.st_mtime_ns, stat.st_size)
        blocks = list(blocks_tuple)
    if not blocks:
        raise ValueError("Fragment table must not be empty")
    return blocks, blocks


def resolve_config(config: ExperimentConfig):
    """Resolve capacities and validate all assets before constructing Ray actors."""
    from moldr.chemutils import get_mol, validate_reassembly_mol
    from moldr.scoring import get_objective

    config = copy.deepcopy(config).validate()
    objective = get_objective(config.scoring.objective)
    if config.scoring.input_mode == "mol" and objective.mol_function is None:
        raise ValueError(
            f"Objective {config.scoring.objective!r} has no Mol evaluation path"
        )
    if (
        config.scoring.cache_size or config.evaluation.selector == "value_lookahead"
    ) and not objective.deterministic:
        raise ValueError(
            "Caching and value lookahead require a deterministic registered objective"
        )
    if (
        config.representation.name == "mol2vec" or config.fragments.source == "vae"
    ) and not Path(config.representation.model_path).is_file():
        raise ValueError(
            f"Mol2Vec asset is missing: {config.representation.model_path}"
        )
    if config.representation.name == "mol2vec" or config.fragments.source == "vae":
        asset = Path(config.representation.model_path)
        stat = asset.stat()
        size, detected_radius = _mol2vec_metadata(
            str(asset.resolve()), stat.st_mtime_ns, stat.st_size
        )
        required = (
            config.representation.obs_dim
            if config.representation.name == "mol2vec"
            else 300
        )
        if size != required:
            raise ValueError(
                f"Mol2Vec asset dimension {size} does not match required dimension {required}"
            )
        required_radius = (
            config.representation.radius
            if config.representation.name == "mol2vec"
            else 1
        )
        if detected_radius is not None and detected_radius != required_radius:
            raise ValueError(
                f"Mol2Vec asset radius {detected_radius} does not match required radius {required_radius}"
            )
        if config.fragments.source == "vae":
            from moldr.generative.proposer import VAEProposer

            proposer = VAEProposer.from_spec({"dir": config.fragments.proposer_dir})
            if proposer.vae.context_dim != size:
                raise ValueError(
                    f"VAE context dimension {proposer.vae.context_dim} does not match frozen Mol2Vec dimension {size}"
                )
    base_mol = get_mol(config.environment.base_smiles)
    if base_mol is None:
        raise ValueError(
            "Initial molecule and fragment catalog must contain valid SMILES"
        )
    validate_reassembly_mol(base_mol)
    blocks, catalog = load_fragments(config)
    if config.fragments.source == "fixed" and not config.fragments.smiles:
        path = Path(config.fragments.path).resolve()
        stat = path.stat()
        _, max_fragment_atoms = _load_fixed_file(
            str(path), stat.st_mtime_ns, stat.st_size
        )
    else:
        mols = [get_mol(s) for s in catalog]
        for mol in mols:
            if mol is None:
                raise ValueError(
                    "Initial molecule and fragment catalog must contain valid SMILES"
                )
            validate_reassembly_mol(mol)
        max_fragment_atoms = max(m.GetNumAtoms() for m in mols)
    if config.environment.max_episode_steps is None:
        config.environment.max_episode_steps = (
            2 if config.environment.mode == "e2e" else 1
        ) * config.environment.atom_limit
    if config.representation.name == "mpnn":
        required_atoms = (
            max(base_mol.GetNumAtoms(), config.environment.atom_limit)
            + max_fragment_atoms
        )
        # A chemical adjacency merge may add bonds without adding atoms. The
        # complete simple graph is a conservative bound, including that case.
        required_bonds = required_atoms * (required_atoms - 1) // 2
        for field, required in (
            ("max_atoms", required_atoms),
            ("max_bonds", required_bonds),
        ):
            current = getattr(config.representation, field)
            if current is None:
                setattr(config.representation, field, required)
            elif current < required:
                raise ValueError(
                    f"representation.{field}={current} is below derived capacity {required}"
                )
    return config, blocks, catalog


def make_env_config(config, blocks=None, catalog=None):
    from moldr.scoring import BUILTIN_OBJECTIVES, get_objective

    if blocks is None:
        config, blocks, catalog = resolve_config(config)
    objective = get_objective(config.scoring.objective)
    scoring = asdict(config.scoring)
    if BUILTIN_OBJECTIVES.get(objective.name) is not objective:
        # Runtime registrations are absent in a fresh rollout worker. Ship the
        # immutable definition and its evaluators with the environment spec.
        scoring["objective"] = objective
    env = {
        "BUILDING_BLOCKS": blocks,
        "CATALOG_SMILES": catalog or blocks,
        "SCORE_FUNCTION": objective.smiles_function,
        "MODEL_PATH": config.representation.model_path,
        "BASE_SMILES": config.environment.base_smiles,
        "SCORE_THRESHOLD": config.environment.score_threshold,
        "LENGTH": config.environment.atom_limit,
        "MAX_EPISODE_STEPS": config.environment.max_episode_steps,
        "OBS_DIM": config.representation.obs_dim,
        "MAX_ATTACH": config.environment.max_attach,
        "OPTION_POOL_SIZE": config.fragments.pool_size,
        "REWARD_MODE": config.environment.reward,
        "FINAL_WEIGHT": config.environment.final_weight,
        "TERMINATE_ON_SCORE_DECREASE": config.environment.terminate_on_score_decrease,
        "CHEMISTRY": asdict(config.chemistry),
        "SCORING": scoring,
        "REPRESENTATION": asdict(config.representation),
        "CONTEXT": config.selection.context,
        "SEED": config.training.seed,
    }
    if config.fragments.source == "vae":
        env["PROPOSER"] = {"dir": config.fragments.proposer_dir}
        env["RECOND_EVERY"] = config.fragments.recond_every
    if config.environment.mode == "legacy":
        import gymnasium as gym

        high = np.full(config.representation.obs_dim, np.finfo(np.float32).max)
        env["ACTION_SPACE"] = gym.spaces.Discrete(len(blocks))
        env["OBS_SPACE"] = gym.spaces.Box(-high, high, dtype=np.float32)
    return env


def e2e_v2_env_creator(env_config):
    from moldr.e2e.env_v2 import MolEnvE2EV2

    return MolEnvE2EV2(env_config)


def make_environment(config):
    env_config = make_env_config(config)
    if config.environment.mode == "legacy":
        from moldr.env import MolEnvValueMax

        return MolEnvValueMax(env_config)
    return e2e_v2_env_creator(env_config)


def make_ppo_config(config: ExperimentConfig):
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    config, blocks, catalog = resolve_config(config)
    env_config = make_env_config(config, blocks, catalog)
    if config.environment.mode == "legacy":
        from moldr.config import env_creator

        name = "moldr_env"
        register_env(name, env_creator)
        model = {
            "fcnet_hiddens": [config.training.hidden, 128, 128],
            "fcnet_activation": "relu",
            "max_seq_len": 100,
        }
    else:
        from moldr.e2e.action_dist import ACTION_DIST_NAME
        from moldr.e2e.model_v2 import MODEL_NAME_V2, register_pointer_model_v2

        name = ENV_NAME_V2
        register_env(name, e2e_v2_env_creator)
        register_pointer_model_v2()
        representation = config.representation
        model = {
            "custom_model": MODEL_NAME_V2,
            "custom_action_dist": ACTION_DIST_NAME,
            "custom_model_config": {
                "representation": representation.name,
                "trainable": representation.trainable,
                "obs_dim": representation.obs_dim,
                "hidden": config.training.hidden,
                "layers": representation.layers,
                "graph_hidden": representation.hidden,
                "graph_output": representation.output,
                "atom_chunk_size": representation.atom_chunk_size,
                "gradient_checkpointing": representation.gradient_checkpointing,
                "embedding_cache_size": representation.embedding_cache_size,
                "context": config.selection.context,
                "candidate_value": config.selection.candidate_value,
                "alpha": config.selection.alpha
                if config.selection.mode == "actor_value"
                else 0.0,
                "temperature": config.selection.temperature,
                "auxiliary_loss_coeff": config.selection.auxiliary_loss_coeff,
                "catalog_smiles": catalog,
                "max_attach": config.environment.max_attach,
            },
        }
    training = config.training
    ppo = (
        PPOConfig()
        .environment(name, env_config=env_config)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .resources(num_gpus=training.num_gpus)
        .env_runners(
            num_env_runners=training.num_workers,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            sample_timeout_s=training.sample_timeout_s,
            rollout_fragment_length=training.rollout_fragment_length or "auto",
        )
        .training(
            model=model,
            train_batch_size=training.train_batch_size,
            minibatch_size=training.minibatch_size,
            num_epochs=training.num_epochs,
            lr=training.lr,
            gamma=training.gamma,
            lambda_=training.lambda_,
            vf_loss_coeff=training.vf_loss_coeff,
            clip_param=training.clip_param,
            grad_clip=training.grad_clip,
            entropy_coeff=training.entropy_coeff,
        )
        .debugging(seed=training.seed, log_level="ERROR")
        .reporting(
            min_sample_timesteps_per_iteration=training.train_batch_size,
            min_time_s_per_iteration=0,
        )
    )
    if config.environment.mode == "e2e":
        ppo.experimental(_disable_preprocessor_api=True)
    return ppo
