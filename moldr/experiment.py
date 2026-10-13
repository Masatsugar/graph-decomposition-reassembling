"""Strict experiment settings shared by training, sampling and profiling.

Precedence is defaults, YAML, explicit flags, then ordered ``--set`` overrides.
Loading a configuration never starts Ray or reads model weights.
"""

from __future__ import annotations

import copy
import math
import types
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import yaml


@dataclass
class EnvironmentConfig:
    mode: str = "e2e"
    base_smiles: str = "C"
    atom_limit: int = 100
    max_episode_steps: int | None = None
    reward: str | None = None
    score_threshold: float = 0.9
    final_weight: float = 1.0
    terminate_on_score_decrease: bool | None = None
    max_attach: int = 128


@dataclass
class FragmentsConfig:
    source: str = "fixed"
    path: str = "data/building_blocks/guacamol/guacamol_s10000_l2.csv"
    smiles: list[str] = field(default_factory=list)
    proposer_dir: str | None = None
    pool_size: int | None = None
    recond_every: str = "episode"


@dataclass
class ChemistryConfig:
    map_storage: str = "int"
    shared_preparation: bool = True
    bond_scan: str = "nonzero"
    candidate_representation: str = "mapped"
    dedup_key: str = "kekule"
    mol_transfer: str = "smiles"


@dataclass
class ScoringConfig:
    objective: str = "QED"
    input_mode: str = "smiles"
    cache_size: int = 0


@dataclass
class RepresentationConfig:
    name: str = "mol2vec"
    trainable: bool = False
    model_path: str = "models/model_300dim.pkl"
    obs_dim: int = 300
    layers: int = 3
    hidden: int = 128
    output: int = 128
    max_atoms: int | None = None
    max_bonds: int | None = None
    atom_chunk_size: int = 4096
    gradient_checkpointing: bool = False
    structure_cache_size: int = 1024
    embedding_cache_size: int = 0
    reuse_embeddings: bool = True


@dataclass
class SelectionConfig:
    mode: str = "actor"
    context: str = "rich"
    candidate_value: bool = False
    auxiliary_loss_coeff: float = 0.5
    alpha: float = 1.0
    temperature: float = 1.0


@dataclass
class TrainingConfig:
    seed: int = 0
    iterations: int = 100
    num_workers: int = 8
    num_cpus: int | None = None
    num_gpus: float = 0.0
    train_batch_size: int = 4000
    minibatch_size: int = 128
    num_epochs: int = 30
    hidden: int = 256
    lr: float = 5e-5
    gamma: float = 0.99
    lambda_: float = 0.95
    vf_loss_coeff: float = 1.0
    clip_param: float = 0.3
    grad_clip: float | None = None
    entropy_coeff: float = 0.1
    sample_timeout_s: float = 360.0
    rollout_fragment_length: int | None = None
    torch_num_threads: int = 1


@dataclass
class EvaluationConfig:
    selector: str = "actor"
    episodes: int = 12
    explore: bool = False
    random: bool = False


@dataclass
class ProfilingConfig:
    trace: bool = False
    warmup_iterations: int = 3
    measured_iterations: int = 10
    time_budget_seconds: float | None = None
    max_events: int = 250000


@dataclass
class OutputConfig:
    directory: str = "outputs/experiments"
    run_name: str | None = None
    checkpoint: str | None = None
    checkpoint_every: int = 20
    logger: str = "local"
    project_name: str = "moldr"


@dataclass
class ExperimentConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    fragments: FragmentsConfig = field(default_factory=FragmentsConfig)
    chemistry: ChemistryConfig = field(default_factory=ChemistryConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.environment.reward is None:
            self.environment.reward = "sparse" if self.environment.mode == "legacy" else "improvement"
        if self.environment.terminate_on_score_decrease is None:
            self.environment.terminate_on_score_decrease = self.environment.mode == "legacy"
        choices = {
            "environment.mode": {"e2e", "legacy"},
            "environment.reward": {"improvement", "sparse"},
            "fragments.source": {"fixed", "vae"},
            "fragments.recond_every": {"never", "episode", "step"},
            "chemistry.map_storage": {"int", "string"},
            "chemistry.bond_scan": {"nonzero", "python"},
            "chemistry.candidate_representation": {"mapped", "mapless"},
            "chemistry.dedup_key": {"kekule", "aromatic"},
            "chemistry.mol_transfer": {"smiles", "direct"},
            "scoring.input_mode": {"smiles", "mol"},
            "representation.name": {"mol2vec", "mpnn"},
            "selection.mode": {"actor", "actor_value"},
            "selection.context": {"classic", "rich"},
            "evaluation.selector": {"actor", "value_lookahead"},
            "output.logger": {"local", "tensorboard", "wandb"},
        }
        for path, allowed in choices.items():
            value = get_setting(self, path)
            if value not in allowed:
                raise ValueError(f"{path}: expected one of {sorted(allowed)}, got {value!r}")
        for section in fields(self):
            obj = getattr(self, section.name)
            for entry in fields(obj):
                value = getattr(obj, entry.name)
                _check_type(value, get_type_hints(type(obj))[entry.name], f"{section.name}.{entry.name}")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{section.name}.{entry.name} must be finite")
        positive = ["environment.atom_limit", "environment.max_attach", "environment.max_episode_steps",
                    "fragments.pool_size", "representation.obs_dim", "representation.layers",
                    "representation.hidden", "representation.output", "representation.max_atoms",
                    "representation.max_bonds", "representation.atom_chunk_size", "selection.temperature",
                    "training.num_cpus", "training.train_batch_size", "training.minibatch_size",
                    "training.num_epochs", "training.hidden", "training.lr", "training.sample_timeout_s",
                    "training.rollout_fragment_length", "training.torch_num_threads", "evaluation.episodes",
                    "profiling.measured_iterations", "profiling.time_budget_seconds", "profiling.max_events",
                    "output.checkpoint_every"]
        for path in positive:
            value = get_setting(self, path)
            if value is not None and value <= 0:
                raise ValueError(f"{path} must be positive")
        for path in ["scoring.cache_size", "representation.structure_cache_size", "representation.embedding_cache_size",
                     "selection.auxiliary_loss_coeff", "selection.alpha", "training.seed", "training.iterations",
                     "training.num_workers", "training.num_gpus", "profiling.warmup_iterations", "environment.final_weight"]:
            if get_setting(self, path) < 0:
                raise ValueError(f"{path} must be nonnegative")
        if self.training.num_gpus > 1:
            raise ValueError("This shared legacy PPO factory supports at most one learner GPU")
        if self.training.num_cpus is not None and self.training.num_cpus < self.training.num_workers + 1:
            raise ValueError("training.num_cpus must cover all workers and the PPO driver")
        if self.training.minibatch_size > self.training.train_batch_size:
            raise ValueError("training.minibatch_size exceeds train_batch_size")
        if not 0 <= self.training.gamma <= 1 or not 0 <= self.training.lambda_ <= 1:
            raise ValueError("gamma and lambda_ must lie in [0,1]")
        if self.chemistry.mol_transfer == "direct" and self.chemistry.candidate_representation != "mapless":
            raise ValueError("chemistry.mol_transfer=direct requires candidate_representation=mapless")
        if self.representation.name == "mol2vec" and self.representation.trainable:
            raise ValueError("Mol2Vec observations are fixed; joint encoder training requires mpnn")
        if self.representation.name == "mol2vec" and (self.representation.embedding_cache_size or self.representation.gradient_checkpointing):
            raise ValueError("Persistent embedding caches and gradient checkpointing require mpnn; Mol2Vec supports reuse_embeddings")
        if self.representation.trainable and self.representation.embedding_cache_size:
            raise ValueError("Embedding caches require a frozen encoder")
        if self.selection.mode == "actor_value" and not self.selection.candidate_value:
            raise ValueError("selection.mode=actor_value requires selection.candidate_value=true")
        if self.fragments.source == "vae" and not self.fragments.proposer_dir:
            raise ValueError("fragments.source=vae requires proposer_dir")
        if self.fragments.source == "fixed" and self.fragments.proposer_dir:
            raise ValueError("proposer_dir is only supported with fragments.source=vae")
        if self.environment.mode == "legacy":
            if self.representation.name != "mol2vec" or self.selection.mode != "actor" or self.selection.candidate_value:
                raise ValueError("legacy requires fixed Mol2Vec and actor without candidate value")
            if self.fragments.pool_size is not None:
                raise ValueError("legacy does not support a sampled fragment pool")
        if self.environment.mode == "e2e" and self.environment.terminate_on_score_decrease:
            raise ValueError("score-decrease termination is only supported by legacy")
        if self.evaluation.selector == "value_lookahead" and (self.environment.mode != "e2e" or self.fragments.source != "fixed" or self.fragments.pool_size is not None):
            raise ValueError("value_lookahead requires E2E, fixed fragments and the full fragment table")
        if self.evaluation.random and self.evaluation.selector == "value_lookahead":
            raise ValueError("Random evaluation cannot use value_lookahead")
        return self


def _check_type(value, annotation, path):
    origin = get_origin(annotation)
    if origin is types.UnionType:
        if any(_type_matches(value, arg) for arg in get_args(annotation)):
            return
    elif _type_matches(value, annotation):
        return
    raise ValueError(f"{path}: value {value!r} does not match {annotation}")


def _type_matches(value, annotation):
    if annotation is type(None):
        return value is None
    if annotation is float:
        return type(value) in (int, float)
    if get_origin(annotation) is list:
        return isinstance(value, list) and all(_type_matches(v, get_args(annotation)[0]) for v in value)
    return type(value) is annotation


def get_setting(config, path):
    section, key = path.split(".", 1)
    return getattr(getattr(config, section), key)


def _merge_dataclass(obj, values, prefix=""):
    if not isinstance(values, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")  # noqa: TRY004 -- uniform configuration errors
    known = {f.name: f for f in fields(obj)}
    hints = get_type_hints(type(obj))
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in known:
            raise ValueError(f"Unknown config key: {path}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _merge_dataclass(current, value, path)
        else:
            _check_type(value, hints[key], path)
            setattr(obj, key, value)


def config_from_dict(values, *, validate=True):
    config = ExperimentConfig()
    _merge_dataclass(config, values)
    return config.validate() if validate else config


def load_config(path=None, overrides=(), cli_values=None, *, base=None, validate=True):
    config = copy.deepcopy(base) if base is not None else ExperimentConfig()
    if path is not None:
        _merge_dataclass(config, yaml.safe_load(Path(path).read_text()) or {})
    for key, value in (cli_values or {}).items():
        if value is not None:
            overrides = [f"{key}={yaml.safe_dump(value, default_flow_style=True).strip().removesuffix('...').strip()}", *overrides]
    for override in overrides:
        path_key, sep, raw = override.partition("=")
        if not sep or path_key.count(".") != 1:
            raise ValueError(f"Expected --set section.key=value, got {override!r}")
        section, key = path_key.split(".")
        _merge_dataclass(config, {section: {key: yaml.safe_load(raw)}})
    return config.validate() if validate else config


def config_diff(baseline, config):
    before, after = baseline.to_dict(), config.to_dict()
    return {f"{section}.{key}": {"before": before[section][key], "after": value}
            for section, values in after.items() for key, value in values.items()
            if before[section][key] != value}
