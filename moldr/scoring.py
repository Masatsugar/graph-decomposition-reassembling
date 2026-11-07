"""Deterministic objective registry, Mol evaluation, and counted LRU scoring."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen


@dataclass(frozen=True)
class Objective:
    name: str
    smiles_function: Callable
    mol_function: Callable | None = None
    deterministic: bool = True


OBJECTIVE_REGISTRY: dict[str, Objective] = {}


def _name(name):
    normalized = str(name).strip().lower().replace("_", "-")
    aliases = {
        "qedsa": "qed-sa",
        "logp8": "logp-8",
        "logp (target: 8.0)": "logp-8",
        "logp (target = 8.0)": "logp-8",
        "logp-1": "logp-minus1",
        "logp (target: -1.0)": "logp-minus1",
        "logp (target = -1.0)": "logp-minus1",
    }
    return aliases.get(normalized, normalized)


def register_objective(name, smiles_function, mol_function=None, deterministic=True):
    """Register a pure deterministic objective explicitly to permit caching."""
    if not callable(smiles_function) or (
        mol_function is not None and not callable(mol_function)
    ):
        raise TypeError("Objective evaluators must be callable.")
    objective = Objective(
        _name(name), smiles_function, mol_function, bool(deterministic)
    )
    OBJECTIVE_REGISTRY[objective.name] = objective
    return objective


def get_objective(name):
    normalized = _name(name)
    try:
        return OBJECTIVE_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown objective {name!r}; registered objectives: {sorted(OBJECTIVE_REGISTRY)}"
        ) from exc


def _parse(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Scoring requires a valid molecule.")
    return mol


def _qed_sa(mol):
    from moldr.sascore import calculateScore

    return max(0.0, QED.qed(mol) - 0.1 * calculateScore(mol))


register_objective("qed", lambda smi: QED.qed(_parse(smi)), QED.qed)
register_objective("qed-sa", lambda smi: _qed_sa(_parse(smi)), _qed_sa)
for _target, _key in ((8.0, "logp-8"), (-1.0, "logp-minus1")):

    def _logp(mol, target=_target):
        return math.exp(-0.5 * (Crippen.MolLogP(mol) - target) ** 2)

    register_objective(_key, lambda smi, fn=_logp: fn(_parse(smi)), _logp)

# Runtime registration may replace built-ins as well as add names. Keep the
# original immutable objects available so factories can distinguish importable
# built-ins from metadata that must travel with a worker's environment config.
BUILTIN_OBJECTIVES = MappingProxyType(dict(OBJECTIVE_REGISTRY))


class ScoringService:
    """Evaluate objectives while separating requests, computations and hits.

    Registered names or immutable ``Objective`` metadata may use an LRU cache.
    Metadata carries custom functions to a fresh worker without requiring the
    driver registry to be recreated there. Unregistered
    callables must explicitly declare purity with ``deterministic=True`` or a
    ``deterministic`` attribute before enabling it. Counters are lifetime counts,
    and clearing the cache does not reset them.
    """

    def __init__(
        self, objective="qed", input_mode="smiles", cache_size=0, deterministic=None
    ):
        if input_mode not in {"smiles", "mol"}:
            raise ValueError("scoring.input_mode must be 'smiles' or 'mol'.")
        if (
            isinstance(cache_size, bool)
            or not isinstance(cache_size, int)
            or cache_size < 0
        ):
            raise ValueError("scoring.cache_size must be a nonnegative integer.")
        self.input_mode = input_mode
        self.cache_size = cache_size
        self.is_registered = isinstance(objective, (str, Objective))
        if self.is_registered:
            self.objective = (
                get_objective(objective) if isinstance(objective, str) else objective
            )
            self.deterministic = self.objective.deterministic
            if deterministic is False:
                self.deterministic = False
            self.function = (
                self.objective.smiles_function
                if input_mode == "smiles"
                else self.objective.mol_function
            )
            if self.function is None:
                raise ValueError(
                    f"Objective {self.objective.name!r} has no Mol evaluation path."
                )
        elif callable(objective):
            self.objective = objective
            self.deterministic = bool(
                getattr(objective, "deterministic", False)
                if deterministic is None
                else deterministic
            )
            self.function = objective
        else:
            raise TypeError(
                "Objective must be a registered name, Objective metadata, or callable."
            )
        if cache_size and not self.deterministic:
            raise ValueError(
                "Scoring caches require an explicitly deterministic pure objective."
            )
        self._builtin = self.is_registered and (
            BUILTIN_OBJECTIVES.get(self.objective.name) is self.objective
        )
        self._cache = OrderedDict()
        self.requests = self.computations = self.hits = 0

    @property
    def stats(self):
        return {
            "requests": self.requests,
            "computations": self.computations,
            "hits": self.hits,
        }

    def clear_cache(self):
        self._cache.clear()

    def score(self, value):
        self.requests += 1
        if self.input_mode == "mol":
            argument = _parse(value) if isinstance(value, str) else value
            if not isinstance(argument, Chem.Mol):
                raise TypeError("Mol scoring requires an RDKit molecule or SMILES.")
        else:
            argument = (
                value
                if isinstance(value, str)
                else Chem.MolToSmiles(value, kekuleSmiles=True)
            )
        # Exact SMILES keys avoid conflating custom objectives whose input
        # representation affects their result. Mol keys preserve atom maps too.
        key = None
        if self.cache_size:
            if isinstance(argument, str):
                key = argument
            elif self._builtin:
                key = Chem.MolToSmiles(argument, kekuleSmiles=True)
            else:
                key = argument.ToBinary(Chem.PropertyPickleOptions.AllProps)
        if self.cache_size and key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.computations += 1
        result = float(self.function(argument))
        if not np.isfinite(result):
            raise ValueError("SCORE_FUNCTION must return finite scores.")
        if self.cache_size:
            self._cache[key] = result
            self._cache.move_to_end(key)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def score_many(self, values):
        return np.asarray([self.score(value) for value in values], dtype=float)

    __call__ = score
