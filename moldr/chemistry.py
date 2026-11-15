"""Shared configurable candidate generation for legacy and E2E environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import ClassVar

from rdkit import Chem

from moldr.chemutils import (
    canonical_smiles,
    get_mol,
    get_smiles,
    mol_with_atom_index,
)
from moldr.core.reassemble import (
    edge_candidate_molecules,
    node_candidate_molecules,
    prepare_pair,
)


@dataclass(frozen=True)
class Candidate:
    smiles: str
    mol: Chem.Mol


class CandidateGenerator:
    """Generate sorted, validated candidates without modifying the inputs.

    By default, candidates are sanitized before serialization, transferred
    directly as RDKit molecules without atom maps, and deduplicated by canonical
    aromatic SMILES to prevent Kekule resonance duplicates and invalid fused-ring
    roundtrip drops. Non-default combinations remain configurable for ablations.
    """

    DEFAULTS: ClassVar[dict] = {
        "map_storage": "int",
        "shared_preparation": True,
        "bond_scan": "nonzero",
        "candidate_representation": "mapless",
        "dedup_key": "aromatic",
        "mol_transfer": "direct",
    }

    def __init__(self, config=None, **overrides):
        if config is None:
            values = {}
        elif is_dataclass(config):
            values = asdict(config)
        elif isinstance(config, Mapping):
            values = dict(config)
        else:
            raise TypeError("Chemistry configuration must be a mapping or dataclass.")
        values.update(overrides)
        unknown = set(values) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown chemistry keys: {sorted(unknown)}")
        self.config = {**self.DEFAULTS, **values}
        options = {
            "map_storage": {"int", "string"},
            "bond_scan": {"nonzero", "python"},
            "candidate_representation": {"mapped", "mapless"},
            "dedup_key": {"kekule", "aromatic"},
            "mol_transfer": {"smiles", "direct"},
        }
        for key, allowed in options.items():
            if self.config[key] not in allowed:
                raise ValueError(f"chemistry.{key} must be one of {sorted(allowed)}.")
        if not isinstance(self.config["shared_preparation"], bool):
            raise TypeError("chemistry.shared_preparation must be boolean.")
        if (
            self.config["mol_transfer"] == "direct"
            and self.config["candidate_representation"] != "mapless"
        ):
            raise ValueError("Direct Mol transfer requires mapless candidates.")
        self.stats = {
            "generated": 0,
            "raw_unique": 0,
            "valid": 0,
            "duplicates": 0,
            "invalid": 0,
        }

    @staticmethod
    def _remove_maps(mol):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return mol

    def generate(self, base_mol, fragment_mol):
        cfg = self.config
        prepared = (
            prepare_pair(base_mol, fragment_mol) if cfg["shared_preparation"] else None
        )
        kwargs = {
            "_prepared": prepared,
            "shared_preparation": cfg["shared_preparation"],
            "bond_scan": cfg["bond_scan"],
        }
        direct = cfg["mol_transfer"] == "direct"
        stats = {key: 0 for key in self.stats}

        if (
            cfg["candidate_representation"] == "mapless"
            and cfg["dedup_key"] == "aromatic"
        ):
            by_key = {}
            for _, enumerate_molecules in (
                ("node", node_candidate_molecules),
                ("edge", edge_candidate_molecules),
            ):
                kind_keys = set()
                for mol in enumerate_molecules(base_mol, fragment_mol, **kwargs):
                    stats["generated"] += 1
                    if mol is None:
                        stats["invalid"] += 1
                        continue
                    self._remove_maps(mol)
                    try:
                        Chem.SanitizeMol(mol)
                        for bond in mol.GetBonds():
                            if bond.GetIsAromatic():
                                bond.SetBondType(Chem.BondType.AROMATIC)
                        key = Chem.MolToSmiles(mol)
                    except (RuntimeError, ValueError):
                        stats["invalid"] += 1
                        continue
                    kind_keys.add(key)
                    if key not in by_key:
                        try:
                            if direct:
                                Chem.Kekulize(mol)
                            else:
                                mol = get_mol(key)
                                if mol is None:
                                    stats["invalid"] += 1
                                    continue
                        except (RuntimeError, ValueError):
                            stats["invalid"] += 1
                            continue
                        by_key[key] = Candidate(key, mol)
                stats["raw_unique"] += len(kind_keys)
            candidates = [by_key[key] for key in sorted(by_key)]
            stats["valid"] = len(candidates)
            stats["duplicates"] = max(
                0, stats["generated"] - stats["invalid"] - stats["valid"]
            )
            self.stats = stats
            return candidates

        raw = []
        for _, enumerate_molecules in (
            ("node", node_candidate_molecules),
            ("edge", edge_candidate_molecules),
        ):
            by_smiles = {}
            for mol in enumerate_molecules(base_mol, fragment_mol, **kwargs):
                stats["generated"] += 1
                if mol is None:
                    stats["invalid"] += 1
                    continue
                try:
                    Chem.SanitizeMol(mol)
                    Chem.Kekulize(mol)
                    if cfg["candidate_representation"] == "mapped":
                        mol_with_atom_index(mol, cfg["map_storage"])
                    else:
                        self._remove_maps(mol)
                    smi = get_smiles(mol)
                except (RuntimeError, ValueError):
                    stats["invalid"] += 1
                    continue
                by_smiles[smi] = mol
            raw.extend((smi, by_smiles[smi]) for smi in sorted(by_smiles))
        stats["raw_unique"] = len(raw)
        by_key = {}
        for raw_smiles, raw_mol in raw:
            try:
                mol = Chem.Mol(raw_mol) if direct else get_mol(raw_smiles)
                if mol is None:
                    stats["invalid"] += 1
                    continue
                self._remove_maps(mol)
                if cfg["dedup_key"] == "aromatic":
                    key = canonical_smiles(mol)
                    if not direct:
                        mol = get_mol(key)
                        if mol is None:
                            stats["invalid"] += 1
                            continue
                    if key not in by_key:
                        by_key[key] = Candidate(key, mol)
                else:
                    smi = get_smiles(mol)
                    if not direct:
                        mol = get_mol(smi)
                        if mol is None:
                            stats["invalid"] += 1
                            continue
                    by_key[smi] = Candidate(smi, mol)
            except (RuntimeError, ValueError):
                stats["invalid"] += 1
                continue
        candidates = [by_key[key] for key in sorted(by_key)]
        stats["valid"] = len(candidates)
        stats["duplicates"] = max(
            0, stats["generated"] - stats["invalid"] - stats["valid"]
        )
        self.stats = stats
        return candidates


def reassemble_candidates(base_mol, fragment_mol, config=None):
    """Convenience entry point returning validated RDKit molecules."""
    return [
        candidate.mol
        for candidate in CandidateGenerator(config).generate(base_mol, fragment_mol)
    ]
