"""Local candidate construction preserving the adjacency builder's ordering.

Only the construction step changes. Eligibility, candidate order and later
sanitization/deduplication retain their existing semantics. Features and bond
caches belong to one prepared pair; no state is reused between input molecules.
"""

from __future__ import annotations

from collections.abc import Iterator

from rdkit import Chem

from moldr.chemutils import validate_reassembly_mol
from moldr.core.reassemble import valence

_SUPPORTED_BONDS = frozenset(
    {
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    }
)
_Bond = tuple[int, int, Chem.BondType]


class _Features:
    """Atom-indexed features copied anew for one input of a prepared pair."""

    def __init__(self, mol: Chem.Mol):
        # Ring queries can initialize or change RDKit's computed properties.
        # Keep even those changes off the caller's molecule.
        mol = Chem.Mol(mol)
        atoms = list(mol.GetAtoms())
        self.n_atoms = len(atoms)
        self.symbols = [atom.GetSymbol() for atom in atoms]
        self.implicit_valences = [valence(atom, explicit=False) for atom in atoms]
        self.explicit_valences = [valence(atom, explicit=True) for atom in atoms]
        self.total_valences = [atom.GetTotalValence() for atom in atoms]
        self.ring_targets = sorted(
            (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            )
            for bond in mol.GetBonds()
            if bond.IsInRing()
            and self.implicit_valences[bond.GetBeginAtomIdx()] != 0
            and self.implicit_valences[bond.GetEndAtomIdx()] != 0
        )


class LocalPair:
    """One validated pair with mapless raw node/edge candidate iterators.

    Construct through ``prepare_local_pair`` so unsupported bond types use the
    original builder. Each candidate has the same atom and bond insertion order
    as an upper-triangle adjacency scan, including the resulting Kekule choice.
    """

    def __init__(self, base_mol: Chem.Mol, fragment_mol: Chem.Mol):
        self._base = _Features(base_mol)
        self._fragment = _Features(fragment_mol)
        combined = Chem.CombineMols(base_mol, fragment_mol)
        self._atoms_only = Chem.RWMol()
        for atom in combined.GetAtoms():
            self._atoms_only.AddAtom(atom)
        for atom in self._atoms_only.GetAtoms():
            atom.SetAtomMapNum(0)

        self._neighbors = [[] for _ in range(combined.GetNumAtoms())]
        self._bonds: list[_Bond] = []
        for bond in combined.GetBonds():
            i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            bond_type = bond.GetBondType()
            self._bonds.append((i, j, bond_type))
            self._neighbors[i].append((j, bond_type))
            self._neighbors[j].append((i, bond_type))
        self._bonds.sort()
        self._remaining_bonds: dict[frozenset[int], list[_Bond]] = {}

    def _build(self, moved: list[_Bond], dropped: frozenset[int]) -> Chem.Mol:
        remaining = self._remaining_bonds.get(dropped)
        if remaining is None:
            remaining = [
                bond
                for bond in self._bonds
                if bond[0] not in dropped and bond[1] not in dropped
            ]
            self._remaining_bonds[dropped] = remaining
        # Atom deletion preserves the relative order of the surviving indices.
        # Sorting before deletion therefore matches the old adjacency scan.
        bonds = remaining + moved
        bonds.sort()
        mol = Chem.RWMol(self._atoms_only)
        add_bond = mol.AddBond
        for i, j, bond_type in bonds:
            add_bond(i, j, bond_type)
        for index in sorted(dropped, reverse=True):
            mol.RemoveAtom(index)
        return mol.GetMol()

    def node_molecules(self) -> Iterator[Chem.Mol]:
        """Yield every historical node candidate, including invalid duplicates."""
        base, fragment = self._base, self._fragment
        for src in range(base.n_atoms):
            if base.implicit_valences[src] == 0:
                continue
            for j in range(fragment.n_atoms):
                # This is exactly check_valance, not a neutral-element maximum
                # valence heuristic: charged and explicit-H inputs are retained.
                if fragment.implicit_valences[j] == 0 or (
                    base.explicit_valences[src] + fragment.explicit_valences[j]
                    > max(base.total_valences[src], fragment.total_valences[j])
                ):
                    continue
                dst = base.n_atoms + j
                pairs = (
                    ((dst, src), (src, dst))
                    if base.symbols[src] != fragment.symbols[j]
                    else ((src, dst),)
                )
                for keep, drop in pairs:
                    moved = [
                        (keep, neighbor, bond_type)
                        if keep < neighbor
                        else (neighbor, keep, bond_type)
                        for neighbor, bond_type in self._neighbors[drop]
                    ]
                    yield self._build(moved, frozenset((drop,)))

    def edge_molecules(self) -> Iterator[Chem.Mol]:
        """Yield ring-edge candidates in the historical endpoint/direction order."""
        base, fragment = self._base, self._fragment
        for a, b in base.ring_targets:
            base_labels = {base.symbols[a], base.symbols[b]}
            for c0, d0 in fragment.ring_targets:
                c, d = c0 + base.n_atoms, d0 + base.n_atoms
                pairs = (
                    (((c, d), (a, b)), ((a, b), (c, d)))
                    if base_labels != {fragment.symbols[c0], fragment.symbols[d0]}
                    else (((a, b), (c, d)),)
                )
                for (keep0, keep1), (drop0, drop1) in pairs:
                    moved = [
                        (keep0, neighbor, bond_type)
                        if keep0 < neighbor
                        else (neighbor, keep0, bond_type)
                        for neighbor, bond_type in self._neighbors[drop0]
                        if neighbor != drop1
                    ]
                    moved.extend(
                        (keep1, neighbor, bond_type)
                        if keep1 < neighbor
                        else (neighbor, keep1, bond_type)
                        for neighbor, bond_type in self._neighbors[drop1]
                        if neighbor != drop0
                    )
                    yield self._build(moved, frozenset((drop0, drop1)))


def prepare_local_pair(base_mol: Chem.Mol, fragment_mol: Chem.Mol) -> LocalPair | None:
    """Prepare the fast path, or select the original builder with ``None``.

    Unusual bond types must retain the original numeric bond-order conversion.
    For example, DATIVE has numeric order 1.0 and becomes SINGLE there; dropping
    it as an unrecognized enum would silently disconnect the molecule.
    """
    validate_reassembly_mol(base_mol)
    validate_reassembly_mol(fragment_mol)
    if any(
        bond.GetBondType() not in _SUPPORTED_BONDS
        for mol in (base_mol, fragment_mol)
        for bond in mol.GetBonds()
    ):
        return None
    return LocalPair(base_mol, fragment_mol)
