"""Frozen pre-P1 reference; deliberately imports no MOLDR chemistry helpers.

The dense Python scan and string map assignment are independent of the
production optimized implementation. Keep this oracle unchanged when changing
candidate generation, so equality checks cannot pass through shared helpers.
"""

import numpy as np
from rdkit import Chem


def parse(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.Kekulize(mol)
    return mol


def smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def mapped(mol):
    for atom in mol.GetAtoms():
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
    return mol


def validate(mol):
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("invalid input")
    if any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    ) or any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        or bond.GetBondDir() != Chem.BondDir.NONE
        for bond in mol.GetBonds()
    ):
        raise ValueError("Stereochemical input")


def rebuild(base, attach, targets, direction):
    combined = Chem.CombineMols(base, attach)
    atoms = list(combined.GetAtoms())
    adjacency = np.zeros((len(atoms), len(atoms)))
    for bond in combined.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i, j] = adjacency[j, i] = bond.GetBondTypeAsDouble()
    if len(targets) == 2:
        a, b = targets
        row = adjacency[a, :] + adjacency[b, :]
        repeated = np.vstack([row, row])
        adjacency[[a, b], :] = repeated
        adjacency[:, [a, b]] = repeated.T
        removed = [a if direction == "target1" else b]
    else:
        a, b, c, d = targets
        rows = np.vstack(
            [adjacency[a, :] + adjacency[c, :], adjacency[b, :] + adjacency[d, :]]
        )
        repeated = np.vstack([rows, rows])
        adjacency[[a, b, c, d], :] = repeated
        adjacency[:, [a, b, c, d]] = repeated.T
        removed = [a, b] if direction == "target1" else [c, d]
    adjacency = np.delete(np.delete(adjacency, removed, axis=0), removed, axis=1)
    atoms = [atom for i, atom in enumerate(atoms) if i not in removed]
    result = Chem.RWMol()
    for atom in atoms:
        result.AddAtom(Chem.Atom(atom))
    orders = {
        1.0: Chem.BondType.SINGLE,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        1.5: Chem.BondType.AROMATIC,
    }
    for i, row in enumerate(adjacency):
        for j, value in enumerate(row):
            if j > i and value in orders:
                result.AddBond(i, j, orders[value])
    return mapped(result.GetMol())


def implicit(atom):
    return atom.GetValence(Chem.rdchem.ValenceType.IMPLICIT)


def explicit(atom):
    return atom.GetValence(Chem.rdchem.ValenceType.EXPLICIT)


def candidates(base, attach):
    validate(base)
    validate(attach)
    base, attach = mapped(Chem.Mol(base)), mapped(Chem.Mol(attach))
    node, edge = [], []
    offset = base.GetNumAtoms()
    for a in base.GetAtoms():
        for b in attach.GetAtoms():
            if implicit(a) == 0 or implicit(b) == 0:
                continue
            if explicit(a) + explicit(b) > max(
                a.GetTotalValence(), b.GetTotalValence()
            ):
                continue
            directions = (
                ["target1", "target2"] if a.GetSymbol() != b.GetSymbol() else [""]
            )
            for direction in directions:
                node.append(
                    smiles(
                        rebuild(
                            base, attach, (a.GetIdx(), offset + b.GetIdx()), direction
                        )
                    )
                )
    ring_edges = lambda mol: sorted(
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in mol.GetBonds()
        if bond.IsInRing()
        and implicit(bond.GetBeginAtom()) != 0
        and implicit(bond.GetEndAtom()) != 0
    )
    for source in ring_edges(base):
        for target in ring_edges(attach):
            source_labels = {base.GetAtomWithIdx(i).GetSymbol() for i in source}
            target_labels = {attach.GetAtomWithIdx(i).GetSymbol() for i in target}
            directions = (
                ["target1", "target2"] if source_labels != target_labels else [""]
            )
            for direction in directions:
                mol = rebuild(
                    base, attach, (*source, *(i + offset for i in target)), direction
                )
                try:
                    mol = parse(smiles(mol))
                except Exception:  # noqa: BLE001, S110 -- frozen sanitize fallback
                    pass  # frozen sanitize fallback
                if mol is not None:
                    edge.append(smiles(mol))
    raw = sorted(set(node)) + sorted(set(edge))
    final = {}
    for smi in raw:
        mol = parse(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        try:
            key = smiles(mol)
            parsed = parse(key)
        except (ValueError, RuntimeError):
            continue
        if parsed is not None:
            final[key] = parsed
    return (
        sorted(set(node)),
        sorted(set(edge)),
        [smiles(final[key]) for key in sorted(final)],
    )
