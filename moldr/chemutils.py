import networkx as nx
import numpy as np

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from networkx.classes.graph import Graph
from numpy import matrix, ndarray

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import Mol, KekulizeException


LABEL = "label"


def get_mol(smiles: str) -> Optional[Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def get_smiles(mol: Mol) -> str:
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        print(f"[sanitize] warning, fallback to original mol: {e}")
        return mol
    return mol


def force_kekulize(mol: Mol) -> Mol:
    """
    Force kekulization of a molecule, even if it contains aromatic bonds.
    This is useful for ensuring that the molecule is in a specific format.
    """
    try:
        graph = mol_to_graph(mol)
        mol = mol_from_graph(graph)
        return mol
    except KekulizeException as e:
        print(f"[force_kekulize] warning, fallback to original mol: {e}")
        return mol
    except Exception as e:
        print(f"[force_kekulize] unexpected error: {e}")
    return mol


def mol_with_atom_index(mol: Mol) -> Mol:
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp(
            "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
        )
    return mol


def get_atomic_num(mol) -> List[int]:
    return [atom.GetAtomicNum() for atom in mol.GetAtoms()]


def get_bond_type(mol) -> List[Tuple[int, int, float]]:
    return [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
        for bond in mol.GetBonds()
    ]


def mol_to_graph(mol) -> Graph:
    """

    :param mol:
    :return: Graph
    """
    # _mol_with_atom_index(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_label = [atom.GetSymbol() for atom in atoms]
    edge_label = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
        for bond in bonds
    ]

    # add bond information
    # adj = Chem.GetAdjacencyMatrix(mol)
    adj = np.zeros((len(atoms), len(atoms)))
    for src, dst, bond in edge_label:
        adj[src, dst] = bond
        adj[dst, src] = bond

    graph = nx.from_numpy_array(adj)
    _set_node_label(graph, node_label)
    _set_edge_label(graph, edge_label)
    return graph


def mol_from_graph(graph: Graph) -> Optional[Mol]:
    """

    :param graph:
    :return: mol
    """

    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i, atom in graph.nodes.items():
        a = Chem.Atom(atom[LABEL])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    n_atom = len(graph.nodes)
    # adjacency_matrix = nx.adjacency_matrix(graph).toarray()
    adjacency_matrix = np.zeros((n_atom, n_atom))
    for (src, dst), bond in graph.edges.items():
        adjacency_matrix[src][dst] = bond[LABEL]
        adjacency_matrix[dst][src] = bond[LABEL]

    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            if bond == 0.0:
                continue

            elif bond == 1.0:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

            elif bond == 2.0:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

            elif bond == 3.0:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

            elif bond == 1.5:
                bond_type = Chem.rdchem.BondType.AROMATIC
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    return mol.GetMol()


def _list_to_dict(labels, dtype=None):
    label_dict = defaultdict(dtype)
    for i, label in enumerate(labels):
        label_dict[i] = label
    return label_dict


def _set_node_label(graph: Graph, node_label: List[str], label=LABEL) -> None:
    for k, v in enumerate(node_label):
        graph.nodes[k][label] = v


def _set_edge_label(
    graph: Graph, edge_label: List[Tuple[int, int, float]], label=LABEL
) -> None:
    for edge in edge_label:
        src, dst, weight = edge
        graph.edges[(src, dst)][label] = weight


def mutag_convert(mutag_dataset, is_aromatic=False) -> Graph:
    """Convert Grakel Mutag Dataset to Networkx Graphs
         {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}

    :param
        mutag_dataset: MUTAG Dataset from Grakel
        is_aromatic: if true, aromatic label in edge labels can be regarded as single bond type.
    :return: networkx.Graph


    """
    # mutag_node_labels = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
    mutag_node_labels = {i: i for i in range(7)}
    v = 1.0 if is_aromatic else 1.5
    mutag_edge_labels = {
        0: v,
        1: 1.0,
        2: 2.0,
        3: 3.0,
    }  # Aromatic, single, double, triple

    edges, node_label, edge_label = mutag_dataset
    min_num = min(min(edges))
    edges = [(src - min_num, dst - min_num) for src, dst in edges]
    node_label = [mutag_node_labels[n] for n in node_label.values()]
    edge_label = [
        (src - min_num, dst - min_num, mutag_edge_labels[v])
        for (src, dst), v in edge_label.items()
    ]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    _set_node_label(graph, node_label)
    _set_edge_label(graph, edge_label)
    return graph


def summary_molgraph(mols: Mol) -> Dict[str, Union[matrix, List[str]]]:
    atom_symbol = []
    num_atoms = []
    num_bonds = []
    for mol in mols:
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        num_atoms.append(len(atoms))
        num_bonds.append(len(bonds))
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in atom_symbol:
                atom_symbol.append(symbol)

    node_edge = np.c_[num_atoms, num_bonds]
    infos = {"node_edge": node_edge, "atom_symbol": atom_symbol}
    return infos


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(""))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def get_clique_mol(mol, atoms):
    try:
        smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    except KekulizeException:
        smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=False)

    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    # We assume this is not None
    sanitized = sanitize(new_mol)
    return sanitized
