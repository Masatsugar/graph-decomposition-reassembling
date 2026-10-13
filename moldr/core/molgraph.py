from typing import List

import networkx as nx
import numpy as np
import rdkit.Chem as Chem
from moldr.chemutils import (
    get_mol,
    get_smiles,
    mol_to_graph,
    sanitize,
)
from rdkit.Chem import Draw


class MolGraph:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self._graph = None
        self._graph_built = False

    def __repr__(self):
        return f"{self.__class__.__name__}({self.smiles})"

    @property
    def graph(self):
        # Built lazily: the reassembly hot path creates thousands of MolGraph
        # candidates but only reads ``.mol``. Deferring the (expensive)
        # networkx conversion until ``.graph`` is first accessed avoids that
        # wasted work while keeping the attribute available for other callers.
        if not self._graph_built:
            self._graph = self._mol_to_graph(self.mol)
            self._graph_built = True
        return self._graph

    def _mol_to_graph(self, mol):
        if mol is not None:
            return mol_to_graph(mol)
        return None

    def sanitize(self, mol):
        return sanitize(mol)


class MolToGraph:
    def __init__(self, node_funct=None, edge_funct=None):
        self._custom_nlabel_funct = node_funct
        self._custom_elabel_funct = edge_funct
        self.graphs = []
        self.mols = []

    def _node_features_funct(self, mol, method="default"):
        if method == "default":
            return self._default_node_features(mol)
        elif method == "label":
            return self._atom_label_features(mol)
        else:
            return self._custom_nlabel_funct(mol)

    def _default_node_features(self, mol):
        features = np.array(
            [
                [
                    a.IsInRing(),
                    a.GetAtomicNum(),
                    a.GetDegree(),
                    a.GetExplicitValence(),
                    a.GetImplicitValence(),
                    a.GetFormalCharge(),
                    a.GetTotalNumHs(),
                ]
                for a in mol.GetAtoms()
            ],
            dtype=np.int32,
        )
        return np.array(
            [
                "".join([str(f) for f in features[atom]])
                for atom in range(len(mol.GetAtoms()))
            ],
            dtype=str,
        )

    def _atom_label_features(self, mol):
        return [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    # {atom.GetIdx(): f for atom, f in zip(mol.GetAtoms(), feat)}

    def set_node_label(self, mols, method="default"):
        self.mols = mols
        self.graphs = [mol_to_graph(mol) for mol in mols]
        self.labels = [self._node_features_funct(mol, method) for mol in mols]
        self.gs = []
        for i, graph in enumerate(self.graphs):
            for j, node in enumerate(graph.nodes.items()):
                node[j]["label"] = self.labels[i][j]
            self.gs.append(graph)

    def draw_graph(self, idx, **kwargs):
        nx.draw(self.gs[idx], with_labels=True)


def sanitize_molgraph(m_graphs: List[MolGraph]):
    # The candidates are already molecules. Rebuilding from element labels
    # destroys charges, isotopes, explicit hydrogens and stereochemistry.
    # Only the temporary atom-map numbers used by the merge code need removal.
    by_smiles = {}
    for candidate in m_graphs:
        if candidate.mol is None:
            continue
        mol = Chem.Mol(candidate.mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        try:
            smiles = get_smiles(mol)
            parsed = get_mol(smiles)
        except (ValueError, RuntimeError):
            continue
        if parsed is not None:
            by_smiles[smiles] = parsed
    return [by_smiles[smiles] for smiles in sorted(by_smiles)]
