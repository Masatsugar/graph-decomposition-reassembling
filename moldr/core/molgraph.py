from typing import List

import networkx as nx
import numpy as np
import rdkit.Chem as Chem
from moldr.chemutils import (
    get_mol,
    get_smiles,
    mol_from_graph,
    mol_to_graph,
    sanitize,
)
from rdkit.Chem import Draw


class MolGraph:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self.graph = self._mol_to_graph(self.mol)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.smiles})"

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
    graphs = [mol_to_graph(m.mol) for m in m_graphs if m.mol is not None]
    mols = [sanitize(mol_from_graph(g)) for g in graphs]
    smiles = np.unique([get_smiles(m) for m in mols])
    mols = [get_mol(s) for s in smiles]
    return mols
