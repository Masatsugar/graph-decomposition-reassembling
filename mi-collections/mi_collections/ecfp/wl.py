import copy

import numpy as np
import rdkit.Chem as Chem
from mi_collections.chemutils import mol_to_graph
from tqdm import tqdm


def convert_edge_format(e):
    new_e = {}
    for src, dst, val in e.values():
        if dst > src:
            new_e.update({(src, dst): val})
        else:
            new_e.update({(dst, src): val})
    return new_e


class MolWeisfeilerLehman:
    def __init__(self, n_iter=2, node_features=None, is_bond=False):
        self.n_iter = n_iter
        self.is_bond = is_bond
        self.identifier = {}
        self._node_features = (
            node_features if node_features is not None else self.__node_features
        )
        self.K = None

    def fit_transform(self, mols):
        self.fit(mols)
        return self.K

    def fit(self, mols):
        for idx, mol in tqdm(enumerate(mols)):
            identifier = self.calculate(mol)
            self.identifier.update({idx: identifier})

    def calculate(self, mol):
        identifier = {}
        adj = Chem.GetAdjacencyMatrix(mol)
        g, node, edge = mol_to_graph(mol)
        e = convert_edge_format(edge)
        for h in range(self.n_iter + 1):
            if h == 0:
                n_feat = {k: v for k, v in node.items()}
                identifier.update({h: n_feat})
            else:
                vs = {}
                n_feat = identifier[h - 1]
                for atom_id in range(len(mol.GetAtoms())):
                    nei_id = np.nonzero(adj[atom_id])[0]
                    v = n_feat[atom_id] + ",["
                    for i in nei_id:
                        if self.is_bond:
                            edge = (i, atom_id) if atom_id > i else (atom_id, i)
                            bond_type = e[edge]
                            v = self._bond_info(v, bond_type)
                        v += n_feat[i]
                    v += "]"
                    vs.update({atom_id: v})
                identifier.update({h: vs})
        return identifier

    @staticmethod
    def __node_features(self, mol):
        return [atom.GetSymbol() for atom in mol.GetAtoms()]

    @staticmethod
    def _bond_info(self, v, bond_type):
        if bond_type == 1.0:
            v += ""
        elif bond_type == 2.0:
            v += "="
        elif bond_type == 3.0:
            v += "#"
        return v


class WWL:
    def __init__(self, n_iter):
        self.n_iter = n_iter

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        pass

    def transform(self, X):
        pass
