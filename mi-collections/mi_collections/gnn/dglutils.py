from functools import partial

import dgl
import dgllife
import numpy as np
import rdkit.Chem as Chem
import torch
from dgl.data import CoraGraphDataset
from dgllife.utils import (
    BaseAtomFeaturizer,
    BaseBondFeaturizer,
    atom_type_one_hot,
    bond_type_one_hot,
    smiles_to_bigraph,
)
from mi_collections.chemutils import (
    _set_edge_label,
    _set_node_label,
    get_atomic_num,
    get_bond_type,
    mol_to_graph,
)


def smiles2graph(smiles):
    # mol = Chem.MolFromSmiles(smiles)
    atom_featurizer = BaseAtomFeaturizer(
        {
            "n_feat": partial(
                atom_type_one_hot,
                allowable_set=["C", "N", "O", "F", "Si", "P", "S", "Cl" "Br", "I"],
                encode_unknown=True,
            )
        }
    )
    bond_featurizer = BaseBondFeaturizer({"e_feat": bond_type_one_hot})
    # bond_featurizer = BaseBondFeaturizer(
    #     {"e_feat": ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring])}
    # )
    g = smiles_to_bigraph(
        smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer
    )
    return g


def dgl2mol():
    pass


# def smiles2graph(smiles, node_attrs=None, edge_attrs=None, **kwargs):
#     mol = Chem.MolFromSmiles(smiles)
#     if node_attrs is None:
#         atom_num = get_atomic_num(mol)
#         node_attrs = [np.array([n]) for n in atom_num]
#
#     if edge_attrs is None:
#         bond_type = get_bond_type(mol)
#         edge_attrs = []
#         for src, dst, e in bond_type:
#             edge_attrs.append((src, dst, np.array([e])))
#             edge_attrs.append((dst, src, np.array([e])))
#
#     nx_graph = mol_to_graph(mol)
#     _set_node_label(nx_graph, node_attrs, label='n_feat')
#     return dgl.from_networkx(nx_graph=nx_graph, node_attrs=node_attrs, edge_attrs=edge_attrs, **kwargs)


class SmilesDataset:
    def __init__(self):
        # Initialize Dataset and preprocess data
        self.smiles = []
        self.graphs = []
        self.labels = []

    def __getitem__(self, index):
        # Return the corresponding DGLGraph/label needed for training/evaluation based on index
        return self.smiles[index], self.graphs[index], self.labels[index]

    def __len__(self):
        return len(self.graphs)

    def add(self, smiles, label):
        self._smiles2graph(smiles)
        self.labels.append(torch.Tensor([label]))

    def _smiles2graph(self, smiles):
        self.smiles.append(smiles)
        self.graphs.append(smiles2graph(smiles))


def load_cora_data():
    dataset = CoraGraphDataset()
    g = dataset[0]
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    return g, features, labels, train_mask, test_mask
