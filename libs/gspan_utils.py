import os
import rdkit
import rdkit.Chem as Chem

import pandas as pd
import numpy as np
from typing import List

from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
from networkx.algorithms import isomorphism

from libs.chemutils import MolTree, get_mol_node_label, get_mol_edge_label
from libs.chemutils import get_mol, get_smiles, get_clique_info, mol_from_graph


@dataclass
class FLAGS:
    database_file_name: str
    min_support: int
    lower_bound_of_num_vertices: int
    upper_bound_of_num_vertices: float = float('inf')
    num_graphs: float = float('inf')
    directed: bool = True
    verbose: bool = False
    plot: bool = False
    where: bool = False


def preprocess_mols(mols, fname='gspan.data', method="raw"):
    """

    :param mols:
    :param fname: save file name
    :param method: either "raw" or "jt". preprocessing method for gSpan.
    :return:
    """
    if method == "raw":
        save_gspan_dataset(mols, sname=fname)
    elif method == "jt":
        save_jt_dataset(mols, sname=fname)
    else:
        raise ValueError("Method is either 'raw' or 'jt'. ")


def save_jt_dataset(mols, sname='gspan_jt.data'):
    """Save junction tree dataset for gSpan from Mols.

    :param mols:
    :param sname:
    :return:
    """
    tup_list, smiles2id = create_junction_tree_dataset(mols)
    id2smiles = {v: k for k, v in smiles2id.items()}
    # save gSpan data
    save_graph(tup_list, sname)

    # save vocabulary (node labels of cliques in JT) for reconstruction.
    pd.DataFrame.from_dict(id2smiles, orient='index', columns=['smiles']).to_csv(f'{sname}vocab.csv')


def save_gspan_dataset(mols, sname="gspan.data"):
    """Save dataset for gSpan from Mols.

    :param mols:
    :param fpath:
    :return:
    """
    tup_list = create_gspan_dataset(mols)
    save_graph(tup_list, sname)


def save_graph(tuple_list, fpath="gpsan.data") -> None:
    """save gSpan data.

    :param tuple_list:
    :param fpath:
    :return:
    """
    with open(fpath, mode="w") as f:
        for var_s in tuple_list:
            for var in var_s:
                f.write(str(var))
                f.write(" ")
            f.write("\n")


def create_gspan_dataset(mols: List[rdkit.Chem.rdchem.Mol]) -> List[tuple]:
    """Create dataset for gSpan.

    :param mols: mol list
    :param node_labels:
    :return:
    """
    tuple_list = []
    for idx, mol in enumerate(tqdm(mols)):
        node_list = get_mol_node_label(mol)
        bond_types = get_mol_edge_label(mol)

        adj = Chem.GetAdjacencyMatrix(mol)
        adj[np.tril_indices(adj.shape[0])] = 0
        edges_list = []

        s_id = ("t", "#", str(idx))
        s_end = ("t", "#", "-1")
        tuple_list.extend([s_id])
        for i, node_label in enumerate(node_list):
            tuple_list.append(("v", i, node_label))

        for l, r, edge_label in bond_types:
            edges_list.append(("e", int(l), int(r), int(edge_label)))

        tuple_list.extend(edges_list)
    tuple_list.extend([s_end])
    return tuple_list


def create_junction_tree_dataset(mols: List[rdkit.Chem.rdchem.Mol]) -> List[tuple]:
    """preprocess jucntion tree molecules to gSpan dataset.

    :param mols:
    :return:
    """
    tuple_list = []
    cnt = 0
    smiles2id = defaultdict()
    for idx, mol in enumerate(tqdm(mols)):
        smile = Chem.MolToSmiles(mol)
        mt = MolTree(smile)
        n_lab = defaultdict()
        for i, node in enumerate(mt.nodes):
            # node.smiles = node.smiles + str(":") + str(node.clique)
            n_lab[i] = node.smiles
            if node.smiles not in smiles2id:
                smiles2id[node.smiles] = cnt
                cnt += 1

        # TODO: ADD EDGE LABEL, IONIZATION
        edge_list = [1 for _ in range(len(mt.edges))]
        node_list = [smiles2id[x.smiles] for x in mt.nodes]

        s_id = ("t", "#", str(idx))
        s_end = ("t", "#", "-1")
        tuple_list.extend([s_id])
        for i, n_label in enumerate(node_list):
            # print("v", i, n_label)
            tuple_list.append(("v", i, n_label))

        edges_list = []
        for (l, r), e_label in zip(mt.edges, edge_list):
            # print("e", l, r, edgelabel)
            edges_list.append(("e", l, r, e_label))

        tuple_list.extend(edges_list)

    tuple_list.extend([s_end])

    return tuple_list, smiles2id


def _get_new_e_dict(e_dict) -> dict:
    new_e_dict = {}
    for i, (k, v) in enumerate(e_dict.items()):
        src, dst = k
        new_e_dict.update({i: (src, dst, v)})
    return new_e_dict


def gspan_to_mol(gspan, method='raw', dataset=None) -> List[rdkit.Chem.Mol]:
    """

    :param gspan: gSpan graph objects
    :param method: select 'raw' or 'jt'.
    :param dataset: use original dataset if method is 'jt' for reconstructing mols.
    :return:
    """
    if method == 'raw':
        return _nm_to_mol(gspan)
    elif method == 'jt':
        if dataset is None:
            raise ValueError("Set original dataset")
        return _jt_to_mol(gspan, dataset)
    else:
        raise ValueError("Method is either 'raw' or 'jt'. ")


def _nm_to_mol(gspan) -> List[rdkit.Chem.Mol]:
    """Convert gSpan raw graph to mols

    :param gspan:
    :return:
    """
    new_mols = []
    for gs in tqdm(gspan.ggnx):
        g, n_dict, e_dict, _ = gs.values()
        new_e_dict = _get_new_e_dict(e_dict)
        mol = mol_from_graph(n_dict, new_e_dict)
        new_mols.append(mol)
    return new_mols


def _get_node_edge(jt_infos):
    """return node and edge dict.

    :param jt_infos: dict value from get_clique_info(mol).
    :return:
    """
    i = 0
    g = nx.Graph()
    n_dict = defaultdict()
    e_dict = defaultdict()
    for info in jt_infos:
        for edge in info['edge']:
            src, dst, bond = edge
            g.add_edges_from([(src, dst)], bond=bond)
            e_dict[i] = (src, dst, bond)
            i += 1

        for idx, node in zip(info['clique'], info['node']):
            n_dict[idx] = node

    return n_dict, e_dict


def _relabel(n_dict, e_dict):
    """Relabel node label of cliques in junction tree from vertex 0.
    """
    new_mapping_num = {}
    new_n_dict = {}
    new_e_dict = {}
    for i, (k, v) in enumerate(n_dict.items()):
        new_mapping_num[k] = i
        new_n_dict[i] = v

    for k, v in e_dict.items():
        src, dst, bond = v
        new_e_dict[k] = (new_mapping_num[src], new_mapping_num[dst], bond)

    return new_n_dict, new_e_dict


def _get_target_node_label(target):
    node_labels = []
    for i in range(len(target.nodes)):
        node_label = target.nodes[i]['label']
        if not node_label in node_labels:
            node_labels.append(node_label)
    return node_labels


def _jt_to_mol(gspan, dataset: pd.DataFrame) -> List[rdkit.Chem.Mol]:
    """Convert gSpan JT graph objects to rdkit.Chem.Mol.

    :param gspan: gSpan graph objects.
    :param dataset: original db
    :return: reconstructed mols
    """
    # TODO: consider each position when combining nodes.
    _gspan_mols = []
    N = gspan._report_df.shape[0]
    for test_id in tqdm(range(0, N)):
        original_id = int(gspan._report_df.support_ids[test_id + 1].split(',')[0])
        original, original_n, original_e = gspan.graphs[original_id].get_result()
        target, target_n, target_e, _ = gspan.ggnx[test_id].values()
        smiles = dataset.values[original_id][0]
        mt = MolTree(smiles)
        clique_info = get_clique_info(mt)
        target_node = _get_target_node_label(target)
        check_node = isomorphism.categorical_node_match('label', target_node)
        gs = isomorphism.ISMAGS(original, target, node_match=check_node)
        # sub = list(gs.isomorphisms_iter(symmetry=True))[0]
        # sub = list(gs.largest_common_subgraph())[0]
        # mapping = {v:k for k, v in sub.items()}
        for m in list(gs.largest_common_subgraph()):
            mapping = {v: k for k, v in m.items()}
            jt_infos = [clique_info[i] for i in mapping.values() if any(clique_info[i])]
            n_dict, e_dict = _get_node_edge(jt_infos)
            new_n_dict, new_e_dict = _relabel(n_dict, e_dict)
            mol = mol_from_graph(new_n_dict, new_e_dict)
            _gspan_mols.append(mol)

    # sanitize molecules
    unique_smis = np.unique([Chem.MolToSmiles(m) for m in _gspan_mols if m is not None])
    unique_mols = [Chem.MolFromSmiles(s) for s in unique_smis]
    # Kekule form
    gspan_smiles = [get_smiles(m) for m in unique_mols if m is not None]
    gspan_mols = [get_mol(s) for s in gspan_smiles]

    return gspan_mols
