import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx
import networkx as nx
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
from mi_collections.chemutils import (
    _set_edge_label,
    _set_node_label,
    get_clique_mol,
    get_mol,
    get_smiles,
    mol_from_graph,
)
from mi_collections.moldr.moltree import MolTree
from networkx.algorithms import isomorphism
from tqdm import tqdm


@dataclass
class FLAGS:
    database_file_name: str
    min_support: int
    lower_bound_of_num_vertices: int
    upper_bound_of_num_vertices: float = float("inf")
    num_graphs: float = float("inf")
    directed: bool = True
    verbose: bool = False
    plot: bool = False
    where: bool = False


def preprocess_graphs(graphs, fname):
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True, exist_ok=True)

    tup_list = create_gspan_dataset_nx(graphs)
    save_graph(tup_list, fname)


def preprocess_mols(mols, fname: Path, method: str = " raw"):
    """

    :param mols:
    :param fname: save file name
    :param method: either "raw" or "jt". preprocessing method for gSpan.
    :return:
    """

    if not fname.parent.exists():
        fname.parent.mkdir(parents=True, exist_ok=True)

    if method == "raw":
        save_gspan_dataset(mols, sname=fname)
    elif method == "jt":
        save_jt_dataset(mols, sname=fname)
    else:
        raise ValueError("Method is either 'raw' or 'jt'. ")


def save_jt_dataset(mols, sname):
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
    pd.DataFrame.from_dict(id2smiles, orient="index", columns=["smiles"]).to_csv(
        f"{sname}vocab.csv"
    )


def save_gspan_dataset(mols, sname):
    """Save dataset for gSpan from Mols.

    :param mols:
    :param fpath:
    :return:
    """
    tup_list = create_gspan_dataset(mols)
    save_graph(tup_list, sname)


def save_graph(tuple_list, fpath: Path) -> None:
    """save gSpan data.

    :param tuple_list:
    :param fpath:
    :return:
    """
    with fpath.open(mode="w") as f:
        for var_s in tuple_list:
            for var in var_s:
                f.write(str(var))
                f.write(" ")
            f.write("\n")
        print(f"save to {fpath}")


def create_gspan_dataset_nx(
    nx_graphs: List[networkx.Graph], node_attrs: str, edge_attrs: str = "label"
) -> List[tuple]:
    """Create gSpan dataset from networkx Graph.

    Args:
        nx_graphs:
        node_attrs:
        edge_attrs:

    Returns:

    Examples:
        >>> graphs = [nx.Graph()]
        >>> create_gspan_dataset(graphs, node_attrs="attr")
        >>> [('t', '#', '0'), ('v', 0, 0)]
    """
    tuple_list = []
    for idx, graph in enumerate(tqdm(nx_graphs)):
        node_dict = {k: v[node_attrs] for k, v in graph.nodes.items()}
        node_dict = OrderedDict(sorted(node_dict.items()))
        edge_label = [
            (src, dst, graph.edges[(src, dst)][edge_attrs])
            for (src, dst), v in graph.edges.items()
        ]

        edges_list = []
        s_id = ("t", "#", str(idx))
        s_end = ("t", "#", "-1")
        tuple_list.extend([s_id])
        for k, v in node_dict.items():
            tuple_list.append(("v", k, int(v)))

        for src, dst, edge_label in edge_label:
            edges_list.append(("e", int(src), int(dst), int(edge_label)))

        tuple_list.extend(edges_list)
    tuple_list.extend([s_end])
    return tuple_list


def create_gspan_dataset(mols: List[rdkit.Chem.rdchem.Mol]) -> List[tuple]:
    """Create dataset for gSpan.

    :param mols: mol list
    :param node_labels:
    :return:
    """
    tuple_list = []
    for idx, mol in enumerate(tqdm(mols)):
        node_list = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
        bond_types = np.array(
            [
                (
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond.GetBondTypeAsDouble(),
                )
                for bond in mol.GetBonds()
            ]
        )

        adj = Chem.GetAdjacencyMatrix(mol)
        adj[np.tril_indices(adj.shape[0])] = 0
        edges_list = []
        s_id = ("t", "#", str(idx))
        s_end = ("t", "#", "-1")
        tuple_list.extend([s_id])
        for i, node_label in enumerate(node_list):
            tuple_list.append(("v", i, node_label))

        for left, right, edge_label in bond_types:
            edges_list.append(("e", int(left), int(right), str(edge_label)))

        tuple_list.extend(edges_list)
    tuple_list.extend([s_end])
    return tuple_list


def create_junction_tree_dataset(
    mols: List[rdkit.Chem.rdchem.Mol],
) -> Tuple[List[tuple], Dict[str, int]]:
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


def _convert_edge_type(
    e_dict: Dict[Tuple[int, int], str]
) -> List[Tuple[int, int, float]]:
    """convert edge type {(src, dst): edge label} into (src, dst, edge label).

    :param e_dict:
    :return: List
    """
    new_e_list = []
    for k, v in e_dict.items():
        src, dst = k
        new_e_list.append((int(src), int(dst), float(v)))
    return new_e_list


def gspan_to_mol(
    gspan, method: str = "raw", smiles_list: Optional[List[str]] = None
) -> List[rdkit.Chem.Mol]:
    """convert gSpan into molecules

    Parameters
    ----------
    gspan
    method: ("raw" or "jt")
    dataset: SMILES LIST

    Returns
    -------

    """
    if method == "raw":
        return _nm_to_mol(gspan)
    elif method == "jt":
        if smiles_list is None:
            raise ValueError("Set original SMILES LIST")
        return _jt_to_mol(gspan, smiles_list)
    else:
        raise ValueError("Method is either 'raw' or 'jt'. ")


def _nm_to_mol(gspan) -> List[rdkit.Chem.Mol]:
    """Convert gSpan (raw graphs) to molecules

    Parameters
    ----------
    gspan: object

    Returns
    -------
    rdkit molecules

    """

    new_mols = []
    for gs in tqdm(gspan.ggnx):
        g, n_dict, e_dict, _ = gs.values()
        new_e_dict = _convert_edge_type(e_dict)
        _set_node_label(g, list(n_dict.values()))
        _set_edge_label(g, list(new_e_dict))
        mol = mol_from_graph(g)
        new_mols.append(mol)
    return new_mols


def _get_node_edge(
    jt_infos,
) -> Tuple[Dict[int, Tuple[int, int, str]], Dict[str, Tuple[int, int, str]]]:
    """return node and edge dict

    Parameters
    ----------
    jt_infos

    Returns
    -------

    """

    i = 0
    g = nx.Graph()
    n_dict = defaultdict()
    e_dict = defaultdict()
    for info in jt_infos:
        for edge in info["edge"]:
            src, dst, bond = edge
            g.add_edges_from([(src, dst)], bond=bond)
            e_dict[i] = (src, dst, bond)
            i += 1

        for idx, node in zip(info["clique"], info["node"]):
            n_dict[idx] = node

    return n_dict, e_dict


def _relabel(n_dict, e_dict):
    """Relabel node label of cliques in junction tree from vertex 0."""
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
        node_label = target.nodes[i]["label"]
        if node_label not in node_labels:
            node_labels.append(node_label)
    return node_labels


def edge_label_to_set(edge_label):
    edges = []
    bond_types = []
    for e in edge_label:
        edges.append({e[0], e[1]})
        bond_types.append(e[2])
    return edges, bond_types


def get_edge_from_clique(clique, edges, bonds):
    #
    cli = set(clique)
    res = []
    for edge, bond in zip(edges, bonds):
        inter = edge.intersection(cli)
        if len(inter) > 1:
            inter = [int(i) for i in inter] + [bond]
            res.append(tuple(inter))

    return res


def get_clique_info(mt):
    """add information to Junction tree node"""
    # mol_with_atom_index(mt.mol)
    # cliques = mt.cliques
    # new_node_id = [int(atom.GetProp('molAtomMapNumber')) for atom in mt.mol.GetAtoms()]
    node_label = np.array([atom.GetSymbol() for atom in mt.mol.GetAtoms()])
    edge_label = np.array(
        [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
            for bond in mt.mol.GetBonds()
        ]
    )

    res = defaultdict(dict)
    edge_set, bond_types = edge_label_to_set(edge_label)
    for i, node in enumerate(mt.nodes):
        clique_node = list(node_label[node.clique])
        clique_edge = get_edge_from_clique(
            node.clique, edge_set, bond_types
        )  # to do; use partial
        nei_id = [n.nid for n in node.neighbors]
        smiles = node.smiles
        res[i] = dict(
            nid=node.nid,
            clique=node.clique,
            node=clique_node,
            edge=clique_edge,
            smiles=smiles,
            nei_id=nei_id,
        )

    return res


def _jt_to_mol(gspan, smiles_list: List[str]) -> List[rdkit.Chem.Mol]:
    """reconstruct gSpan objects (junction tree) to rdkit.Chem.Mol.

    Parameters
    ----------
    gspan: gSpan object
    smiles_list: Original dataset used for gSpan.

    Returns
    -------
    mols: Reconstructed molecules from junction tree minded via gSpan.

    """
    _gspan_mols = []
    N = gspan._report_df.shape[0]
    for test_id in tqdm(range(0, N)):
        original_id = int(gspan._report_df.support_ids[test_id + 1].split(",")[0])
        original, original_n, original_e = gspan.graphs[original_id].get_result()
        target, target_n, target_e, _ = gspan.ggnx[test_id].values()
        smiles = get_smiles(get_mol(smiles_list[original_id]))  # canonical SMILES
        mt = MolTree(smiles)
        clique_info = get_clique_info(mt)
        target_node = _get_target_node_label(target)
        check_node = isomorphism.categorical_node_match("label", target_node)
        gs = isomorphism.ISMAGS(original, target, node_match=check_node)
        for m in list(gs.largest_common_subgraph()):
            mapping = {v: k for k, v in m.items()}
            jt_infos = [clique_info[i] for i in mapping.values() if any(clique_info[i])]
            mined_nodes = []
            for cli in jt_infos:
                mined_nodes.extend(cli["clique"])

            mined_nodes_set = set(mined_nodes)
            mol = get_clique_mol(mt.mol, mined_nodes_set)
            _gspan_mols.append(mol)

    # sanitize molecules
    unique_mols = [
        Chem.MolFromSmiles(s)
        for s in np.unique([Chem.MolToSmiles(m) for m in _gspan_mols])
    ]
    gspan_smiles = [Chem.MolToSmiles(m) for m in unique_mols if m is not None]
    gspan_mols = [get_mol(s) for s in gspan_smiles]

    return gspan_mols
