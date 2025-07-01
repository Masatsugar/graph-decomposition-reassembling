import copy
import networkx as nx
import numpy as np

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Tuple

import rdkit.Chem as Chem
from rdkit.Chem import Descriptors, EState
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.deprecation")


from moldr.chemutils import (
    _set_edge_label,
    _set_node_label,
    get_mol,
    get_smiles,
    mol_from_graph,
    mol_to_graph,
    mol_with_atom_index,
    sanitize,
)
from moldr.core.moltree import tree_decomp
from moldr.core.molgraph import MolGraph

LABEL = "label"
AMAP = "molAtomMapNumber"
# vt = Chem.rdchem.Atom.ValenceType


def list_to_dict(labels, type=None):
    label_dict = defaultdict(type)
    for i, label in enumerate(labels):
        label_dict[i] = label
    return label_dict


def key2val(label: dict):
    return dict([(v, k) for k, v in label.items()])


def val2key(label: dict):
    return dict([(k, v) for k, v in label.items()])


def pop_bond_type(e_dict):
    """dict[key] = (src, dst, label)からラベルを除く"""
    check_size(e_dict)
    return dict([(k, v[0:2]) for k, v in e_dict.items()])


def check_size(e_dict):
    for val in e_dict.values():
        if len(val) < 3:
            raise (ValueError("Tuple size error"))
    return True


def sum_tuple(tup, i):
    return tuple([t + i for t in tup])


def adj_mat_to_list(adj_matrix) -> List[Tuple[int, int, int]]:
    src, dst = np.nonzero(adj_matrix)
    adj_list = [(s, d, adj_matrix[(s, d)]) for (s, d) in zip(src, dst)]
    return adj_list


def adj_list_to_mat(adj_list):
    n_node = len(np.unique(adj_list))
    adj_mat = np.zeros((n_node, n_node), dtype=int)
    for tup in adj_list:
        src, dst, edge_type = tup
        adj_mat[(src, dst)] = edge_type
    return adj_mat


def _reconstruct(adj_mat, nodes):
    graph = nx.from_numpy_array(adj_mat)
    edges = adj_mat_to_list(adj_mat)
    _set_node_label(graph, nodes)
    _set_edge_label(graph, edges)
    return mol_from_graph(graph)


def num_rings(mol):
    """RDKit 2022.09 以降／以前の両方でリング数(int)を返す"""
    sssr = Chem.GetSSSR(mol)
    return sssr if isinstance(sssr, int) else len(sssr)


def merge_mol_from_node(mol1, mol2, id1, id2, direction=""):
    if mol2 is not None:
        mol = Chem.CombineMols(mol1, mol2)
    else:
        mol = mol1
    graph = mol_to_graph(mol)
    nodes = [node[LABEL] for node in graph.nodes.values()]
    new_adj = copy.deepcopy(nx.adjacency_matrix(graph).toarray())
    tmp = new_adj[id1, :] + new_adj[id2, :]
    rep_vars = np.vstack([tmp, tmp])
    new_adj[[id1, id2], :] = rep_vars
    new_adj[:, [id1, id2]] = rep_vars.T

    if direction == "target1":
        new_adj = np.delete(np.delete(new_adj, id1, axis=0), id1, axis=1)
        del nodes[id1]
    else:
        new_adj = np.delete(np.delete(new_adj, id2, axis=0), id2, axis=1)
        del nodes[id2]

    mol = _reconstruct(adj_mat=new_adj, nodes=nodes)
    if mol is None:
        return None
    return mol


def merge_mol_from_edge(mol1, mol2, edge_type1: tuple, edge_type2: tuple, direction=""):
    if mol2 is not None:
        mol = Chem.CombineMols(mol1, mol2)
    else:
        mol = mol1

    graph = mol_to_graph(mol)
    _new_adj = copy.deepcopy(nx.adjacency_matrix(graph).toarray())
    nodes = [node[LABEL] for node in graph.nodes.values()]

    id11, id12 = edge_type1
    id21, id22 = edge_type2
    tmp = np.vstack(
        [_new_adj[id11, :] + _new_adj[id21, :], _new_adj[id12, :] + _new_adj[id22, :]]
    )
    rep_vars = np.vstack([tmp, tmp])
    _new_adj[[id11, id12, id21, id22], :] = rep_vars
    _new_adj[:, [id11, id12, id21, id22]] = rep_vars.T

    if direction == "target1":
        # new_nodes = []
        new_nodes = copy.deepcopy(nodes)
        new_adj = np.delete(
            np.delete(_new_adj, [id11, id12], axis=0), [id11, id12], axis=1
        )
        # for i in range(len(nodes)):
        #     if i != id11 or i != id12:
        #         new_nodes.append(nodes[i])
        del new_nodes[id11], new_nodes[id12 - 1]
    else:
        # new_nodes = []  # copy.deepcopy(nodes)
        new_nodes = copy.deepcopy(nodes)
        new_adj = np.delete(
            np.delete(_new_adj, [id21, id22], axis=0), [id21, id22], axis=1
        )
        # for i in range(len(nodes)):
        #     if i != id21 or i != id22:
        #         new_nodes.append(nodes[i])
        #
        del new_nodes[id21], new_nodes[id22 - 1]

    mol = _reconstruct(adj_mat=new_adj, nodes=new_nodes)
    if mol is None:
        return None

    return mol


def init_clique(mols, edge_list):
    adj_list = defaultdict(list)
    for edge in edge_list:
        src, dst = edge
        adj_list[src].append(dst)
        adj_list[dst].append(src)

    used = [True] * len(adj_list)
    cliques = defaultdict()
    for nid, edges in adj_list.items():
        if nid == 0:
            clique = [atom.GetIdx() for atom in mols[nid].GetAtoms()]
            cliques[nid] = clique
            max_num = clique[-1]
            used[nid] = False

        counter = 0
        for v in edges:
            if used[v]:
                nAtoms = len(mols[v].GetAtoms())
                # print(f"node={nid}, edge={v}, clique={clique}, nAtom={nAtoms}")
                if nAtoms > 2:
                    cliques[v] = [cliques[nid][counter - 1]] + [
                        max_num + i for i in range(1, nAtoms)
                    ]
                else:
                    cliques[v] = [max(cliques[nid])] + [
                        max_num + i for i in range(1, nAtoms)
                    ]

                if max_num < max(cliques[v]):
                    max_num = max(cliques[v])

                counter += 1
                used[v] = False

    return cliques


def valence(atom, explicit: bool):
    try:
        return atom.GetValence(vt.EXPLICIT if explicit else vt.IMPLICIT)
    except (AttributeError, TypeError):
        # 2022.06 以前（bool 受け取り版）の後方互換
        return atom.GetValence(explicit)


def check_valance(atom1, atom2):
    iv1 = atom1.GetImplicitValence()
    iv2 = atom2.GetImplicitValence()
    ev1 = atom1.GetExplicitValence()
    ev2 = atom2.GetExplicitValence()
    # iv1 = valence(atom1, explicit=False)
    # iv2 = valence(atom2, explicit=False)
    # ev1 = valence(atom1, explicit=True)
    # ev2 = valence(atom2, explicit=True)

    if iv1 == 0:
        return False
    if iv2 == 0:
        return False
    if ev1 + ev2 > max(atom1.GetTotalValence(), atom2.GetTotalValence()):
        return False
    return True


def check_edge_valance(atom1, atom2):
    iv1 = atom1.GetImplicitValence()
    iv2 = atom2.GetImplicitValence()
    if iv1 == 0:
        return False
    if iv2 == 0:
        return False
    return True


def check_mol(mol1, mol2):
    return (
        True if Descriptors.ExactMolWt(mol1) == Descriptors.ExactMolWt(mol2) else False
    )


def reorder_target_obj(target):
    new_target = []
    if target[0] > target[1]:
        new_target.append((target[1], target[0]))
    else:
        return target

    return new_target[0]


def get_ring_targets(mol) -> List[Tuple]:
    cliques, _ = tree_decomp(mol)
    rings = [cli for cli in cliques if len(cli) > 2]
    targets = []
    for idx, ring in enumerate(rings):
        for i in range(len(ring)):
            if i == len(ring) - 1:
                targets.append((ring[i], ring[0]))
            else:
                targets.append((ring[i], ring[i + 1]))

    # remove duplication of target atom
    target_atom_list = []
    for lst in targets:
        if lst not in target_atom_list:
            target_atom_list.append(lst)

    target_edge_list = [reorder_target_obj(target) for target in target_atom_list]

    edge_list = []
    for target_edge in target_edge_list:
        src, dst = target_edge
        atom1 = mol.GetAtomWithIdx(src)
        atom2 = mol.GetAtomWithIdx(dst)
        if check_edge_valance(atom1, atom2):
            edge_list.append(target_edge)

    return edge_list


def check_target_edge(base_mol, attach_mol) -> Dict[int, List[int]]:
    target_edges_src = get_ring_targets(base_mol)
    target_edges_dst = get_ring_targets(attach_mol)
    target_edge = defaultdict(list)
    for i in target_edges_src:
        for j in target_edges_dst:
            target_edge[i].append(j)

    return target_edge


def check_target_atom(base_mol, attach_mol) -> Dict[int, List[int]]:
    """

    :param base_mol:
    :param attach_mol:
    :return: src to dst candidate of nodes.
    """
    mol_with_atom_index(base_mol)
    mol_with_atom_index(attach_mol)
    targets = defaultdict(list)
    for atom1 in base_mol.GetAtoms():
        target1 = int(atom1.GetProp(AMAP))
        for atom2 in attach_mol.GetAtoms():
            # src_node_label = atom1.GetSymbol()
            # dst_node_label = atom2.GetSymbol()
            target2 = len(base_mol.GetAtoms()) + int(atom2.GetProp(AMAP))
            if check_valance(atom1, atom2):
                targets[target1].append(target2)
    return targets


def merge_node(base_mol, attach_mol) -> List[MolGraph]:
    smis = []
    target_list = check_target_atom(base_mol, attach_mol)
    n_atoms = len(base_mol.GetAtoms())
    for src, dst_list in target_list.items():
        src_symbol = base_mol.GetAtomWithIdx(src).GetSymbol()
        for dst in dst_list:
            dst_symbol = attach_mol.GetAtomWithIdx(dst - n_atoms).GetSymbol()
            if src_symbol != dst_symbol:
                for direction in ["target1", "target2"]:
                    new_mol = merge_mol_from_node(
                        base_mol, attach_mol, src, dst, direction=direction
                    )
                    mol_with_atom_index(new_mol)
                    new_smiles = get_smiles(new_mol)
                    smis.append(new_smiles)
            else:
                new_mol = merge_mol_from_node(base_mol, attach_mol, src, dst)
                mol_with_atom_index(new_mol)
                new_smiles = get_smiles(new_mol)
                smis.append(new_smiles)

    smis = np.unique(smis)
    mol_graphs = [MolGraph(smi) for smi in smis]
    return mol_graphs


def merge_edge(base_mol, attach_mol):
    smis = []
    target_list = check_target_edge(base_mol, attach_mol)
    for src, dst_list in target_list.items():
        if num_rings(base_mol) >= 1 and num_rings(attach_mol) >= 1:
            for dst in dst_list:
                src_node_label = set(
                    [
                        atom.GetSymbol()
                        for atom in map(lambda x: base_mol.GetAtoms()[x], src)
                    ]
                )
                dst_node_label = set(
                    [
                        atom.GetSymbol()
                        for atom in map(lambda x: attach_mol.GetAtoms()[x], dst)
                    ]
                )
                dst = tuple(map(lambda s: s + len(base_mol.GetAtoms()), dst))
                if src_node_label != dst_node_label:
                    for direction in ["target1", "target2"]:
                        res = merge_mol_from_edge(
                            base_mol, attach_mol, src, dst, direction=direction
                        )
                        res = mol_with_atom_index(res)
                        res = sanitize(res)
                        if res is not None:
                            smi = get_smiles(res)
                            smis.append(smi)
                else:
                    res = merge_mol_from_edge(base_mol, attach_mol, src, dst)
                    res = mol_with_atom_index(res)
                    res = sanitize(res)
                    if res is not None:
                        smi = get_smiles(res)
                        smis.append(smi)
        else:
            return

    smis = np.unique(smis)
    mol_graphs = [MolGraph(smi) for smi in smis]
    return mol_graphs


class FeaturesType(Enum):
    EState = "E-state"
    ABC = "ABC"


class GraphScoreMixin:
    def __init__(self, is_kekule=True):
        self.is_kekule = is_kekule

    def _alpha_centrality(self, mol):
        g, _, _ = mol_to_graph(mol)
        # nx.eigenvector_centrality(g, nstart=dict(g.degree()), weight=1/n)
        n = len(g.nodes)

        e = np.array([val for val in dict(g.degree()).values()]) / (n - 1)
        A = nx.adjacency_matrix(g).toarray()
        alpha = 1 / n

        x = np.linalg.inv(np.eye(n) - alpha * A.T).dot(e)
        # r = np.sort(x) / n ** 2
        r = x / n**2
        return r  # np.hstack([r, max(0, 1 - sum(r))])

    def _abc_index(self, mol):
        """Atom Bond Connectivity Index"""
        bonds = mol.GetBonds()

        abc_vals = {}
        for bond in bonds:
            src = bond.GetBeginAtom()
            dst = bond.GetEndAtom()
            du = src.GetDegree()
            dv = dst.GetDegree()
            u = src.GetIdx()
            v = dst.GetIdx()

            atomic_num = (src.GetAtomicNum() + dst.GetAtomicNum()) / 6.0
            score = (
                np.sqrt((du + dv - 2.0) / (du * dv))
                + bond.GetBondTypeAsDouble()
                + atomic_num
            )
            abc_vals.update(
                {(u, v): round(score, 5)} if u < v else {(v, u): round(score, 5)}
            )

        return abc_vals

    def _balaban_j(self, mol):
        dist_mat = Chem.GetDistanceMatrix(mol, useBO=1, useAtomWts=0, force=1)
        adj_mat = Chem.GetAdjacencyMatrix(
            mol, useBO=0, emptyVal=0, force=0, prefix="NoBO"
        )

        # q = mol.GetNumBonds()
        # n = mol.GetNumAtoms()
        # mu = q - n + 1
        j_index = {}
        s = dist_mat.sum(axis=1)  # vertex degree
        for i in range(len(s)):
            si = s[i]
            for j in range(i, len(s)):
                if adj_mat[i, j] == 1:
                    val = round(1.0 / np.sqrt(si * s[j]), 5)
                    j_index.update({(i, j): val})

        return j_index

    def calculate_node_score(self, mol, methods=FeaturesType.EState):
        if methods == FeaturesType.EState:
            return {i: round(a, 5) for i, a in enumerate(EState.EStateIndices(mol))}
        else:
            return [round(a, 5) for a in self._alpha_centrality(mol)]

    def calculate_edge_score(self, mol, methods=FeaturesType.ABC):
        if methods == FeaturesType.ABC:
            return self._abc_index(mol)
        else:
            return self._balaban_j(mol)


def mol_with_amap(mol, nid=0, cliques=None):
    if cliques is None:
        cliques = [mol.GetAtomWithIdx(i).GetIdx() for i in range(mol.GetNumAtoms())]

    for idx, clique in enumerate(cliques):
        amap = mol.GetAtoms()[idx].GetSymbol() + str(nid) + ":" + str(clique)
        mol.GetAtomWithIdx(idx).SetProp("amap", amap)
    return mol


class MergeBase:
    def __init__(self, node, attach_mol):
        self.node = self.mol_with_atom_index(node, nid=0)
        self.attach_mol = self.mol_with_atom_index(attach_mol, nid=1)

    def mol_with_atom_index(self, mol, nid, cliques=None):
        if cliques is None:
            cliques = [mol.GetAtomWithIdx(i).GetIdx() for i in range(mol.GetNumAtoms())]
        for idx, clique in enumerate(cliques):
            amap = mol.GetAtoms()[idx].GetSymbol() + str(nid) + ":" + str(clique)
            mol.GetAtomWithIdx(idx).SetProp("amap", amap)
        return mol


class NodeMerge(MergeBase):
    def __init__(self, node, attach_mol):
        super(NodeMerge, self).__init__(node, attach_mol)
        self.target_list = self._calc_target_atom()
        self.keys = list(self.target_list.keys())

    def _calc_target_atom(self):
        targets = defaultdict(list)
        for atom1 in self.node.GetAtoms():
            target1 = int(atom1.GetProp("amap").split(":")[-1])
            for atom2 in self.attach_mol.GetAtoms():
                # src_node_label = atom1.GetSymbol()
                # dst_node_label = atom2.GetSymbol()
                target2 = len(self.node.GetAtoms()) + int(
                    atom2.GetProp("amap").split(":")[-1]
                )
                if check_valance(atom1, atom2):
                    targets[target1].append(target2)
        return targets


class EdgeMerge(MergeBase):
    def __init__(self, node, attach_mol):
        super(EdgeMerge, self).__init__(node, attach_mol)
        self.target_list = self._calc_target_edge()
        self.keys = list(self.target_list.keys())
        self.depth = 0

    def _calc_target_edge(self, depth=0):
        target_edges_src = get_ring_targets(self.node, depth)
        target_edges_dst = get_ring_targets(self.attach_mol, depth + 1)
        target_edge = defaultdict(list)
        for i in target_edges_src:
            for j in target_edges_dst:
                target_edge[i].append(j)

        return target_edge
