import copy
from collections import defaultdict

import numpy as np
import networkx as nx

import rdkit.Chem as Chem
from rdkit.Chem import EState, Descriptors

from libs.chemutils import tree_decomp, sanitize
from libs.chemutils import mol_to_graph, mol_from_graph, get_mol, get_smiles


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


def adj_mat_to_list(adj_matrix):
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
    node_dict = defaultdict()
    for i, val in enumerate(nodes.values()):
        node_dict[i] = val.split(":")[0]

    edge_dict = list_to_dict(adj_mat_to_list(adj_mat))
    mol = mol_from_graph(node_dict, edge_dict)
    return mol


def merge_mol_from_node(mol1, mol2, id1, id2, direction=''):
    mol = Chem.CombineMols(mol1, mol2)
    g, nodes, edges = mol_to_graph(mol)
    new_adj = copy.deepcopy(nx.adj_matrix(g).todense())
    # num_nodes = len(mol1.GetAtoms())

    # 結合対象とするノード同士を足し合わせる
    tmp = new_adj[id1, :] + new_adj[id2, :]
    rep_vars = np.vstack([tmp, tmp])
    new_adj[[id1, id2], :] = rep_vars
    new_adj[:, [id1, id2]] = rep_vars.T

    if direction == 'target1':
        new_adj = np.delete(np.delete(new_adj, id1, axis=0), id1, axis=1)
        del nodes[id1]
    else:
        new_adj = np.delete(np.delete(new_adj, id2, axis=0), id2, axis=1)
        del nodes[id2]

    mol = _reconstruct(adj_mat=new_adj, nodes=nodes)
    if mol is None:
        return None

    return mol


def merge_mol_from_edge(mol1, mol2, edge_type1: tuple, edge_type2: tuple, direction=''):
    mol = Chem.CombineMols(mol1, mol2)
    g, nodes, edges = mol_to_graph(mol)
    new_adj = copy.deepcopy(nx.adj_matrix(g).todense())
    num_nodes = len(mol1.GetAtoms())
    # num_nodes = list(nx.connected_components(g))[0]
    id11, id12 = edge_type1
    id21, id22 = edge_type2

    # 結合対象とするノード同士を足し合わせる
    tmp = np.vstack([new_adj[id11, :] + new_adj[id21, :], new_adj[id12, :] + new_adj[id22, :]])
    rep_vars = np.vstack([tmp, tmp])
    new_adj[[id11, id12, id21, id22], :] = rep_vars
    new_adj[:, [id11, id12, id21, id22]] = rep_vars.T

    if direction == 'target1':
        new_adj = np.delete(np.delete(new_adj, [id11, id12], axis=0), [id11, id12], axis=1)
        del nodes[id11], nodes[id12]
    else:
        new_adj = np.delete(np.delete(new_adj, [id21, id22], axis=0), [id21, id22], axis=1)
        del nodes[id21], nodes[id22]

    mol = _reconstruct(adj_mat=new_adj, nodes=nodes)
    if mol is None:
        return None

    return mol


def merge_mol_from_edge_all(mol, edge_type1: tuple, edge_type2: tuple, direction=''):
    g, nodes, edges = mol_to_graph(mol)
    new_adj = copy.deepcopy(nx.adj_matrix(g).todense())
    # num_nodes = len(mol1.GetAtoms())
    num_nodes = list(nx.connected_components(g))[0]
    id11, id12 = edge_type1
    id21, id22 = edge_type2

    # 結合対象とするノード同士を足し合わせる
    tmp = np.vstack([new_adj[id11, :] + new_adj[id21, :], new_adj[id12, :] + new_adj[id22, :]])
    rep_vars = np.vstack([tmp, tmp])
    new_adj[[id11, id12, id21, id22], :] = rep_vars
    new_adj[:, [id11, id12, id21, id22]] = rep_vars.T

    if direction == 'target1':
        new_adj = np.delete(np.delete(new_adj, [id11, id12], axis=0), [id11, id12], axis=1)
        del nodes[id11], nodes[id12]
    else:
        new_adj = np.delete(np.delete(new_adj, [id21, id22], axis=0), [id21, id22], axis=1)
        del nodes[id21], nodes[id22]

    mol = _reconstruct(adj_mat=new_adj, nodes=nodes)
    if mol is None:
        print("None")

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

                # 最初の選び方で結合位置は変わる
                if nAtoms > 2:
                    cliques[v] = [cliques[nid][counter - 1]] + [max_num + i for i in range(1, nAtoms)]
                else:
                    cliques[v] = [max(cliques[nid])] + [max_num + i for i in range(1, nAtoms)]

                if max_num < max(cliques[v]):
                    max_num = max(cliques[v])

                counter += 1
                used[v] = False

    return cliques


def check_valance(atom1, atom2):
    iv1 = atom1.GetImplicitValence()
    iv2 = atom2.GetImplicitValence()
    ev1 = atom1.GetExplicitValence()
    ev2 = atom2.GetExplicitValence()
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
    return True if Descriptors.ExactMolWt(mol1) == Descriptors.ExactMolWt(mol2) else False


def reorder_target_obj(target: list):
    new_target = []
    if target[0] > target[1]:
        new_target.append((target[1], target[0]))
    else:
        return target

    return new_target[0]


def get_ring_targets(attach_mol, depth=0) -> list:
    num = str(depth)
    cliques, _ = tree_decomp(attach_mol)
    rings = [cli for cli in cliques if len(cli) > 2]
    targets = []
    for idx, ring in enumerate(rings):
        # mol = get_clique_mol(attach_mol, ring)
        # atom_types = [attach_mol.GetAtomWithIdx(i).GetSymbol() for i in ring]
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
        atom1 = attach_mol.GetAtomWithIdx(src)
        atom2 = attach_mol.GetAtomWithIdx(dst)
        if check_edge_valance(atom1, atom2):
            edge_list.append(target_edge)

    return edge_list


class MergeBase:
    def __init__(self, node, attach_mol):
        self.node = self.mol_with_atom_index(node, nid=0)
        self.attach_mol = self.mol_with_atom_index(attach_mol, nid=1)

    def mol_with_atom_index(self, mol, nid, cliques=None):
        if cliques is None:
            cliques = [mol.GetAtomWithIdx(i).GetIdx() for i in range(mol.GetNumAtoms())]
        for idx, clique in enumerate(cliques):
            amap = mol.GetAtoms()[idx].GetSymbol() + str(nid) + ":" + str(clique)
            mol.GetAtomWithIdx(idx).SetProp('amap', amap)
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
                src_node_label = atom1.GetSymbol()
                dst_node_label = atom2.GetSymbol()
                target2 = len(self.node.GetAtoms()) + int(atom2.GetProp("amap").split(":")[-1])
                if check_valance(atom1, atom2):
                    targets[target1].append(target2)
        return targets


# @ray.remote
def merge_node(obj, src):
    smis = []
    dst_list = obj.target_list[src]
    n_atoms = len(obj.node.GetAtoms())
    for dst in dst_list:
        if obj.node.GetAtomWithIdx(src).GetSymbol() != obj.attach_mol.GetAtomWithIdx(dst - n_atoms).GetSymbol():
            for direction in ['target1', 'target2']:
                res = merge_mol_from_node(obj.node, obj.attach_mol, src, dst, direction=direction)
                res = obj.mol_with_atom_index(res, nid=0)
                smi = get_smiles(res)
                smis.append(smi)
        else:
            res = merge_mol_from_node(obj.node, obj.attach_mol, src, dst)
            res = obj.mol_with_atom_index(res, nid=0)
            smi = get_smiles(res)
            smis.append(smi)

    ms = [get_mol(s) for s in smis]
    smis = [get_smiles(m) for m in ms if m is not None]

    return smis


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


def merge_edge(self, src):
    smis = []
    dst_list = self.target_list[src]
    if Chem.GetSSSR(self.node) >= 1 and Chem.GetSSSR(self.attach_mol) >= 1:
        for dst in dst_list:
            src_node_label = set([atom.GetSymbol() for atom in map(lambda x: self.node.GetAtoms()[x], src)])
            dst_node_label = set([atom.GetSymbol() for atom in map(lambda x: self.attach_mol.GetAtoms()[x], dst)])
            dst = tuple(map(lambda s: s + len(self.node.GetAtoms()), dst))
            if src_node_label != dst_node_label:
                for direction in ['target1', 'target2']:
                    res = merge_mol_from_edge(self.node, self.attach_mol, src, dst, direction=direction)
                    res = self.mol_with_atom_index(res, nid=self.depth)
                    smi = get_smiles(res)
                    smis.append(smi)
            else:
                res = merge_mol_from_edge(self.node, self.attach_mol, src, dst)
                res = self.mol_with_atom_index(res, nid=self.depth)
                smi = get_smiles(res)
                smis.append(smi)
    else:
        return

    ms = [get_mol(s) for s in smis]
    smis = [get_smiles(m) for m in ms if m is not None]
    return smis


class ScoreMixin:
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
        r = x / n ** 2
        return r  # np.hstack([r, max(0, 1 - sum(r))])

    def _abc_index(self, mol):
        """Atom Bond Connectivity Index
        """
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
            score = np.sqrt((du + dv - 2.0) / (du * dv)) + bond.GetBondTypeAsDouble() + atomic_num
            abc_vals.update({(u, v): round(score, 5)} if u < v else {(v, u): round(score, 5)})

        return abc_vals

    def _balaban_j(self, mol):
        dist_mat = Chem.GetDistanceMatrix(mol, useBO=1, useAtomWts=0, force=1)
        adj_mat = Chem.GetAdjacencyMatrix(mol, useBO=0, emptyVal=0, force=0, prefix="NoBO")

        q = mol.GetNumBonds()
        n = mol.GetNumAtoms()
        mu = q - n + 1
        j_index = {}
        s = dist_mat.sum(axis=1)  # vertex degree
        for i in range(len(s)):
            si = s[i]
            for j in range(i, len(s)):
                if adj_mat[i, j] == 1:
                    val = round(1. / np.sqrt(si * s[j]), 5)
                    j_index.update({(i, j): val})

        return j_index

    def calculate_node_score(self, mol, methods='E-state'):
        if methods == 'E-state':
            return {i: round(a, 5) for i, a in enumerate(EState.EStateIndices(mol))}
        else:
            return [round(a, 5) for a in self._alpha_centrality(mol)]

    def calculate_edge_score(self, mol, methods='ABC'):
        if methods == 'ABC':
            return self._abc_index(mol)
        else:
            return self._balaban_j(mol)


def mol_with_amap(mol, nid=0, cliques=None):
    if cliques is None:
        cliques = [mol.GetAtomWithIdx(i).GetIdx() for i in range(mol.GetNumAtoms())]

    for idx, clique in enumerate(cliques):
        amap = mol.GetAtoms()[idx].GetSymbol() + str(nid) + ":" + str(clique)
        mol.GetAtomWithIdx(idx).SetProp('amap', amap)
    return mol

# import dgl

# class GraphBuilder(dgl.DGLGraph):
#     def __init__(self, dat, id2smiles):
#         super(GraphBuilder, self).__init__()
#         self.nodes_dict = {}
#         # gpsan data
#         gnx, vlbs, elbs = dat["graph"], dat["vlbs"], dat['elbs']
#         node_list = [key for key in vlbs.keys()]
#         edge_list = [key for key in elbs.keys()]
#
#         nodeLabel = [id2smiles[int(val)] for val in vlbs.values()]
#         nodeDict = defaultdict(str)
#         for i in range(len(nodeLabel)):
#             stri = int(i)
#             nodeDict[stri] = nodeLabel[i]
#
#         mols = [get_mol(smi) for smi in nodeLabel]
#         root = 0
#
#         self.nodeDict = nodeDict
#         # edgeからクリークの番号を作る必要がある。修正必要
#         # cliques = [[atom.GetIdx() + i for atom in mol.GetAtoms()] for i, mol in enumerate(mols)]
#         # defaultdict()で返す
#         cliques = init_clique(mols, edge_list)
#         for i, c in cliques.items():
#             cmol = mols[i]
#             csmiles = get_smiles(cmol)
#             self.nodes_dict[i] = dict(
#                 smiles=csmiles,
#                 mol=get_mol(csmiles),
#                 clique=c,
#             )
#             if min(c) == 0:
#                 root = i
#
#         self.add_nodes(len(node_list))
#
#         edges = edge_list
#         src = np.zeros((len(edges) * 2,), dtype='int')
#         dst = np.zeros((len(edges) * 2,), dtype='int')
#         for i, (_x, _y) in enumerate(edges):
#             x = 0 if _x == root else root if _x == 0 else _x
#             y = 0 if _y == root else root if _y == 0 else _y
#             src[2 * i] = x
#             dst[2 * i] = y
#             src[2 * i + 1] = y
#             dst[2 * i + 1] = x
#
#         # for src, dst in edge_list:
#         self.add_edges(src, dst)
#
#         # The clique with atom ID 0 becomes root
#         if root > 0:
#             for attr in self.nodes_dict[0]:
#                 self.nodes_dict[0][attr], self.nodes_dict[root][attr] = \
#                     self.nodes_dict[root][attr], self.nodes_dict[0][attr]
#
#         for i in self.nodes_dict:
#             self.nodes_dict[i]['nid'] = i + 1
#             if self.out_degree(i) > 1:  # Leaf node mol is not marked
#                 set_atommap(self.nodes_dict[i]['mol'],
#                             self.nodes_dict[i]['nid'])
#             self.nodes_dict[i]['is_leaf'] = (self.out_degree(i) == 1)
#
#     def _assemble_node(self, i):
#         neighbors = [
#             self.nodes_dict[j] for j in self.successors(i).numpy()
#             if self.nodes_dict[j]['mol'].GetNumAtoms() > 1
#         ]
#         neighbors = sorted(neighbors,
#                            key=lambda x: x['mol'].GetNumAtoms(),
#                            reverse=True)
#         singletons = [
#             self.nodes_dict[j] for j in self.successors(i).numpy()
#             if self.nodes_dict[j]['mol'].GetNumAtoms() == 1
#         ]
#         neighbors = singletons + neighbors
#
#         cands = enum_assemble_nx(self.nodes_dict[i], neighbors)
#
#         if len(cands) > 0:
#             self.nodes_dict[i]['cands'], self.nodes_dict[i]['cand_mols'], _ = list(zip(*cands))
#             self.nodes_dict[i]['cands'] = list(self.nodes_dict[i]['cands'])
#             self.nodes_dict[i]['cand_mols'] = list(self.nodes_dict[i]['cand_mols'])
#         else:
#             self.nodes_dict[i]['cands'] = []
#             self.nodes_dict[i]['cand_mols'] = []
#
#     def _recover_node(self, i, original_mol):
#         node = self.nodes_dict[i]
#
#         clique = []
#         clique.extend(node['clique'])
#         if not node['is_leaf']:
#             for cidx in node['clique']:
#                 original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(node['nid'])
#
#         for j in self.successors(i).numpy():
#             nei_node = self.nodes_dict[j]
#             clique.extend(nei_node['clique'])
#             if nei_node['is_leaf']:  # Leaf node, no need to mark
#                 continue
#             for cidx in nei_node['clique']:
#                 # allow singleton node override the atom mapping
#                 if cidx not in node['clique'] or len(nei_node['clique']) == 1:
#                     atom = original_mol.GetAtomWithIdx(cidx)
#                     atom.SetAtomMapNum(nei_node['nid'])
#
#         clique = list(set(clique))
#         label_mol = get_clique_mol(original_mol, clique)
#         node['label'] = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
#         node['label_mol'] = get_mol(node['label'])
#
#         for cidx in clique:
#             original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)
#
#         return node['label']
#
#     def recover(self):
#         for i in self.nodes_dict:
#             self._recover_node(i, self.mol)
#
#     def assemble(self):
#         for i in self.nodes_dict:
#             self._assemble_node(i)
#         # return mol
