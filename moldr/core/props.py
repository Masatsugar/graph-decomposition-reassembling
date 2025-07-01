from collections import defaultdict

import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import QED, Descriptors

from moldr.chemutils import get_mol, sanitize
from moldr.core.reassemble import EdgeMerge, NodeMerge, merge_edge, merge_node
from moldr.sascore import reward_penalized_log_p


def scoring_function_plogp(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    score = [reward_penalized_log_p(mol) for mol in mols]
    return score


def scoring_function_logp(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    score = [Descriptors.MolLogP(mol) for mol in mols]
    return score


def scoring_function_qed(smiles):
    mols = [get_mol(smi) for smi in smiles]
    score = [Chem.QED.qed(mol) for mol in mols]
    return score


# mol_with_atom_index(node)
class MolPropSearch:
    def __init__(self):
        self.li_smiles = []
        self.mols = []
        self.tree = defaultdict(list)
        self.scores = defaultdict(list)
        self.bestLogP = 1.0

    def set_smiles(self, smiles):
        nid = len(self.li_smiles)
        self.li_smiles.append(smiles)
        mol = self.mol_with_atom_index(sanitize(get_mol(smiles)), nid)
        self.mols.append(mol)

    def mol_with_atom_index(self, mol, nid, cliques=None):
        if cliques is None:
            cliques = [mol.GetAtomWithIdx(i).GetIdx() for i in range(mol.GetNumAtoms())]

        for idx, clique in enumerate(cliques):
            amap = mol.GetAtoms()[idx].GetSymbol() + str(nid) + ":" + str(clique)
            mol.GetAtomWithIdx(idx).SetProp("amap", amap)
        return mol

    def action(self, nodes, depth):
        smis = []
        for node in nodes:
            node_mol = self.mol_with_atom_index(node, depth)
            attach_mol = self.mol_with_atom_index(self.mols[depth + 1], depth + 1)

            # node
            node_target = NodeMerge(node_mol, attach_mol)
            for s in node_target.keys:
                smis.extend(merge_node(node_target, s))
                # smis.extend(ray.get(merge_node.remote(node_target, s)))

            # edge
            edge_target = EdgeMerge(node_mol, attach_mol)
            for s in edge_target.keys:
                smis.extend(merge_edge(edge_target, s))
                # smis.extend(ray.get(merge_edge.remote(edge_target, s)))

        return smis

    def calculate_score(self, cand_mols, objective="logP"):
        # greedy search (+ beam search strategy)
        if objective == "logP":
            scores = np.array([Descriptors.MolLogP(mol) for mol in cand_mols])
            return scores
        elif objective == "QED":
            scores = np.array([Chem.QED.qed(mol) for mol in cand_mols])
            return scores
        elif objective == "PlogP":
            scores = np.array([reward_penalized_log_p(mol) for mol in cand_mols])
            return scores
        else:
            # CUSTOM PROPERTY OPTIMIZATION:
            print("other property optimization")
            logP = np.array([Descriptors.MolLogP(mol) for mol in cand_mols])
            qed = np.array([Chem.QED.qed(mol) for mol in cand_mols])
            if len(logP) == 0:
                self.bestLogP = 1.0
                logP = 0
            else:
                maxLogP = max(logP)
                if maxLogP > self.bestLogP:
                    self.bestLogP = maxLogP

            logP = logP / self.bestLogP
            scores = qed * logP
            return scores

    def greedy_search(self, nodes=[], depth=0, top=10, objective="logP"):
        if depth == len(self.mols) - 1:
            return

        if not nodes:
            nodes = [self.mols[0]]

        smis = self.action(nodes, depth)
        mols = [sanitize(get_mol(uni)) for uni in np.unique(smis)]
        cand_mols = [mol for mol in mols if mol is not None]
        scores = self.calculate_score(cand_mols, objective)
        max_ids = np.argsort(scores)
        cands = [cand_mols[i] for i in max_ids[-top:]]
        self.scores[depth] = [scores[i] for i in max_ids[-top:]]
        nodes = [
            self.mol_with_atom_index(res, depth) for res in cands if res is not None
        ]
        self.tree[depth + 1].append(nodes)
        # print(f'tree depth: {depth+1}, #candidates={len(nodes)}')
        self.greedy_search(nodes, depth + 1, top)

    # === TEST ====
    def run(self):
        pass

    def _calc_score_test(self, cand_mols, depth=0, top=1000):
        logP = [Descriptors.MolLogP(mol) for mol in cand_mols]
        qed = [Chem.QED.qed(mol) for mol in cand_mols]
        scores = qed * logP
        max_ids = np.argsort(scores)
        cands = [cand_mols[i] for i in max_ids[-top:]]
        self.scores[depth] = [scores[i] for i in max_ids[-top:]]
        nodes = [
            self.mol_with_atom_index(res, depth) for res in cands if res is not None
        ]
        self.tree[depth + 1].append(nodes)

    def test_combine(self, nodes=[], depth=0):
        if depth == len(self.mols) - 1:
            return

        if not nodes:
            nodes = [self.mols[0]]

        smis = self.action(nodes, depth)
        mols = [sanitize(get_mol(uni)) for uni in np.unique(smis)]
        candidate_mols = [mol for mol in mols if mol is not None]
        nodes = [
            self.mol_with_atom_index(res, depth)
            for res in candidate_mols
            if res is not None
        ]
        self.tree[depth + 1].append(nodes)
        # print(f'tree depth: {depth}, #candidates={len(nodes)}')
        self.test_combine(nodes, depth + 1)
