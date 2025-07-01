import random
from collections import defaultdict

import numpy as np
from mi_collections.chemutils import get_mol, get_smiles
from mi_collections.moldr.props import MolPropSearch
from rdkit.Chem import Descriptors
from tqdm import tqdm

np.random.seed(0)

# Exploaration parameter
C_PUCT = 10

# BEAM SEARCH WIDTH
N_CHOICE = 1

# Tentative: if selector is epsilon greedy, set value > 0. Not be used in the experiment.
EPSILON = 0

# Terminal condition
MAX_MW = 1000
MAX_NODES = 100

# Objective property
OBJECTIVE = "PlogP"


def selector_eg(init_smiles, target_scores, num, epsilon):
    """select the building block based on the epsilon greedy."""
    if random.random() >= epsilon:
        max_ids = np.argsort(target_scores)
        cands = [init_smiles[i] for i in max_ids[-num:]]
        return cands
    else:
        return np.random.choice(init_smiles, num)


def selector(init_smiles):
    """select a building block based on UCT.

    :param init_smiles: building_blocks SMILES
    :return:
    """
    return np.random.choice(init_smiles, 1)


def candidate_smiles(smi1, smi2, top=20, objective="PlogP"):
    """Return the reassembled molecules based on the property score.

    :param smi1: scaffold smiles
    :param smi2: building block smiles
    :param top: beam search width
    :param objective: {'PlogP', "logP", "QED"}
    :return: smiles, mol
    """
    prop = MolPropSearch()
    prop.set_smiles(smi1)
    prop.set_smiles(smi2)
    prop.greedy_search(top=top, objective=objective)
    mols = prop.tree[1][0]
    smiles = [get_smiles(mol) for mol in mols]
    return smiles, mols


class MCTSNode:
    def __init__(self, smiles, W=0, N=0, P=0):
        self.smiles = smiles
        self.children = []
        self.W = W
        self.N = N
        self.P = P  # Predictive Score
        self.mol = get_mol(smiles)  # [get_mol(smi) for smi in self.smiles]
        self.MW = Descriptors.MolWt(
            self.mol
        )  # [Descriptors.MolWt(mol) for mol in self.mols]

    def __repr__(self):
        w = round(self.W, 4)
        p = round(self.P, 4)
        return f"{self.__class__.__name__}(W={w}, N={self.N}, P={p}, MW={self.MW})"

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return C_PUCT * self.P * np.sqrt(n) / (1 + self.N)

    def UCT(self, n):
        # upper confidence tree
        C = 2.0
        return self.W / n + C * np.sqrt(self.N / n)

    def reward(self, n):
        pass

    # nodes = MCTSNode(smiles='C', W=0, N=0, P=0)


class MCTS:
    def __init__(
        self,
        root_smiles,
        building_blocks,
        scoring_function,
        objective="PlogP",
        n_choice=1,
        epsilon=0,
        max_mw=MAX_MW,
        max_nodes=MAX_NODES,
    ):
        """

        :param root_smiles: SMILES
        :param building_blocks: LIST[SMILES]
        :param scoring_function: function
        :param objective: str
        :param n_choice: int
        :param epsilon: int
        """
        self.building_blocks = building_blocks
        self.scoring_function = scoring_function
        self.objective = objective
        self.n_choice = n_choice
        self.epsilon = epsilon
        self.max_mw = max_mw
        self.max_nodes = max_nodes

        # Memos generated molecules
        self.state_map = defaultdict(smiles=root_smiles)
        self.root = MCTSNode(root_smiles)
        self.target_scores = self.calc_target_scores(building_blocks=building_blocks)

    def __repr__(self):
        return f"{self.__class__.__name__}(OBJECTIVE={self.objective}, MAX_MW={self.max_mw})"

    def set_scoring_function(self, score_func, objective):
        self.scoring_function = score_func
        self.objective = objective

    def set_threshold(self):
        pass

    def calc_target_scores(self, building_blocks):
        if not isinstance(building_blocks, list):
            raise ValueError("Building Blocks SMILES must be List")

        return self.scoring_function(building_blocks)

    def rollout(self, node, state_map, scoring_function):
        # print(node.MW)
        if node.MW > self.max_mw:
            return node.P

        if len(node.mol.GetAtoms()) > self.max_nodes:
            return node.P

        # cand_smis = selector_eg(self.init_smis, self.target_scores, num=self.n_choice, epsilon=self.epsilon)
        selected_smiles = selector(self.building_blocks)
        for cand_smi in selected_smiles:
            # select SMILES based on the property score.
            new_smiles, _ = candidate_smiles(
                node.smiles, cand_smi, top=self.n_choice, objective=self.objective
            )
            for new_smile in new_smiles:
                if new_smile in state_map:
                    new_node = state_map[new_smile]
                else:
                    new_node = MCTSNode(new_smile)
                    node.children.append(new_node)
                state_map[new_node.smiles] = new_node
                if len(node.children) == 0:
                    return node.P  # cannot find leaves

                scores = scoring_function([x.smiles for x in node.children])
                for child, score in zip(node.children, scores):
                    child.P = score

        sum_count = sum([c.N for c in node.children])
        # TODO: Need to bug fix.
        selected_node = max(node.children, key=lambda x: x.Q() + x.U(n=sum_count))
        try:
            v = self.rollout(
                node=selected_node,
                state_map=self.state_map,
                scoring_function=self.scoring_function,
            )
        except Exception:
            # print(node.children)
            # print(node.smiles, new_smiles)
            v = 0
        # Backward pass
        selected_node.W += v
        selected_node.N += 1
        return v

    def run(self, n_rollout=100):
        for _ in tqdm(range(n_rollout)):
            self.rollout(self.root, self.state_map, self.scoring_function)


class SmilesNode:
    def __init__(self, mcts: MCTS):
        self.smiles = []
        self._get_smiles_tree(mcts.root)

    def _get_smiles_tree(self, root):
        if not root:
            return

        for children in root.children:
            if children:
                self.smiles.append(children.smiles)
                self._get_smiles_tree(children)
