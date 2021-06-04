import os
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from libs.chemutils import get_mol, get_smiles
from libs.search_tree import MCTS, SmilesNode
from libs.property_simulator import scoring_function_plogp, scoring_function_qed
from libs.search_tree import selector_eg
from multiprocessing import Pool
from datetime import datetime
import argparse

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='bulding blocks path')
    parser.add_argument('--prop', type=str, default='PlogP', help='select PlogP or QED objective properties')
    parser.add_argument('--rollout', type=int, default=1000, help='maximum number of searching tree')
    parser.add_argument('--c_puct', type=float, default=10, help='exploration parameter of MCTS')
    parser.add_argument('--max_node', type=int, default=100, help='termination condition of maximum node')
    parser.add_argument('--max_mw', type=int, default=1000, help='termination condition of molecular weight')
    parser.add_argument('--top', type=int, default=1000, help='the maximum number of building blocks using')
    parser.add_argument('--ncpu', type=int, default=1,
                        help='the number of parallel cores (not implemented yet)')
    parser.add_argument('--epsilon', type=int, default=0,
                        help='parameter for selecting building blocks by epsilon greedy')
    parser.add_argument('--save', action='store_true', help='save result')

    args = parser.parse_args()

    scoring_function = scoring_function_plogp if args.prop == 'PlogP' else scoring_function_qed

    # READ: Building Blocks
    mined_smiles = pd.read_csv(args.data, index_col=0)
    mined_mols = [get_mol(s) for s in mined_smiles.smiles]
    building_blocks_smiles = list([get_smiles(m) for m in mined_mols])

    # add for minor differences support
    vocab = [s for s in pd.read_csv('data/graph/vocab.csv', index_col=0).smiles if len(s) > 1]

    building_blocks_smiles = vocab#building_blocks_smiles# + vocab

    scores = np.array(scoring_function(building_blocks_smiles))

    # select top N score building blocks for better substructures.
    building_blocks_smiles = selector_eg(building_blocks_smiles, scores, num=args.top, epsilon=args.epsilon)

    mcts = MCTS(root_smiles="CC",
                building_blocks=building_blocks_smiles,
                scoring_function=scoring_function,
                objective=args.prop,
                max_mw=args.max_mw)

    print(mcts)
    mcts.run(n_rollout=args.rollout)
    # pool = Pool(8)
    # pool.map()
    result = SmilesNode(mcts)
    # gen_scores = np.array(scoring_function(result.smiles))

    # Penalized logP value changes if a molecule is kekule type.
    gen_mols = np.array([Chem.MolFromSmiles(s) for s in result.smiles])
    gen_smiles = np.array([Chem.MolToSmiles(m) for m in gen_mols])
    gen_scores = np.array(scoring_function(gen_smiles))

    idx = np.argsort(-gen_scores)
    TOP_N = 5
    print(f"TOP {TOP_N} {args.prop} SCORE:", gen_scores[idx[0:TOP_N]])
    print(f"TOP {TOP_N} {args.prop} SMILES:", gen_smiles[idx[0:TOP_N]])
    if args.save:
        date = datetime.today().strftime("%Y%m%d%H%M%S")
        save_name = date + '_' + args.prop + '.csv'
        pd.DataFrame(gen_smiles, columns=['smiles']).to_csv(f"data/results/generated/{save_name}")
