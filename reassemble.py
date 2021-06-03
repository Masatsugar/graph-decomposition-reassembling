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
    parser.add_argument('--data', required=True)
    parser.add_argument('--prop', type=str, default='PlogP')
    parser.add_argument('--rollout', type=int, default=1000)
    parser.add_argument('--c_puct', type=float, default=10)
    parser.add_argument('--max_node', type=int, default=20)
    parser.add_argument('--max_mw', type=int, default=300)
    parser.add_argument('--top', type=int, default=100)
    parser.add_argument('--ncpu', type=int, default=1)  # not implemented yet
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    scoring_function = scoring_function_plogp if args.prop == 'PlogP' else scoring_function_qed

    # Building Blocks path
    # fpath = 'data/results/tmp/zinc_jt_gspan_s1000_l7.csv'
    mined_smiles = pd.read_csv(args.data, index_col=0)
    mined_mols = [get_mol(s) for s in mined_smiles.smiles]
    building_blocks_smiles = list([get_smiles(m) for m in mined_mols])

    # add for minor differences support
    vocab = [s for s in pd.read_csv('data/graph/vocab.csv', index_col=0).smiles if len(s) > 1]

    scores = np.array(scoring_function(building_blocks_smiles))

    # select top N score building blocks for better substructures.
    building_blocks_smiles = selector_eg(building_blocks_smiles, scores, num=args.top, epsilon=0) + vocab

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
    gen_scores = np.array(scoring_function(result.smiles))
    gen_smiles = np.array(result.smiles)
    # gen_mols = np.array([Chem.MolFromSmiles(s) for s in gen_smiles])

    idx = np.argsort(-gen_scores)
    TOP_N = 5
    print(f"TOP {TOP_N} {args.prop} SCORE:", gen_scores[idx[0:TOP_N]])
    print(f"TOP {TOP_N} {args.prop} SMILES:", gen_smiles[idx[0:TOP_N]])

    if args.save:
        save_name = datetime.today().strftime('%Y_%m%d_%s') + '_' + args.prop + '.csv'
        pd.DataFrame(gen_smiles, columns=['smiles']).to_csv(f"data/results/generated/{save_name}")

