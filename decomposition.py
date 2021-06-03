import os
import pandas as pd
from tqdm import tqdm

from libs.chemutils import get_mol, get_smiles
from libs.gspan_utils import preprocess_mols, gspan_to_mol

from gspan_mining.gspan import gSpan
import argparse

SAVE_DIR_GSPAN = 'data/graph'
SAVE_DIR_MINING = 'data/results'


def run_gspan(args, graph, dataset=None):
    gspan = gSpan(
        database_file_name=graph,
        min_support=args.minsup,
        min_num_vertices=args.length,
        max_num_vertices=float('inf'),
        max_ngraphs=float('inf'),
        is_undirected=True,
        verbose=False,
        visualize=False,
        where=False
    )
    gspan.run()
    gspan.time_stats()
    new_mols = gspan_to_mol(gspan, method=args.method, dataset=dataset)
    new_smis = [get_smiles(m) for m in new_mols]
    if len(new_smis) < 1:
        print("NO MINED MOLECULES.")
        return

    save_dir_smiles = os.path.join(SAVE_DIR_MINING, f'zinc_{args.method}_gspan_s{args.minsup}_l{args.length}.csv')
    pd.DataFrame(new_smis, columns=['smiles']).to_csv(save_dir_smiles)
    print(f"SAVE SMILES TO {save_dir_smiles}")


def save_gspan(args):
    dataset = pd.read_table(args.data, header=None)
    method = 'jt' if args.jt else 'raw'
    if args.preprocess:
        print("Convert SMILES to MOLECULES")
        gspan_data = os.path.join(SAVE_DIR_GSPAN, f"gspan_{args.method}.data")
        mols = [get_mol(s[0]) for s in tqdm(dataset.values)]
        print("Convert MOLECULES to gSpan dataset")
        preprocess_mols(mols, gspan_data, method=method)
    if not args.gspan:
        print("--gspan False. END.")
        return
    else:
        print("==" * 50)
        print(f"method={method}: minsup={args.minsup}, length={args.length}")
        print("==" * 50)
        gspan_data = args.gspan_data

    run_gspan(args, graph=gspan_data, dataset=dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--minsup', type=int, default=1000)
    parser.add_argument('--length', type=int, default=7)
    parser.add_argument('--jt', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--gspan', action='store_true')
    parser.add_argument('--gspan_data')
    args = parser.parse_args()
    save_gspan(args)
