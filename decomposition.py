import os
import pandas as pd
from tqdm import tqdm

from libs.chemutils import get_mol, get_smiles
from libs.gspan_utils import preprocess_mols, gspan_to_mol

from gspan_mining.config import parser
from gspan_mining.main import main
import argparse


SAVE_DIR = 'data/graph/mining'


def run_gspan(flag):
    return main(flag)


def save_gspan(minsup, length, method, is_gspan=True, dataset=None):
    print("==" * 50)
    print(f"method={method}: minsup={minsup}, length={length}")
    print("==" * 50)
    path_raw = os.path.join(SAVE_DIR, f"sample_gspan_{method}.data")
    preprocess_mols(mols, path_raw, method=method)
    if not is_gspan:
        return "END."

    args_str = f'-s {minsup} -d False -l {length} -p False -w False {path_raw}'
    flag, _ = parser.parse_known_args(args=args_str.split())
    gspan = run_gspan(flag)
    new_mols = gspan_to_mol(gspan, method=method, dataset=dataset)
    new_smis = [get_smiles(m) for m in new_mols]
    if len(new_smis) < 1:
        print("NO MINED MOLECULES.")
        return

    save_dir_smiles = os.path.join(SAVE_DIR, f'zinc_{method}_gspan_s{minsup}_l{length}.csv')
    pd.DataFrame(new_smis, columns=['smiles']).to_csv(save_dir_smiles)
    print(f"SAVE SMILES TO {save_dir_smiles}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--support', type=int, default=1000)
    parser.add_argument('--length', type=int, default=5)
    args = parser.parse_args()

    df_zinc = pd.read_table(args.data, header=None)
    print("Convert SMILES to MOLECULES")
    mols = [get_mol(s[0]) for s in tqdm(df_zinc.values)]
    save_gspan(minsup=args.minsup, length=args.length, method=args.method, dataset=df_zinc)
