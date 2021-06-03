import pandas as pd
import rdkit.Chem as Chem
import numpy as np

from libs.chemutils import get_mol, get_smiles
from libs.chemutils import mol_from_graph, mol_to_graph, sanitize
from libs.gspan_utils import create_gspan_dataset, create_junction_tree_dataset
from libs.gspan_utils import gspan_to_mol

from rdkit.Chem import Draw
from rdkit.Chem import Descriptors

from dataclasses import dataclass
from gspan_mining.config import parser
from gspan_mining.main import main
import networkx as nx





def run():
    tup_list = create_gspan_dataset(mols)
    tup_list2, smiles2id = create_junction_tree_dataset(mols)


def test_mol_to_graph():
    mols = [sanitize(get_mol(s[0])) for s in df.values]
    graphs = [mol_to_graph(mol) for mol in mols]
    r_mols = []
    for g, n, e in graphs:
        mol = mol_from_graph(nodes=n, edges=e)
        r_mols.append(mol)
    return


args_str = '-s 2 -d False -l 5 -p False -w False data/graph/sample_gspan_jt.data'
FLAG, _ = parser.parse_known_args(args=args_str.split())
def run_gspan(FLAG):
    return main(FLAG)

gspan = run_gspan(FLAG)
len(gspan.ggnx)

df_zinc = pd.read_table("data/zinc/all.txt", header=None).iloc[0:50]
mols = [Chem.MolFromSmiles(s[0]) for s in df_zinc.values]
new_mols = gspan_to_mol(gspan, method='jt', dataset=df_zinc)





