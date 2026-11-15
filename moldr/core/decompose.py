import collections
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import Draw, Mol


from moldr.chemutils import (
    get_mol,
    get_smiles,
    mol_from_graph,
    mol_to_graph,
    mutag_convert,
    sanitize,
)

from moldr.gspan import gSpan
from moldr.gspan.fast import FastMiningResult, mine_fast_gspan
from moldr.gspan.utils import gspan_to_mol, preprocess_graphs, preprocess_mols


@dataclass
class DefaultConfig:
    data_path: Path
    support: int = 1
    lower: int = 2
    upper: int = 4
    directed: bool = False
    is_save: bool = True
    output_csv: bool = False
    method: str = "raw"
    backend: str = "fast-gspan"
    num_threads: int = 0
    timeout: Optional[float] = None
    show_progress: bool = False


def check_substructures(path, max_nums=100, **kwargs):
    df = pd.read_csv(path, index_col=0)
    sub_mols = [Chem.MolFromSmiles(s) for s in df["0"]]
    Draw.MolsToGridImage(sub_mols[0:max_nums], **kwargs).show()


def get_matrix(report_df, wl_kernel, y):
    # report_df = pd.read_csv(self._config.path.with_name(f"{name}_info.csv"))[
    #     "support_ids"
    # ]
    # report_df = gspan._report_df['support_ids']
    ncols = report_df.shape[0]
    nums = report_df.to_numpy()
    mat = np.zeros((len(y), ncols))
    for i in range(ncols):
        cnt = Counter(nums[i].split(","))
        for key, val in cnt.items():
            mat[int(key), i] = val

    mat = np.array(mat)  # pd.DataFrame(mat)
    X = np.c_[wl_kernel.X[0].X.toarray(), mat]
    return X


class GraphMining:
    def __init__(self, config: DefaultConfig):
        self.config = config
        self.save_name = None

    def _run_gspan(self):
        if self.config.backend == "fast-gspan":
            return mine_fast_gspan(self.config)
        if self.config.backend != "python":
            raise ValueError("backend must be 'fast-gspan' or 'python'")
        gs = gSpan(
            database_file_name=self.config.data_path,
            min_support=self.config.support,
            min_num_vertices=self.config.lower,
            max_num_vertices=self.config.upper,
            is_undirected=not self.config.directed,
        )
        gs.run()
        gs.time_stats()
        return gs

    def decompose(self, graphs=None):
        if graphs:
            preprocess_graphs(graphs, self.config.data_path)
        gspan_object = self._run_gspan()
        return gspan_object


class MolsMining(GraphMining):
    def __init__(self, config: DefaultConfig):
        super(MolsMining, self).__init__(config=config)
        self.config = config
        self.save_name = None

    def decompose(self, mols: Optional[List[Mol]] = None) -> gSpan | FastMiningResult:
        if mols is not None:
            preprocess_mols(
                mols, fname=self.config.data_path, method=self.config.method
            )

        gspan_object = self._run_gspan()
        if self.config.is_save:
            self.save_csv(gspan_object=gspan_object, mols=mols)

        return gspan_object

    def save_csv(
        self,
        gspan_object: gSpan | FastMiningResult,
        mols: Optional[List[Mol]] = None,
        suffix: str = ".pickle",
    ):
        cnf = self.config
        save_name = (
            f"{cnf.data_path.name.split('.')[0]}_s{cnf.support}l{cnf.lower}u{cnf.upper}"
        )

        # fpath = cnf.path.with_name(save_name).with_suffix(suffix)
        self.save_name = save_name

        # Save as CSV
        smiles = [get_smiles(m) for m in mols] if mols is not None else None
        sub_mols = self.gspan_to_mols(gspan_object, smiles_list=smiles)
        sub_smiles = [get_smiles(m) for m in sub_mols]
        pd.DataFrame(sub_smiles, columns=["0"]).to_csv(
            cnf.data_path.with_name(save_name).with_suffix(".csv")
        )
        gspan_object._report_df["support_ids"].to_csv(
            cnf.data_path.with_name(f"{save_name}_info.csv")
        )

    def gspan_to_mols(
        self, gspan: gSpan | FastMiningResult, smiles_list: Optional[List[str]] = None
    ):
        return gspan_to_mol(gspan, self.config.method, smiles_list=smiles_list)

    @staticmethod
    def save(fpath: Path, obj: gSpan | FastMiningResult):
        with fpath.open("wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(fpath):
        with fpath.open("rb") as f:
            obj = pickle.load(f)
        return obj


if __name__ == "__main__":
    test_smiles = [
        "CC1CCC2=CC=CC=C2O1",
        "CC",
        "COC",
        "c1ccccc1",
        "CC1C(=O)NC(=O)S1",
        "CO",
    ]
    test_mols = [get_mol(s) for s in test_smiles]

    cnf = DefaultConfig(data_path=Path("outputs/tests/gspan_jt.data"), method="jt")
    runner = MolsMining(config=cnf)
    # gspan_obj = runner._run_gspan()
    gspan_obj = runner.decompose(test_mols)
    runner.save(Path("outputs/test/gspan_jt.pickle"), gspan_obj)
    test = runner.load(cnf.data_path.with_name(runner.save_name))
    sub_mols = gspan_to_mol(gspan_obj, method=cnf.method, smiles_list=test_smiles)
    sub_smiles = [get_smiles(m) for m in sub_mols]

    dat, info = get_identifier(test_mols[0], radius=2)
    dat = pd.DataFrame(dat).T.to_numpy().flatten()
    dat = [d for d in dat if d is not None]
    cd = collections.Counter(dat)
    ds = np.unique(dat)
