import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from moldr.core.decompose import GraphMining, DefaultConfig, MolsMining
from moldr.gspan.utils import create_gspan_dataset_nx, save_graph
from moldr.chemutils import (
    get_mol,
    get_smiles,
    mol_from_graph,
    mol_to_graph,
    sanitize,
    force_kekulize,
)


def run_and_save_(config, nx_graph_list):
    dataset = create_gspan_dataset_nx(nx_graph_list, "label", "label")
    if not config.data_path.parent.exists():
        config.data_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph(dataset, fpath=config.data_path)


class TestGSpan:
    def __init__(self, config: DefaultConfig):
        self.config = config
        self.save_name = None

    def test_gspan(self, graphs):
        print("TEST GSPAN")
        run_and_save_(self.config, graphs)
        runner = GraphMining(self.config)
        gspan = runner.decompose(graphs=[])
        gspan._report_df.sort_values(by="support")

    def test_gspan_mol(self, mols):
        print("TEST GSPAN MOL")
        runner = MolsMining(self.config)
        gspan = runner.decompose(mols=mols)
        gspan._report_df.sort_values(by="support")
        print(gspan._report_df.head(10))
        return gspan


def main():
    smiles = pd.read_csv("./data/zinc/all.txt", header=None).iloc[:100, :]
    mols = [get_mol(s[0]) for s in smiles.values]
    mols = [sanitize(m) for m in mols if m is not None]
    graphs = [mol_to_graph(m) for m in mols if m is not None]
    # mols = [force_kekulize(m) for m in mols if m is not None]

    save_dir = "./tmp/building_blocks/zinc"
    minsup = int(len(mols) * 0.5)
    config = DefaultConfig(
        data_path=Path(save_dir) / "test.data",
        support=minsup,
        lower=1,
        upper=1000,
        method="jt",
    )
    test_gspan = TestGSpan(config)
    test_gspan.test_gspan(graphs)
    test_gspan.test_gspan_mol(mols)


if __name__ == "__main__":
    main()
