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


def run_gspan(graphs, config):
    run_and_save_(config, graphs)
    # run_and_save_(config, graphs)
    runner = GraphMining(config)
    gspan = runner.decompose(graphs=[])
    gspan._report_df.sort_values(by="support")


def run_gspan_mol(config, mols):
    runner = MolsMining(config)
    gspan = runner.decompose(mols=mols)
    gspan._report_df.sort_values(by="support")
    print(gspan._report_df.head(10))
    return gspan


def main():
    smiles = pd.read_csv("./data/zinc/all.txt", header=None).iloc[:100, :]
    mols = [get_mol(s[0]) for s in smiles.values]
    mols = [sanitize(m) for m in mols if m is not None]
    # mols = [force_kekulize(m) for m in mols if m is not None]

    save_dir = "./tmp/building_blocks/zinc"
    minsup = int(len(mols) * 0.1)
    config = DefaultConfig(
        data_path=Path(save_dir) / "test.data",
        support=minsup,
        lower=1,
        upper=1000,
        method="jt",
    )
    run_gspan_mol(config, mols)


if __name__ == "__main__":
    main()
