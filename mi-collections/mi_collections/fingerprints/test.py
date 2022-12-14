from __future__ import print_function

import os
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import seaborn as sns
from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman
from matplotlib import pyplot as plt
from mi_collections.chemutils import (
    get_mol,
    get_smiles,
    mol_from_graph,
    mol_to_graph,
    mutag_convert,
    sanitize,
)
from mi_collections.gspan.config import parser
from mi_collections.gspan.main import main
from mi_collections.gspan.utils import gspan_to_mol, preprocess_mols
from mi_collections.moldr.decompose import DefaultConfig, MolsMining
from rdkit.Chem import Draw
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

os.environ["HTTP_PROXY"] = "http://10.58.254.244:7080"
os.environ["HTTPS_PROXY"] = "http://10.58.254.244:7080"


def __test_grapkel():
    # Loads the MUTAG dataset
    MUTAG = fetch_dataset("MUTAG", verbose=False)
    G, y = MUTAG.data, MUTAG.target

    # Splits the dataset into a training and a test set
    G_train, G_test, y_train, y_test = train_test_split(
        G, y, test_size=0.1, random_state=42
    )

    # Uses the shortest path kernel to generate the kernel matrices
    # gk = ShortestPath(normalize=True)
    wl_kernel = WeisfeilerLehman(n_iter=2)
    K_train = wl_kernel.fit_transform(G_train)
    K_test = wl_kernel.transform(G_test)

    # Uses the SVM classifier to perform classification
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    # Computes and prints the classification accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", str(round(acc * 100, 2)) + "%")


def vertex_histogram_concat(wl_kernel):
    wl_mat = {
        "X": [],
    }
    for k, v in wl_kernel.X.items():
        # x = v.X.shape[1]
        wl_mat["X"].append(v.X.toarray())

    for k, v in wl_mat.items():
        wl_mat[k] = np.hstack(v)

    return wl_mat["X"]


def test_result(n_iter=2, method="ridge", is_concat=True, is_kernel="linear"):
    accuracy_scores = []
    stds = []
    subs = []
    np.random.seed(42)

    MUTAG = fetch_dataset("MUTAG", verbose=True)
    G, y = MUTAG.data, MUTAG.target
    wl_kernel = WeisfeilerLehman(n_iter=n_iter)
    wl_kernel.fit(G)
    all_X = vertex_histogram_concat(wl_kernel)
    total_nums = 0

    cv = StratifiedKFold(n_splits=10, shuffle=True)
    for i, vertex_hist in wl_kernel.X.items():
        if i == 30:
            break

        if method == "ridge":
            clf = RidgeClassifierCV()
        else:
            clf = SVC(kernel="rbf")
        # clf = MLPClassifier(hidden_layer_sizes=(10000, 2))
        accs = []
        if is_concat:
            nums = wl_kernel.X[i].X.shape[1]
            X = all_X[:, 0 : total_nums + nums]
        else:
            X = vertex_hist.X.toarray()

        subs.append(X.shape[1])
        for train_index, test_index in cv.split(X, y):
            y_train, y_test = y[train_index], y[test_index]
            X_train, X_test = X[train_index, :], X[test_index, :]
            if is_kernel == "linear":
                K_train = X_train.dot(X_train.T)
                K_test = X_test.dot(X_train.T)
                X_train = K_train
                X_test = K_test
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            acc = accuracy_score(y_test, pred)
            accs.append(acc)

        print(f"{i}, ave={np.mean(accs)} +/-{np.std(accs)}, nums={X.shape[1]}")
        accuracy_scores.append(np.mean(accs))
        stds.append(np.std(accs))

    return {"acc": accuracy_scores, "std": stds}


def plot_acc(res, method="svm", is_concat=False, is_kernel="linear"):
    accuracy_scores, stds = res["acc"], res["std"]
    N = len(stds)
    idx = [i for i in range(0, N)]

    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    sns.set_style("whitegrid")

    config = {"figsize": (6, 4), "dpi": 1000}
    fig = plt.figure(**config)
    ax = fig.gca()
    plt.plot(idx, accuracy_scores)
    ax.fill_between(
        idx,
        [accuracy_scores[i] - stds[i] for i in range(N)],
        [accuracy_scores[i] + stds[i] for i in range(N)],
        alpha=0.16,
        color=sns.color_palette()[0],
    )
    # plt.title()
    plt.ylabel("Accuracy")
    plt.xlabel("Ns")
    plt.show()


def read_esoldata():

    path = Path("data/esol/delaney-processed.csv")

    df = pd.read_csv(path)
    mols = [get_mol(s) for s in df.smiles]
    return mols


def read_mutagdata(is_aromatic=False):
    MUTAG = fetch_dataset("MUTAG", verbose=False)
    graphs = [mutag_convert(data, is_aromatic=False) for data in MUTAG.data]
    mols = [mol_from_graph(graph) for graph in graphs]
    return mols


if __name__ == "__main__":

    save_name = "graph"
    save_dir = Path("data/mutag/graphlets") / save_name
    save_path = save_dir.with_suffix(".data")
    is_gspan = False
    max_length = 10
    mols = read_mutagdata()
    if is_gspan:
        for u in tqdm(range(1, max_length)):
            config = DefaultConfig(path=save_path, upper=u)
            runner = MolsMining(config=config)
            res = runner.decompose(mols)

    # test_result(n_iter=30)
    # Draw.MolsToGridImage(mols[0:100], molsPerRow=10).show()
    # Draw.MolsToGridImage(res["sub_mols"][0:100], molsPerRow=10).show()
    n_iter = 10
    method = "svm"
    is_concat = True
    is_kernel = False

    exp_config = {"n_iter": 10, "method": "svm", "is_concat": True, "is_kernel": False}
    res = test_result(**exp_config)
    plot_acc(res, **exp_config)
