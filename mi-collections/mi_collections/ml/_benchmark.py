from collections import Counter

import numpy as np
import pandas as pd
from grakel import WeisfeilerLehman
from mi_collections.chemutils import get_mol
from mi_collections.ecfp.features import get_all_identifiers
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR


def read_dataset():
    path = "./data/delaney-processed.csv"
    data = pd.read_csv(path, index_col=0)
    y = data["measured log solubility in mols per litre"]
    mols = [get_mol(s) for s in data.smiles]
    return {"mols": mols, "y": y}


def preprocess(mols, invariants=[]):
    # example of invariant atoms:
    # invariant = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    sentences = [
        get_all_identifiers(mol, radius=2, is_dropna=True, invariants=invariants)
        for mol in mols
    ]
    vertex_histogram = [Counter(sentence) for sentence in sentences]
    return pd.DataFrame(vertex_histogram).fillna(0)


def evaluate_models(model, X, y, n_splits=10):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    acc_list = []
    for train_index, test_index in cv.split(X, y):
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X[train_index, :], X[test_index, :]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        acc_list.append(acc)
    print(f"ave={np.mean(acc_list)} +/-{np.std(acc_list)}, nums={X.shape[1]}")
    return acc_list


def exponential_wl(G, y, n_iter=2):
    wl_kernel = WeisfeilerLehman(n_iter=n_iter, normalize=True)
    wl_kernel.fit(G)
    model = RidgeCV()
    accs = []
    for i, vertex_hist in wl_kernel.X.items():
        X = vertex_hist.X.toarray()
        acc = evaluate_models(model, X, y, n_splits=10)
        accs.append(acc)
