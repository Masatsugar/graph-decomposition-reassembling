import os

import pandas as pd
import rdkit.Chem as Chem
from mi_collections.mol2vec.model import Mol2Vec


def test_train():
    smiles = ["CC", "CCO", "CCC", "c1ccccc1"]
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    model = Mol2Vec(vector_size=100, epochs=10, threshold=None, save_model=True)
    model.fit(mols)
    assert model.transform(mols).shape[1] == model.vector_size


if __name__ == "__main__":
    test_train()
