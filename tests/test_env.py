from mi_collections.chemutils import get_mol
from mi_collections.mol2vec.model import Mol2Vec

from moldr.env import MolEnvValueMax


def test_mol2vec():
    sample_smiles = ["CC", "CCO", "CCC"]
    sample_mols = [get_mol(s) for s in sample_smiles]
    mol2vec = Mol2Vec(model_path="./models/model_300dim.pkl")
    vecs = mol2vec.fit_transform(sample_mols)
    assert vecs.shape[0] == len(sample_mols)


def scoring_function(smiles) -> float:
    mol = get_mol(smiles)
    return len(mol.GetAtoms()) / 100


if __name__ == "__main__":
    mol2vec = Mol2Vec(model_path="./models/model_300dim.pkl")
