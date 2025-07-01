import moldr.chemutils as chem
from moldr.mol2vec import Mol2Vec
from moldr.env import MolEnvValueMax


def test_mol2alt_sentence(radius=2):
    sample_smiles = ["CC", "CCO", "CCC"]
    sample_mols = [chem.get_mol(s) for s in sample_smiles]
    sentences = chem.mol2alt_sentence(sample_mols[0], radius)
    return sentences


def test_mol2vec():
    sample_smiles = ["CC", "CCO", "CCC"]
    sample_mols = [chem.get_mol(s) for s in sample_smiles]
    mol2vec = Mol2Vec(model_path="./models/model_300dim.pkl")
    vecs = mol2vec.fit_transform(sample_mols)
    assert vecs.shape[0] == len(sample_mols)


def scoring_function(smiles) -> float:
    mol = chem.get_mol(smiles)
    return len(mol.GetAtoms()) / 100


if __name__ == "__main__":
    # test_mol2alt_sentence()
    test_mol2vec()
    # mol2vec = Mol2Vec(model_path="./models/model_300dim.pkl")
