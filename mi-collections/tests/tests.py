import unittest
from dataclasses import dataclass
from pathlib import Path

import rdkit.Chem as Chem
from grakel.datasets import fetch_dataset
from mi_collections.chemutils import mol_from_graph, mol_to_graph, mutag_convert
from mi_collections.mol2vec.model import Mol2Vec
from mi_collections.moldr.reassemble import merge_edge, merge_node

model_path = "mi_collections/mol2vec/model/model_300dim.pkl"


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# mol2vec config
config = dotdict(
    {
        "data_path": "data/mol2vec/first_200props.sdf",
        "threshold": 2,
        "train_config": dotdict(
            {
                "method": "skip-gram",
                "window": 8,
                "vector_size": 100,
                "epochs": 10,
            }
        ),
    }
)


class MyTestCase(unittest.TestCase):
    mol2vec = Mol2Vec(n_radius=2, model_path=model_path)
    mols = Chem.SDMolSupplier(config.data_path)

    def test_mol2vec(self):
        vec = self.mol2vec.fit_transform(self.mols)
        self.assertEqual(300, vec.shape[1])

    def test_train(self):
        train_config = config.train_config
        _input = config.data_path
        _output = str(Path(_input).with_suffix(""))
        corpus = _output + f"_threshold{config.threshold}"
        # _output.with_name(f'{_output.name}_threshold{config.threshold}')
        self.mol2vec.generate_corpus(_input, _output, threshold=config.threshold)
        self.mol2vec.train(corpus, **train_config)
        self.mol2vec.load_model(
            f"{corpus}_{train_config.method}_vec{train_config.vector_size}_window{train_config.window}_min.pkl"
        )
        vec = self.mol2vec.fit_transform(self.mols)
        self.assertEqual(train_config.vector_size, vec.shape[1])

    def test_reconstruct_molgraph(self):
        _test_reconstruct_molgraph()

    def test_reassemble(self):
        _test_reassemble()


def read_sdf(path):
    mols = []
    with Chem.SDMolSupplier(path) as suppl:
        for mol in suppl:
            if mol is None:
                continue
            mols.append(mol)
    return mols


def _test_reassemble():
    mol1 = Chem.MolFromSmiles("C1CC1CCC(=O)N(C)(C)")
    mol2 = Chem.MolFromSmiles("C1C=C(C)C=NC=1C(=O)C")

    new_mols1 = merge_node(mol1, mol2)
    new_mols2 = merge_edge(mol1, mol2)
    print(new_mols1)
    print(new_mols2)
    if len(new_mols2) == 1:
        return True
    else:
        return False


def _test_reconstruct_molgraph():
    smiles = "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O"
    mol = Chem.MolFromSmiles(smiles)
    canonical_smiles = Chem.MolToSmiles(mol)
    new_smiles = Chem.MolToSmiles(mol_from_graph(mol_to_graph(mol)))
    assert canonical_smiles == new_smiles


if __name__ == "__main__":
    unittest.main()
