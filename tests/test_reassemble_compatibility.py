"""Behavioral compatibility against original main and corrected chemistry."""

import json
import pickle
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

from moldr.chemutils import get_mol, get_smiles, mol_with_atom_index
from moldr.core.molgraph import MolGraph, sanitize_molgraph
from moldr.core.reassemble import merge_edge, merge_node
from tests.reference_reassembly import candidates as reference_candidates

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "reassembly_main_301fd9a.json"


def snapshot(base, fragment, **options):
    nodes = merge_node(base, fragment, **options)
    edges = merge_edge(base, fragment, **options)
    return {
        "node": [candidate.smiles for candidate in nodes],
        "edge": [candidate.smiles for candidate in edges],
        "final": [get_smiles(mol) for mol in sanitize_molgraph(nodes + edges)],
    }


@pytest.mark.parametrize("case", json.loads(FIXTURE_PATH.read_text())["cases"])
def test_neutral_candidates_and_order_match_original_main(case):
    assert snapshot(get_mol(case["base"]), get_mol(case["fragment"])) == case["candidates"]


@pytest.mark.parametrize("base,fragment", [
    ("C[NH3+]", "CC"), ("[13CH3]CO", "CC"), ("c1cc[nH]c1", "CC"),
    ("[NH2]C", "CO"), ("C1CC2CCC1C2", "C1CCC2CC1C2"),
    ("[13CH2]1CCCCC1", "C1CCCCC1"), ("C1CC[NH2+]CC1", "C1CCCCC1"),
])
def test_corrected_chemistry_matches_independent_reference(base, fragment):
    molecules = get_mol(base), get_mol(fragment)
    node, edge, final = reference_candidates(*molecules)
    assert snapshot(*molecules) == {"node": node, "edge": edge, "final": final}


@pytest.mark.parametrize("storage,shared,scan", list(product(
    ["int", "string"], [True, False], ["nonzero", "python"]
)))
def test_optimization_switches_preserve_candidate_order(storage, shared, scan):
    molecules = get_mol("N1CCCCC1"), get_mol("C1CCOCC1")
    node, edge, final = reference_candidates(*molecules)
    assert snapshot(*molecules, map_storage=storage, shared_preparation=shared, bond_scan=scan) == {
        "node": node, "edge": edge, "final": final,
    }


@pytest.mark.parametrize("storage", ["int", "string"])
def test_atom_zero_map_retained(storage):
    mol = mol_with_atom_index(get_mol("CO"), map_storage=storage)
    assert mol.GetAtomWithIdx(0).HasProp("molAtomMapNumber")
    assert mol.GetAtomWithIdx(0).GetIntProp("molAtomMapNumber") == 0
    assert ":0]" in get_smiles(mol)


def test_reassembly_does_not_mutate_input_properties():
    molecules = get_mol("C[NH3+]"), get_mol("CO")
    molecules[0].GetAtomWithIdx(0).SetIntProp("molAtomMapNumber", 42)
    before = [mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in molecules]
    snapshot(*molecules)
    assert before == [mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in molecules]


def test_graph_assignment_preserves_public_attribute_contract():
    molecule = MolGraph("CCO")
    replacement = nx.Graph()
    replacement.add_node(99, label="replacement")
    molecule.graph = replacement
    assert molecule.graph is replacement


def test_graph_from_original_main_pickle_state_is_preserved():
    # Original main pickled a plain graph attribute, without the lazy fields.
    molecule = MolGraph.__new__(MolGraph)
    graph = nx.Graph()
    graph.add_node(99, label="caller-edited")
    molecule.__dict__.update(smiles="CCO", mol=get_mol("CCO"), graph=graph)
    restored = pickle.loads(pickle.dumps(molecule))
    assert list(restored.graph.nodes) == [99]
    assert restored.graph.nodes[99]["label"] == "caller-edited"


@pytest.mark.parametrize("built", [False, True])
def test_lazy_graph_pickle_roundtrip(built):
    molecule = MolGraph("CCO")
    if built:
        assert molecule.graph.number_of_nodes() == 3
    restored = pickle.loads(pickle.dumps(molecule))
    assert restored._graph_built is built
    assert list(restored.graph.edges) == [(0, 1), (1, 2)]


def test_unchanged_legacy_environment_uses_fast_candidates():
    import gymnasium as gym

    from moldr.env import MolEnvValueMax

    model = Path(__file__).resolve().parents[1] / "models/model_300dim.pkl"
    if not model.is_file():
        pytest.skip("Repository Mol2Vec asset is required for the environment smoke test")
    scoring_calls = []

    def score(smiles):
        scoring_calls.append(smiles)
        return get_mol(smiles).GetNumAtoms() / 10.0

    environment = MolEnvValueMax({
        "ACTION_SPACE": gym.spaces.Discrete(2),
        "OBS_SPACE": gym.spaces.Box(-np.inf, np.inf, (300,), dtype=np.float32),
        "BUILDING_BLOCKS": ["CCO", "c1ccccc1"], "SCORE_FUNCTION": score,
        "FINAL_WEIGHT": 1.0, "LENGTH": 10, "SCORE_THRESHOLD": 0.5,
        "BASE_SMILES": "C", "MODEL_PATH": str(model),
    })
    observation, _ = environment.reset(seed=0)
    scoring_calls.clear()  # Count only the selected step, independently of reset scoring.
    assert observation.shape == (300,)
    observation, reward, done, truncated, info = environment.step(0)
    # The legacy algorithm still scores every candidate and takes its argmax.
    assert scoring_calls == info["gen_smiles"]
    scores = [get_mol(smiles).GetNumAtoms() / 10.0 for smiles in info["gen_smiles"]]
    assert environment.base_smiles == info["gen_smiles"][int(np.argmax(scores))]
    assert observation.shape == (300,) and np.isfinite(observation).all()
    assert np.isfinite(reward)
    assert not done and not truncated
