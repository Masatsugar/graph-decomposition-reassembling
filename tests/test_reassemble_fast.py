"""Equivalence and contract tests for the networkx-free reassembly fast path.

Phase 1 replaced the RDKit<->networkx<->SMILES round trips in the merge code
with direct numpy/RWMol helpers (``mol_to_adj`` / ``mol_from_adj``). These tests
pin down that the fast helpers are *exactly* equivalent to the original
networkx path (which still lives in ``chemutils`` as ``mol_to_graph`` /
``mol_from_graph``), and that the public merge functions keep their contract.
"""

import numpy as np
import pytest

from moldr.chemutils import (
    get_mol,
    get_smiles,
    mol_from_adj,
    mol_from_graph,
    mol_to_adj,
    mol_to_graph,
)
from moldr.core.reassemble import (
    get_ring_targets,
    merge_edge,
    merge_mol_from_edge,
    merge_mol_from_node,
    merge_node,
)
from moldr.core.molgraph import MolGraph, sanitize_molgraph

# A deliberately diverse set: chains, hetero atoms, fused/aromatic rings,
# multiple bond orders, charges-free building-block-like fragments.
SMILES = [
    "C",
    "CC",
    "CCO",
    "CC(=O)O",
    "C#N",
    "c1ccccc1",
    "C1CCCCC1",
    "c1ccncc1",
    "CC1CCC2=CC=CC=C2O1",
    "CC1C(=O)NC(=O)S1",
    "C1=CCc2ccccc2C1",
    "O=C(O)c1ccccc1",
    "CN1CCN(C)CC1",
    "c1ccc2[nH]ccc2c1",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
]


def _canonical(mol):
    return get_smiles(mol) if mol is not None else None


@pytest.mark.parametrize("smiles", SMILES)
def test_mol_to_adj_matches_networkx(smiles):
    """mol_to_adj returns the same (adjacency, labels) as the networkx path."""
    mol = get_mol(smiles)
    graph = mol_to_graph(mol)
    nx_nodes = [n["label"] for n in graph.nodes.values()]
    import networkx as nx

    nx_adj = nx.adjacency_matrix(graph).toarray()

    adj, nodes = mol_to_adj(mol)
    assert nodes == nx_nodes
    assert np.array_equal(adj, nx_adj)


@pytest.mark.parametrize("smiles", SMILES)
def test_mol_from_adj_matches_mol_from_graph(smiles):
    """mol_from_adj(mol_to_adj(m)) is identical to the networkx round trip."""
    mol = get_mol(smiles)

    # networkx reference path (the pre-refactor behaviour).
    graph = mol_to_graph(mol)
    ref = mol_from_graph(graph)

    # fast path.
    adj, nodes = mol_to_adj(mol)
    fast = mol_from_adj(adj, nodes)

    assert _canonical(fast) == _canonical(ref)


def test_merge_edge_never_returns_none_without_rings():
    """merge_edge must return a (possibly empty) list, never None.

    MolEnvValueMax._reassemble does ``_mol1.extend(merge_edge(...))`` with no
    guard, so returning None would raise. Ring-less inputs have no edge-merge.
    """
    base = get_mol("CC")  # no ring
    attach = get_mol("CCO")  # no ring
    result = merge_edge(base, attach)
    assert result is not None
    assert isinstance(result, list)
    assert result == []


def test_merge_node_produces_valid_candidates():
    base = get_mol("C")
    attach = get_mol("CC1CCC2=CC=CC=C2O1")
    candidates = merge_node(base, attach)
    assert len(candidates) > 0
    assert all(isinstance(c, MolGraph) for c in candidates)
    # every emitted SMILES must re-parse to a valid molecule.
    assert all(get_mol(c.smiles) is not None for c in candidates)


def test_merge_edge_produces_candidates_for_rings():
    base = get_mol("c1ccccc1")
    attach = get_mol("C1CCCCC1")
    result = merge_edge(base, attach)
    assert isinstance(result, list)
    assert len(result) > 0


def test_reassemble_pipeline_equivalence():
    """End-to-end: the env's reassembly step (merge_node + merge_edge ->
    sanitize_molgraph) yields valid, de-duplicated, re-parseable molecules."""
    base = get_mol("CC1CCC2=CC=CC=C2O1")
    attach = get_mol("c1ccccc1")
    candidates = list(merge_node(base, attach)) + list(merge_edge(base, attach))
    mols = sanitize_molgraph(candidates)
    smis = [get_smiles(m) for m in mols]
    assert len(smis) == len(set(smis))  # de-duplicated
    assert all(get_mol(s) is not None for s in smis)  # all valid


def test_molgraph_graph_is_lazy_but_correct():
    """MolGraph.graph is computed lazily yet equals the eager networkx graph."""
    mg = MolGraph("CC1CCC2=CC=CC=C2O1")
    assert mg._graph_built is False  # not built at construction
    g = mg.graph
    assert mg._graph_built is True
    # matches a direct conversion of the same mol.
    ref = mol_to_graph(get_mol("CC1CCC2=CC=CC=C2O1"))
    assert g.number_of_nodes() == ref.number_of_nodes()
    assert g.number_of_edges() == ref.number_of_edges()


@pytest.mark.parametrize("smiles", ["C[NH3+]", "[13CH3]O", "F[C@H](Cl)Br", "c1cc[nH]c1"])
def test_sanitize_molgraph_preserves_atom_chemistry(smiles):
    candidates = sanitize_molgraph([MolGraph(smiles)])
    assert len(candidates) == 1
    assert get_smiles(candidates[0]) == get_smiles(get_mol(smiles))


def test_sanitize_molgraph_drops_invalid_and_deduplicates_atom_maps():
    candidates = sanitize_molgraph([MolGraph("invalid"), MolGraph("[CH3:1][OH:2]"), MolGraph("CO")])
    assert [get_smiles(m) for m in candidates] == ["CO"]


@pytest.mark.parametrize("smiles", ["C1CC2CCC1C2", "C1CCC2CC1C2", "C1CC2CCC1CC2"])
def test_bridged_ring_targets_are_actual_bonds(smiles):
    mol = get_mol(smiles)
    targets = get_ring_targets(mol)
    expected = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in mol.GetBonds()
        if bond.IsInRing()
    }
    assert set(targets) == expected
    assert len(targets) == len(set(targets))


@pytest.mark.parametrize("direction", ["target1", "target2"])
def test_edge_merge_keeps_atom_labels_for_reversed_endpoints(direction):
    base, attach = get_mol("N1CCCCC1"), get_mol("C1CCOCC1")
    offset = base.GetNumAtoms()
    forward = merge_mol_from_edge(base, attach, (0, 1), (offset, offset + 1), direction)
    reversed_ends = merge_mol_from_edge(base, attach, (1, 0), (offset + 1, offset), direction)
    assert get_smiles(forward) == get_smiles(reversed_ends)


@pytest.mark.parametrize("smiles", ["C[NH3+]", "[13CH3]O", "c1cc[nH]c1"])
def test_mol_from_adj_with_templates_preserves_atom_properties(smiles):
    mol = get_mol(smiles)
    adj, nodes = mol_to_adj(mol)
    rebuilt = mol_from_adj(adj, nodes, atom_templates=list(mol.GetAtoms()))
    assert get_smiles(rebuilt) == get_smiles(mol)


@pytest.mark.parametrize("base,charge,isotope", [
    ("C[NH3+]", 1, 0),
    ("[13CH3]CO", 0, 13),
])
def test_node_merge_preserves_charge_and_isotope(base, charge, isotope):
    mols = sanitize_molgraph(merge_node(get_mol(base), get_mol("CC")))
    assert mols
    for mol in mols:
        assert sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == charge
        if isotope:
            assert sum(atom.GetIsotope() == isotope for atom in mol.GetAtoms()) == 1


@pytest.mark.parametrize("base,charge,isotope", [
    ("C1CC[NH2+]CC1", 1, 0),
    ("[13CH2]1CCCCC1", 0, 13),
])
def test_edge_merge_preserves_charge_and_isotope(base, charge, isotope):
    mols = sanitize_molgraph(merge_edge(get_mol(base), get_mol("C1CCCCC1")))
    assert mols
    for mol in mols:
        assert sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == charge
        if isotope:
            assert sum(atom.GetIsotope() == isotope for atom in mol.GetAtoms()) == 1


@pytest.mark.parametrize("smiles", ["C[C@H](O)F", "C/C=C/C"])
@pytest.mark.parametrize("merge", [merge_node, merge_edge])
def test_reassembly_rejects_stereo_instead_of_silently_corrupting_it(smiles, merge):
    for base, attach in [(get_mol(smiles), get_mol("CC")), (get_mol("CC"), get_mol(smiles))]:
        with pytest.raises(ValueError, match="Stereochemical input"):
            merge(base, attach)


def test_low_level_merge_rejects_stereo():
    with pytest.raises(ValueError, match="Stereochemical input"):
        merge_mol_from_node(get_mol("C[C@H](O)F"), get_mol("CC"), 0, 4)
