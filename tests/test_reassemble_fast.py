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
from moldr.core.reassemble import merge_edge, merge_node
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
