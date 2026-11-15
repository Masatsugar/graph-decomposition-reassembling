"""Exercise the actual fast-gspan binary and MOLDR's conversion boundary."""

import warnings
from dataclasses import replace

import networkx as nx
import pandas as pd
import pytest

from moldr.chemutils import get_mol, get_smiles
from moldr.core.decompose import DefaultConfig, GraphMining, MolsMining
from moldr.gspan.fast import FastMiningResult


@pytest.fixture
def config(tmp_path):
    # Also exercise paths that the old CLI-string parsing could not handle.
    return DefaultConfig(
        tmp_path / "graphs with spaces.data",
        support=2,
        lower=2,
        upper=4,
        num_threads=1,
        is_save=False,
    )


SMILES = ["CCO", "CCCO", "CCN", "CC(=O)O", "C1CCCCC1", "CC1CCCCC1"]


@pytest.mark.parametrize("method", ["raw", "jt"])
def test_fragments_and_support_match_python(config, method):
    mols = [get_mol(s) for s in SMILES]
    results = {}
    for backend in ["fast-gspan", "python"]:
        runner = MolsMining(replace(config, method=method, backend=backend))
        result = runner.decompose(mols)
        fragments = runner.gspan_to_mols(result, SMILES)
        assert all(m is not None for m in fragments)
        results[backend] = (result, {get_smiles(m) for m in fragments})
    fast, python = results["fast-gspan"][0], results["python"][0]
    assert isinstance(fast, FastMiningResult)
    assert results["fast-gspan"][1] == results["python"][1]

    node_match = nx.algorithms.isomorphism.categorical_node_match("label", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", None)
    assert len(fast.ggnx) == len(python.ggnx)
    for index, entry in enumerate(fast.ggnx):
        matches = [
            i
            for i, other in enumerate(python.ggnx)
            if nx.is_isomorphic(
                entry["graph"],
                other["graph"],
                node_match=node_match,
                edge_match=edge_match,
            )
        ]
        assert len(matches) == 1
        row = fast._report_df.iloc[index]
        reference = python._report_df.iloc[matches[0]]
        assert row.support == reference.support
        assert set(row.support_ids.split(",")) == set(reference.support_ids.split(","))
        assert len(row.support_ids.split(",")) == row.support


@pytest.mark.parametrize("method", ["raw", "jt"])
def test_csv_and_pickle_round_trip(config, method):
    runner = MolsMining(replace(config, method=method, is_save=True))
    result = runner.decompose([get_mol(s) for s in SMILES])
    csv_path = config.data_path.with_name(runner.save_name).with_suffix(".csv")
    table = pd.read_csv(csv_path, index_col=0)
    assert list(table.columns) == ["0"]
    assert len(table) > 0
    assert all(get_mol(s) is not None for s in table["0"])
    ids = pd.read_csv(csv_path.with_name(f"{runner.save_name}_info.csv"), index_col=0)
    assert len(ids) == len(result._report_df)

    pickle_path = config.data_path.with_suffix(".pickle")
    runner.save(pickle_path, result)
    restored = runner.load(pickle_path)
    pd.testing.assert_frame_equal(restored._report_df, result._report_df)
    assert {get_smiles(m) for m in runner.gspan_to_mols(restored, SMILES)} == set(
        table["0"]
    )


def test_networkx_labels_are_encoded_and_restored(config):
    graph = nx.Graph()
    graph.add_node(7, label="C")
    graph.add_node(12, label="O")
    graph.add_edge(7, 12, label="1.0")
    result = GraphMining(config).decompose([graph, graph])
    assert result._report_df.support.tolist() == [2]
    assert result._report_df.support_ids.tolist() == ["0,1"]
    assert set(result.ggnx[0]["vlbs"].values()) == {"C", "O"}
    assert list(result.ggnx[0]["elbs"].values()) == ["1.0"]
    # Reading the saved database preserves the existing API.
    reloaded = GraphMining(config).decompose([])
    pd.testing.assert_frame_equal(reloaded._report_df, result._report_df)


@pytest.mark.parametrize("lower,upper", [(1, 1), (1, 3), (2, 2), (3, 3)])
def test_bounds_and_singletons(config, lower, upper):
    runner = MolsMining(replace(config, lower=lower, upper=upper))
    result = runner.decompose([get_mol(s) for s in ["CCO", "CCCO", "O"]])
    assert result._report_df.num_vert.between(lower, upper).all()
    assert len(result._report_df) > 0
    assert all(m is not None for m in runner.gspan_to_mols(result))
    if lower == 1:
        singletons = result._report_df[result._report_df.num_vert == 1]
        assert sorted(singletons.support) == [2, 3]


@pytest.mark.parametrize("method", ["raw", "jt"])
@pytest.mark.parametrize("smiles,support", [([], 1), (["CCO"], 2), (["C", "O"], 1)])
def test_empty_results_have_valid_report_and_csv(config, method, smiles, support):
    runner = MolsMining(replace(config, method=method, support=support, is_save=True))
    result = runner.decompose([get_mol(s) for s in smiles])
    assert result._report_df.empty
    assert list(result._report_df.columns) == [
        "support",
        "description",
        "num_vert",
        "support_ids",
    ]
    path = config.data_path.with_name(runner.save_name).with_suffix(".csv")
    assert pd.read_csv(path, index_col=0).empty


def test_support_ids_preserve_input_positions(config):
    result = MolsMining(config).decompose(
        [get_mol(s) for s in ["N", "CCO", "O", "CCO"]]
    )
    assert set(result._report_df.support_ids) == {"1,3"}


@pytest.mark.parametrize(
    "options,message",
    [
        ({"directed": True}, "undirected"),
        ({"support": 0}, "support must"),
        ({"lower": 4, "upper": 2}, "vertex bounds"),
        ({"backend": "unknown"}, "backend must"),
    ],
)
def test_invalid_configuration(config, options, message):
    with pytest.raises(ValueError, match=message):
        MolsMining(replace(config, **options)).decompose([get_mol("CC")])


def test_low_support_floor_is_explicit(config):
    with pytest.raises(ValueError, match="1% support floor"):
        MolsMining(replace(config, support=1)).decompose([get_mol("CC")] * 200)


def test_backend_failure_is_not_an_empty_success(config, monkeypatch):
    def fail(*args):
        warnings.warn("gBolt exited with code 1", RuntimeWarning)
        return pd.DataFrame()

    monkeypatch.setattr("moldr.gspan.fast.FastgSpan.run_from_graphs", fail)
    with pytest.raises(RuntimeWarning, match="gBolt exited"):
        MolsMining(config).decompose([get_mol("CC")] * 2)


def test_lower_1_matches_python(config):
    smiles = ["CCO", "CCCO", "O"]
    mols = [get_mol(s) for s in smiles]
    results = {}
    for backend in ["fast-gspan", "python"]:
        runner = MolsMining(
            replace(config, method="raw", lower=1, upper=3, support=2, backend=backend)
        )
        result = runner.decompose(mols)
        fragments = runner.gspan_to_mols(result, smiles)
        assert all(m is not None for m in fragments)
        results[backend] = (result, {get_smiles(m) for m in fragments})

    fast, python = results["fast-gspan"][0], results["python"][0]
    assert results["fast-gspan"][1] == results["python"][1]
    assert len(fast.ggnx) == len(python.ggnx) == 5
    assert (fast._report_df["num_vert"] == 1).sum() == 2
    assert (python._report_df["num_vert"] == 1).sum() == 2

    node_match = nx.algorithms.isomorphism.categorical_node_match("label", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", None)
    for index, entry in enumerate(fast.ggnx):
        matches = [
            i
            for i, other in enumerate(python.ggnx)
            if nx.is_isomorphic(
                entry["graph"],
                other["graph"],
                node_match=node_match,
                edge_match=edge_match,
            )
        ]
        assert len(matches) == 1
        row = fast._report_df.iloc[index]
        reference = python._report_df.iloc[matches[0]]
        assert row.support == reference.support
        assert set(row.support_ids.split(",")) == set(reference.support_ids.split(","))


def test_high_projection_exceeding_500k_does_not_drop_patterns(config):
    smiles = "C" * 26 + "O"
    mol = get_mol(smiles)
    n_graphs = 12_000  # 25 C-C bonds * 2 directed projections * 12,000 = 600,000 > 500k
    runner = MolsMining(
        replace(
            config,
            method="raw",
            lower=2,
            upper=4,
            support=n_graphs,
            backend="fast-gspan",
        )
    )
    result = runner.decompose([mol] * n_graphs)
    fragments = {
        get_smiles(m) for m in runner.gspan_to_mols(result, [smiles] * n_graphs)
    }
    assert {"CC", "CCC", "CCCC"} <= fragments
    assert (result._report_df["support"] == n_graphs).all()
