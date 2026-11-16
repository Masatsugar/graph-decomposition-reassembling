"""New chemistry and policy scopes remain consumable by existing trace tools."""

import inspect
import json

import numpy as np
import pytest
import torch

from moldr import profiling
from moldr.e2e.env_v2 import MolEnvE2EV2
from moldr.e2e.model_v2 import PointerPolicyNetV2
from moldr.evaluation import value_lookahead
from moldr.gnn import MPNNEncoder
from moldr.scoring import ScoringService
from scripts.profile_viewer import _prepare_trace


def test_v2_trace_contains_smiles_chemistry_value_and_graph_breakdowns(tmp_path):
    torch.set_num_threads(1)
    profiling.enable(include_rllib=False, max_events=5000)
    try:
        env = MolEnvE2EV2(
            {
                "BUILDING_BLOCKS": ["CC", "CO"],
                "BASE_SMILES": "C",
                "SCORE_THRESHOLD": 2.0,
                "LENGTH": 20,
                "MAX_ATTACH": 4,
                "MAX_EPISODE_STEPS": 2,
                "SCORING": {"objective": "QED", "cache_size": 8},
                "REPRESENTATION": {"name": "mpnn", "max_atoms": 24, "max_bonds": 32},
            }
        )
        env.reset(seed=0)
        obs, *_ = env.step(0)
        choice = value_lookahead(env, lambda observation: 0.2)
        env.step(choice.action)

        encoder = MPNNEncoder(hidden=8, output_dim=4, layers=1)
        encoder(
            torch.from_numpy(obs["base_atoms"])[None],
            torch.from_numpy(obs["base_bonds"])[None],
            torch.from_numpy(obs["base_counts"])[None],
        )
        heads = PointerPolicyNetV2(obs_dim=4, hidden=8, candidate_value=True)
        arguments = (
            torch.zeros(1, 4),
            torch.tensor([[0.0, 1.0]]),
            torch.zeros(1, 4, 4),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.2, 1.0, 1.0]]),
        )
        heads(*arguments)
        heads.candidate_values(*arguments)
        snapshot = profiling.disable_snapshot()
    finally:
        profiling.disable_snapshot()
    events = [event for event in snapshot["traceEvents"] if event["ph"] == "X"]
    names = {event["name"] for event in events}
    assert {
        "MolEnvE2EV2.reset",
        "MolEnvE2EV2.step",
        "CandidateGenerator.generate",
        "prepare_local_pair",
        "LocalPair._build",
        "SMILES.parse",
        "SMILES.serialize",
        "SMILES.parse_score",
        "ScoringService.score",
        "objective.compute",
        "value_lookahead",
        "MPNNEncoder.forward",
        "PointerPolicyNetV2.candidate_values",
        "PointerPolicyNetV2.candidate_value_logits",
    } <= names
    score_events = [
        event for event in events if event["name"] == "ScoringService.score"
    ]
    assert (
        sum(event["args"]["requests"] for event in score_events) == env.scorer.requests
    )
    assert (
        sum(event["args"]["computations"] for event in score_events)
        == env.scorer.computations
    )
    assert sum(event["args"]["hits"] for event in score_events) == env.scorer.hits
    assert any(event["args"]["hits"] == 1 for event in score_events)
    assert all(0 <= event["args"]["self_us"] <= event["dur"] for event in events)
    assert snapshot["dropped_events"] == 0
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(snapshot))
    timeline = _prepare_trace(trace_path)
    assert {"smiles", "gnn", "candidate_value"} <= {
        entry["name"] for entry in timeline["categories"]
    }


@pytest.mark.parametrize(
    "base,fragment,config,local_preparation,local_build",
    [
        ("C", "CO", {}, True, True),
        ("C", "CO", {"bond_scan": "python"}, False, False),
        ("N->[Cu]", "CC", {}, True, False),
    ],
)
def test_candidate_trace_reports_the_builder_that_actually_runs(
    base, fragment, config, local_preparation, local_build
):
    from moldr.chemistry import CandidateGenerator
    from moldr.chemutils import get_mol

    molecules = get_mol(base), get_mol(fragment)
    profiling.disable_snapshot()
    generator = CandidateGenerator(config)
    reference = [candidate.smiles for candidate in generator.generate(*molecules)]
    reference_stats = dict(generator.stats)
    profiling.enable(include_rllib=False, max_events=1000)
    try:
        candidates = generator.generate(*molecules)
        snapshot = profiling.disable_snapshot()
    finally:
        profiling.disable_snapshot()
    assert [candidate.smiles for candidate in candidates] == reference
    assert generator.stats == reference_stats
    events = [event for event in snapshot["traceEvents"] if event["ph"] == "X"]
    names = [event["name"] for event in events]
    assert names.count("CandidateGenerator.generate") == 1
    assert names.count("prepare_local_pair") == int(local_preparation)
    assert names.count("prepare_pair") == int(not local_build)
    generated = generator.stats["generated"]
    assert generated > 0
    assert names.count("LocalPair._build") == (generated if local_build else 0)
    assert names.count("reconstruct_molecule") == (0 if local_build else generated)
    assert sum(
        name in {"merge_mol_from_node", "merge_mol_from_edge"} for name in names
    ) == (0 if local_build else generated)
    outer = next(
        event for event in events if event["name"] == "CandidateGenerator.generate"
    )
    assert outer["args"]["result_count"] == len(candidates)
    assert all(0 <= event["args"]["self_us"] <= event["dur"] for event in events)
    assert snapshot["dropped_events"] == 0


def test_local_builder_trace_disables_and_reenables_without_double_wrapping():
    from moldr import chemistry
    from moldr.chemutils import get_mol
    from moldr.core import reassemble_fast

    molecules = get_mol("C"), get_mol("CO")
    original_prepare = inspect.unwrap(reassemble_fast.prepare_local_pair)
    original_build = inspect.unwrap(reassemble_fast.LocalPair._build)
    profiling.enable(include_rllib=False, max_events=1000)
    try:
        wrapped_prepare = reassemble_fast.prepare_local_pair
        wrapped_build = reassemble_fast.LocalPair._build
        assert chemistry.prepare_local_pair is wrapped_prepare
        generator = chemistry.CandidateGenerator()
        first = [candidate.smiles for candidate in generator.generate(*molecules)]
        snapshot = profiling.disable_snapshot()
        disabled = [candidate.smiles for candidate in generator.generate(*molecules)]
        assert disabled == first
        assert (
            profiling.get_recorder().snapshot()["traceEvents"]
            == snapshot["traceEvents"]
        )
        assert inspect.unwrap(reassemble_fast.prepare_local_pair) is original_prepare
        assert inspect.unwrap(reassemble_fast.LocalPair._build) is original_build

        profiling.enable(include_rllib=False, max_events=1000)
        assert reassemble_fast.prepare_local_pair is wrapped_prepare
        assert chemistry.prepare_local_pair is wrapped_prepare
        assert reassemble_fast.LocalPair._build is wrapped_build
        assert [
            candidate.smiles for candidate in generator.generate(*molecules)
        ] == first
        resumed = profiling.disable_snapshot()
    finally:
        profiling.disable_snapshot()
    names = [event["name"] for event in resumed["traceEvents"] if event["ph"] == "X"]
    assert names.count("CandidateGenerator.generate") == 1
    assert names.count("prepare_local_pair") == 1
    assert names.count("LocalPair._build") == generator.stats["generated"]


def test_mol_score_trace_records_cache_requests_without_smiles_score_parse():
    from moldr.chemutils import get_mol

    profiling.enable(include_rllib=False, max_events=100)
    try:
        scorer = ScoringService("qed", input_mode="mol", cache_size=0)
        molecule = get_mol("CCO")
        first = scorer.score(molecule)
        second = scorer.score(molecule)
        snapshot = profiling.disable_snapshot()
    finally:
        profiling.disable_snapshot()
    np.testing.assert_equal(first, second)
    events = [event for event in snapshot["traceEvents"] if event["ph"] == "X"]
    assert "SMILES.parse_score" not in {event["name"] for event in events}
    score_events = [
        event for event in events if event["name"] == "ScoringService.score"
    ]
    assert all(event["args"]["input_mode"] == "mol" for event in score_events)
    assert all(
        event["args"]["requests"] == event["args"]["computations"] == 1
        for event in score_events
    )
