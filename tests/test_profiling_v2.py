"""New chemistry and policy scopes remain consumable by existing trace tools."""

import json

import numpy as np
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
        "prepare_pair",
        "merge_mol_from_node",
        "reconstruct_molecule",
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
