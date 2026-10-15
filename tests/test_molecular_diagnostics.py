"""Small chemical fixtures and identical-forward capture invariants."""

import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scripts.molecular_diagnostics import MolecularDiagnostics, _policy_entropy


class FakeMol:
    def __init__(self, count):
        self.count = count

    def GetNumAtoms(self):
        return self.count


class FakeGenerator:
    def __init__(self):
        self.config = {}
        self.calls = 0
        self.stats = {}

    def generate(self, base, fragment):
        self.calls += 1
        self.stats = {
            "generated": 3,
            "raw_unique": 3,
            "valid": 2,
            "duplicates": 1,
            "invalid": 0,
        }
        return ["CC", "CO"]


class FakePolicy:
    def __init__(self):
        self.rng = np.random.default_rng(18)
        self.draws = []

    def compute_single_action(self, observation, *, explore=False, full_fetch=False):
        # full_fetch must alter only the returned envelope, exactly as in the
        # RLlib Algorithm API. The same RNG draw/forward runs in both modes.
        self.draws.append(self.rng.random())
        attach = bool(observation["phase"][1])
        logits = (
            np.array([0.0, 1000.0, 1000.0]) if attach else np.log([0.25, 0.75, 1.0])
        )
        action = 0 if attach else 1
        output = action, [], {"action_dist_inputs": logits}
        return output if full_fetch else action


def _env():
    return SimpleNamespace(
        length=2,
        threshold=1.1,
        max_episode_steps=10,
        max_attach=1,
        _base_smiles="C",
        building_blocks=["CC", "CC", "CO"],
        catalog_smiles=["CC", "CC", "CO"],
        reward_mode="improvement",
        candidate_generator=FakeGenerator(),
        scorer=SimpleNamespace(objective=SimpleNamespace(name="QED"), _builtin=True),
    )


class FakeSampler:
    def __init__(self, *, preview=False):
        self.env, self.algo = _env(), FakePolicy()
        self.explore, self.selector, self.preview = False, "actor", preview

    def episode(self, *, seed=None, random=False):
        first = {"phase": [1, 0], "action_mask": [1, 1, 0]}
        second = {"phase": [0, 1], "action_mask": [1, 0, 0]}
        if not random:
            action = self.algo.compute_single_action(first, explore=self.explore)
            assert int(action[0] if isinstance(action, tuple) else action) == 1
        self.env.candidate_generator.generate(FakeMol(1), FakeMol(2))
        if self.preview:
            self.algo.compute_single_action(second, full_fetch=True)
        if not random:
            action = self.algo.compute_single_action(second, explore=self.explore)
            assert int(action[0] if isinstance(action, tuple) else action) == 0
        return _result(
            "CC",
            0.55,
            [
                {
                    "phase": "select_bb",
                    "chosen_bb": 1,
                    "chosen_bb_smiles": "CC",
                    "n_candidates_total": 2,
                    "n_candidates": 1,
                    "prev_smiles": "C",
                },
                {
                    "phase": "select_attach",
                    "chosen_attach": 0,
                    "prev_smiles": "C",
                    "gen_smiles": ["CC"],
                    "final_smiles": "CC",
                    "score": 0.55,
                    "improvement": -0.05,
                },
            ],
        )


def _result(smiles, score, infos, *, terminated=True, truncated=False):
    return SimpleNamespace(
        smiles=smiles,
        score=score,
        infos=infos,
        total_reward=0.3,
        steps=len(infos),
        terminated=terminated,
        truncated=truncated,
        scoring={"requests": 2, "computations": 2, "hits": 0},
    )


def test_capture_preserves_actions_rng_score_requests_and_single_forward():
    baseline, measured = FakeSampler(), FakeSampler()
    expected = baseline.episode(seed=14)
    diagnostics = MolecularDiagnostics()
    actual = diagnostics.capture(measured.episode)(seed=14)
    assert vars(actual) == vars(expected)
    assert measured.algo.draws == baseline.algo.draws
    assert (
        measured.algo.rng.bit_generator.state == baseline.algo.rng.bit_generator.state
    )
    assert (
        measured.env.candidate_generator.calls
        == baseline.env.candidate_generator.calls
        == 1
    )
    assert "generate" not in measured.env.candidate_generator.__dict__
    assert "compute_single_action" not in measured.algo.__dict__
    assert diagnostics.records[0]["scoring"] == actual.scoring
    assert diagnostics.records[0]["seed"] == 14
    assert "gen_smiles" not in json.dumps(diagnostics.records)
    assert "action_mask" not in json.dumps(diagnostics.records)
    assert diagnostics.records[0]["events"][0]["candidate_stats"]["generated"] == 3
    measured.env.candidate_generator.stats["generated"] = 999
    assert diagnostics.records[0]["events"][0]["candidate_stats"]["generated"] == 3
    entropy = diagnostics.records[0]["policy_entropy"]
    assert len(entropy) == 2
    expected_entropy = -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
    assert entropy[0]["entropy"] == pytest.approx(expected_entropy)
    assert entropy[0]["normalized_entropy"] == pytest.approx(
        expected_entropy / math.log(2)
    )
    assert entropy[1]["valid_actions"] == 1
    assert entropy[1]["normalized_entropy"] is None
    assert entropy[1]["entropy"] == 0


def test_unbound_capture_and_explicit_value_previews_are_not_decisions():
    sampler = FakeSampler(preview=True)
    diagnostics = MolecularDiagnostics()
    with patch.object(FakeSampler, "episode", diagnostics.capture(FakeSampler.episode)):
        sampler.episode(seed=8)
    assert len(sampler.algo.draws) == 3
    assert len(diagnostics.records[0]["policy_entropy"]) == 2
    assert diagnostics.env is sampler.env


def test_random_capture_does_not_request_policy_entropy():
    sampler, diagnostics = FakeSampler(), MolecularDiagnostics()
    diagnostics.capture(sampler.episode)(seed=5, random=True)
    assert sampler.algo.draws == []
    assert diagnostics.records[0]["policy_entropy"] == []
    assert diagnostics.records[0]["entropy_available"] is False


def test_policy_without_full_fetch_keeps_its_original_signature():
    sampler, diagnostics = FakeSampler(), MolecularDiagnostics()
    original = sampler.algo.compute_single_action

    def compute(observation, *, explore=False):
        return original(observation, explore=explore)

    sampler.algo.compute_single_action = compute
    diagnostics.capture(sampler.episode)(seed=3)
    assert len(sampler.algo.draws) == 2
    assert sampler.algo.compute_single_action is compute
    assert diagnostics.records[0]["entropy_available"] is False


def test_capture_restores_methods_after_error():
    sampler, diagnostics = FakeSampler(), MolecularDiagnostics()

    def broken(*, seed=None):
        sampler.env.candidate_generator.generate(FakeMol(1), FakeMol(2))
        raise RuntimeError("episode failed")

    with pytest.raises(RuntimeError, match="episode failed"):
        diagnostics = MolecularDiagnostics(sampler.env)
        diagnostics.capture(broken)(seed=5)
    assert "generate" not in sampler.env.candidate_generator.__dict__
    assert diagnostics.records == []


def test_one_diagnostics_instance_cannot_silently_mix_environments():
    diagnostics = MolecularDiagnostics()
    diagnostics.capture(FakeSampler().episode)(seed=0)
    with pytest.raises(ValueError, match="one MolecularDiagnostics"):
        diagnostics.capture(FakeSampler().episode)(seed=1)


def _fragment(fragment_id, fragment, previous, generated=4, retained=1):
    return {
        "phase": "select_bb",
        "chosen_bb": fragment_id,
        "chosen_bb_smiles": fragment,
        "prev_smiles": previous,
        "n_candidates_total": generated,
        "n_candidates": retained,
    }


def _attachment(previous, selected, score, improvement):
    return {
        "phase": "select_attach",
        "chosen_attach": 0,
        "prev_smiles": previous,
        "gen_smiles": [selected],
        "final_smiles": selected,
        "score": score,
        "improvement": improvement,
    }


def _append(diagnostics, result, seed):
    diagnostics._initialize(diagnostics.env)
    record = diagnostics._record(
        result, seed, False, SimpleNamespace(), [], [], False, False
    )
    diagnostics.records.append(record)


def test_offline_report_cap_denominator_recovery_coverage_and_qed(tmp_path):
    env = _env()
    diagnostics = MolecularDiagnostics(env)
    # These are synthetic trajectories for accounting, not claimed QED values.
    # Actual QED descriptors are calculated only by write().
    _append(
        diagnostics,
        _result(
            "CCO",
            0.7,
            [
                _fragment(0, "CC", "C"),
                _attachment("C", "CC", 0.4, -0.2),
                _fragment(2, "CO", "CC", 1, 1),
                _attachment("CC", "CCO", 0.7, 0.3),
            ],
        ),
        0,
    )
    _append(
        diagnostics,
        _result(
            "CCO",
            0.7,
            [
                _fragment(None, "CO", "CCO", 0, 0),
            ],
        ),
        1,
    )
    before = json.dumps(diagnostics.records, sort_keys=True)
    report = json.loads(diagnostics.write(tmp_path).read_text())
    assert json.dumps(diagnostics.records, sort_keys=True) == before
    summary = report["summary"]
    assert summary["episode_count"] == 2
    assert summary["fragment_choices"] == 3  # Includes missing ID and dead end.
    assert summary["capped_choices"] == 1
    assert summary["capped_choice_fraction"] == pytest.approx(1 / 3)
    assert summary["selected_merge_kinds"] == {"node": 2}
    assert (
        summary["negative_transitions"]
        == summary["recovered_negative_transitions"]
        == 1
    )
    assert summary["end_reasons"] == {"atom_limit": 1, "no_attachment_candidates": 1}
    assert summary["score_threshold_attained_episodes"] == 0
    assert summary["fragment_row_coverage"] == pytest.approx(2 / 3)
    assert summary["catalog_row_count"] == 3
    assert summary["catalog_unique_canonical_count"] == 2
    assert summary["canonical_fragment_coverage"] == 1
    assert summary["unique_molecules"] == 1
    assert summary["top1_fraction_of_valid"] == 1
    assert summary["acyclic_fraction_of_valid"] == 1
    assert summary["top1_scaffold_fraction_of_cyclic"] is None
    molecule = report["molecules"][0]
    assert sum(
        molecule["qed_weighted_negative_log_contribution"].values()
    ) == pytest.approx(-math.log(molecule["qed"]))
    assert set(molecule["qed_properties"]) == {
        "MW",
        "ALOGP",
        "HBA",
        "HBD",
        "PSA",
        "ROTB",
        "AROM",
        "ALERTS",
    }
    assert report["cap_audits"]["score_computations"] == 0
    assert env.candidate_generator.calls == 0


def test_edge_classification_ring_scaffold_and_truncated_phase_a(tmp_path):
    env = _env()
    env.building_blocks = env.catalog_smiles = ["C1CC1"]
    env.max_episode_steps = 1
    diagnostics = MolecularDiagnostics(env)
    _append(
        diagnostics,
        _result(
            "C1C2CC12",
            0.5,
            [
                _fragment(0, "C1CC1", "C1CC1", 1, 1),
                _attachment("C1CC1", "C1C2CC12", 0.5, 0.1),
            ],
        ),
        0,
    )
    _append(
        diagnostics,
        _result(
            "C",
            0.4,
            [_fragment(0, "C1CC1", "C", 1, 1)],
            terminated=False,
            truncated=True,
        ),
        1,
    )
    report = json.loads(diagnostics.write(tmp_path).read_text())
    assert report["summary"]["selected_merge_kinds"] == {"edge": 1}
    assert report["summary"]["nonempty_murcko_scaffolds"] == 1
    assert report["summary"]["top1_scaffold_fraction_of_cyclic"] == 1
    assert report["episodes"][1]["trajectory"]["molecular_updates"] == 0
    assert report["episodes"][1]["end_reason"] == "time_limit"


def test_best_visited_reports_selected_intermediate_and_initial_without_rescoring(
    tmp_path,
):
    diagnostics = MolecularDiagnostics(_env())
    _append(
        diagnostics,
        _result(
            "CCO",
            0.5,
            [
                _fragment(0, "CC", "C", 1, 1),
                _attachment("C", "CC", 0.8, 0.2),
                _fragment(2, "CO", "CC", 1, 1),
                _attachment("CC", "CCO", 0.5, -0.3),
            ],
        ),
        0,
    )
    _append(
        diagnostics,
        _result(
            "CC",
            0.5,
            [
                _fragment(0, "CC", "C", 1, 1),
                _attachment("C", "CC", 0.5, -0.1),
            ],
        ),
        1,
    )
    intermediate, initial = diagnostics.records
    assert intermediate["best_visited_smiles"] == "CC"
    assert intermediate["best_visited_score"] == 0.8
    assert intermediate["best_visited_step"] == 2
    assert intermediate["final_gap_from_best_visited"] == pytest.approx(0.3)
    assert initial["best_visited_smiles"] == "C"
    assert initial["best_visited_step"] == 0
    assert initial["best_visited_score"] == pytest.approx(0.6)
    assert initial["final_gap_from_best_visited"] == pytest.approx(0.1)
    assert diagnostics.env.candidate_generator.calls == 0
    report = json.loads(diagnostics.write(tmp_path).read_text())
    assert report["summary"]["episodes_final_below_best_visited"] == 2
    assert report["summary"]["final_gap_from_best_visited"]["mean"] == pytest.approx(
        0.2
    )
    assert report["episodes"][0]["score"] == 0.5
    assert report["episodes"][0]["scoring"]["computations"] == 2


def test_optional_cap_audit_is_explicit_offline_work(tmp_path):
    env = _env()
    diagnostics = MolecularDiagnostics(env, cap_audit_limit=1)
    _append(
        diagnostics,
        _result(
            "CCO",
            0.5,
            [
                _fragment(0, "CCO", "CCO", 6, 1),
                _attachment("CCO", "CCO", 0.5, 0.0),
            ],
        ),
        0,
    )
    assert env.candidate_generator.calls == 0
    report = json.loads(diagnostics.write(tmp_path).read_text())
    assert env.candidate_generator.calls == 0
    assert report["episodes"][0]["scoring"]["computations"] == 2
    audit = report["cap_audits"]
    assert audit["enabled"] is True
    assert audit["normal_environment_score_counters_affected"] is False
    assert audit["score_computations"] > 1
    assert len(audit["pairs"]) == 1
    assert audit["pairs"][0]["retained"] == 1
    assert audit["pairs"][0]["discarded"] > 0


def test_entropy_unavailable_for_missing_or_mismatched_logits():
    assert _policy_entropy({}, 0) is None
    assert _policy_entropy({}, (0, [], {})) is None
    assert (
        _policy_entropy({"action_mask": [1]}, (0, [], {"action_dist_inputs": [0, 0]}))
        is None
    )
