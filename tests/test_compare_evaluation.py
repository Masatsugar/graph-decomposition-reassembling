"""Evaluation comparisons retain preview cost and identical seed budgets."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from moldr.e2e.env_v2 import MolEnvE2EV2
from moldr.experiment import ExperimentConfig
from moldr.provenance import write_json
from moldr.sampling import Sampler
from scripts import compare_evaluation


def _environment():
    return MolEnvE2EV2(
        {
            "BUILDING_BLOCKS": ["CC"],
            "BASE_SMILES": "C",
            "LENGTH": 20,
            "MAX_EPISODE_STEPS": 2,
            "MAX_ATTACH": 4,
            "SCORE_THRESHOLD": 2.0,
            "SCORING": {"objective": "QED", "cache_size": 16},
            "REPRESENTATION": {"name": "mpnn"},
        }
    )


def test_episode_cost_includes_preview_and_real_selected_score():
    sampler = Sampler(
        _environment(),
        SimpleNamespace(compute_single_action=lambda obs: 0),
        selector="value_lookahead",
        value_fn=lambda obs: 0.4,
    )
    episode = sampler.episode(seed=0)
    metrics = compare_evaluation.episode_metrics(episode, seconds=1.0)
    assert metrics["lookahead_previews"] == 1
    assert metrics["lookahead_selected_attachment_requests"] == 1
    assert metrics["score_requests"] == 3  # reset, preview, selected real step
    assert metrics["score_computations"] == 2
    assert metrics["score_hits"] == 1
    assert metrics["lookahead_transitions"] == [
        {"candidates": 1, "actual_score_requests": 2}
    ]
    episode.infos[-1]["scoring"]["requests"] -= 1
    with pytest.raises(RuntimeError, match=r"expected K\+1"):
        compare_evaluation.episode_metrics(episode, seconds=1.0)


def test_comparison_restores_same_checkpoint_and_preserves_seed_budgets(
    monkeypatch, tmp_path
):
    config = ExperimentConfig()
    config.fragments.smiles = ["CC"]
    config.environment.max_episode_steps = 2
    config.environment.score_threshold = 2.0
    config.representation.name = "mpnn"
    calls = []
    monkeypatch.setattr(
        compare_evaluation, "load_saved_config", lambda path: (config, tmp_path)
    )
    monkeypatch.setattr(
        compare_evaluation,
        "resolve_config",
        lambda value: (value.validate(), ["CC"], ["CC"]),
    )
    monkeypatch.setattr(compare_evaluation, "checkpoint_path", lambda path: str(path))

    def evaluate(resolved, checkpoint):
        calls.append(
            (
                checkpoint,
                resolved.training.seed,
                resolved.evaluation.selector,
                resolved.training.num_workers,
                resolved.training.num_gpus,
                resolved.evaluation.episodes,
            )
        )
        env = _environment()
        sampler = Sampler(
            env,
            SimpleNamespace(compute_single_action=lambda obs: 0),
            selector=resolved.evaluation.selector,
            value_fn=lambda obs: 0.4,
        )
        for index in range(resolved.evaluation.episodes):
            sampler.episode(seed=resolved.training.seed + index)
        directory = Path(resolved.output.directory) / resolved.output.run_name
        directory.mkdir(parents=True)
        write_json(directory / "evaluation.json", {"scoring": env.scorer.stats})
        return directory

    monkeypatch.setattr(compare_evaluation, "run_evaluation", evaluate)
    checkpoint = tmp_path / "saved-checkpoint"
    directory = compare_evaluation.compare_checkpoint(
        checkpoint, tmp_path / "comparison", episodes=2
    )
    report = json.loads((directory / "comparison.json").read_text())
    assert len(calls) == 6
    assert all(call[0] == str(checkpoint) and call[3:] == (0, 0.0, 2) for call in calls)
    for seed in (0, 1, 2):
        pair = [run for run in report["runs"] if run["starting_seed"] == seed]
        assert len(pair) == 2
        assert pair[0]["episode_seeds"] == pair[1]["episode_seeds"] == [seed, seed + 1]
    assert (
        report["summary"][1]["score_requests"] > report["summary"][0]["score_requests"]
    )
    assert report["summary"][1]["lookahead_previews"] == 6
    assert (directory / "comparison.png").is_file()
    assert (directory / "comparison.svg").is_file()
