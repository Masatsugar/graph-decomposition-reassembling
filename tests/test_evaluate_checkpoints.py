"""Post-training plans keep seed budgets and independent checkpoint groups."""

import json

import pytest

from scripts import evaluate_checkpoints


def _sweep(tmp_path):
    jobs = []
    for configuration in evaluate_checkpoints.MAIN_CONFIGURATIONS:
        for seed in (0, 1, 2):
            name = f"{configuration}_seed{seed}_workers8"
            run = tmp_path / name
            (run / "checkpoint").mkdir(parents=True)
            (run / "checkpoint" / "algorithm_state.pkl").write_bytes(b"checkpoint")
            (run / "last_checkpoint.json").write_text(
                json.dumps({"path": "checkpoint"})
            )
            jobs.append({"name": name, "status": "complete", "directory": str(run)})
    (tmp_path / "sweep.json").write_text(json.dumps({"jobs": jobs}))
    return jobs


def test_plan_keeps_identical_seed_budget_and_blocks_during_speed_measurement(tmp_path):
    jobs = _sweep(tmp_path)
    plan = evaluate_checkpoints.make_plan(tmp_path, episodes=3, evaluation_seed=10)
    assert plan["ready_checkpoints"] == plan["total_checkpoints"] == 15
    assert plan["evaluation_seeds"] == [10, 11, 12]
    assert plan["num_workers"] == 0 and plan["num_gpus"] == 1.0
    assert plan["explore"] is True and plan["training"] is False
    jobs.append({"name": "map_string_seed0_workers8", "status": "running"})
    (tmp_path / "sweep.json").write_text(json.dumps({"jobs": jobs}))
    with pytest.raises(ValueError, match="speed sweep"):
        evaluate_checkpoints.evaluate_sweep(tmp_path, tmp_path / "posthoc")


def test_summary_uses_checkpoint_seed_variance_and_nullable_diversity():
    rows = [
        {
            "configuration": "mol2vec_actor",
            "episodes": 3,
            "mean_score": score,
            "fingerprint_diversity": None,
        }
        for score in (0.2, 0.4, 0.6)
    ]
    group = evaluate_checkpoints.summarize(rows)[0]
    assert group["n"] == 3 and group["total_episodes"] == 9
    assert group["metrics"]["mean_score"]["mean"] == pytest.approx(0.4)
    assert group["metrics"]["mean_score"]["sample_sd"] == pytest.approx(0.2)
    assert group["metrics"]["fingerprint_diversity"] == {
        "mean": None,
        "sample_sd": None,
        "n": 0,
    }
