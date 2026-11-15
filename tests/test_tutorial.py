"""The tutorial exercises the same legacy transitions it documents."""

import csv
import json

import pytest

from examples import tutorial
from moldr.chemistry import CandidateGenerator
from moldr.chemutils import get_mol
from moldr.scoring import get_objective


@pytest.mark.parametrize("command", ["demo", "train"])
def test_tutorial_defaults_to_legacy_reward_and_termination(command, tmp_path):
    args = tutorial.parser().parse_args(
        [command, "--objective", "QED", "--output", str(tmp_path)]
    )
    config, _, _ = tutorial.make_config(args)
    assert config.environment.mode == "legacy"
    assert config.environment.reward == "sparse"
    assert config.environment.terminate_on_score_decrease
    assert config.training.num_gpus == 0.0


@pytest.mark.parametrize("num_gpus", [0.0, 0.5, 1.0])
def test_tutorial_training_gpu_resources_reach_shared_configuration(num_gpus, tmp_path):
    args = tutorial.parser().parse_args(
        [
            "train",
            "--objective",
            "QED",
            "--output",
            str(tmp_path),
            f"--num-gpus={num_gpus}",
        ]
    )
    config, _, _ = tutorial.make_config(args)
    assert config.training.num_gpus == num_gpus
    assert config.training.num_workers == 0
    assert config.environment.mode == "legacy"


@pytest.mark.parametrize("num_gpus", ["-1", "nan", "inf", "-inf", "2"])
def test_tutorial_rejects_invalid_gpu_resources_before_training(num_gpus, tmp_path):
    args = tutorial.parser().parse_args(
        [
            "train",
            "--objective",
            "QED",
            "--output",
            str(tmp_path),
            f"--num-gpus={num_gpus}",
        ]
    )
    with pytest.raises(ValueError, match="finite|nonnegative|one learner GPU"):
        tutorial.make_config(args)


@pytest.mark.parametrize("reward", ["sparse", "improvement"])
def test_legacy_demo_records_greedy_attachment_and_actual_rewards(tmp_path, reward):
    args = tutorial.parser().parse_args(
        [
            "demo",
            "--objective",
            "QED",
            "--max-steps",
            "2",
            "--output",
            str(tmp_path),
            "--reward",
            reward,
        ]
    )
    config, blocks, catalog = tutorial.make_config(args)
    directory = tutorial.run_demo(config, blocks, catalog)
    with (directory / "trajectory.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    objective = get_objective(config.scoring.objective).smiles_function
    generator = CandidateGenerator()
    previous = config.environment.base_smiles
    for row in rows:
        assert row["phase"] == "reassemble"
        candidates = generator.generate(
            get_mol(previous), get_mol(blocks[int(row["action"])])
        )
        score = float(row["score"])
        if candidates:
            assert score == pytest.approx(
                max(objective(candidate.smiles) for candidate in candidates)
            )
        else:
            assert row["smiles"] == previous
        assert score == pytest.approx(objective(row["smiles"]))
        expected_reward = (
            score * config.environment.final_weight
            if row["terminated"] == "True"
            else 0.0
        )
        if reward == "improvement" and candidates:
            expected_reward += score - objective(previous)
        assert float(row["reward"]) == pytest.approx(expected_reward)
        previous = row["smiles"]
    assert rows[-1]["terminated"] == "True" or rows[-1]["truncated"] == "True"
    molecule = json.loads((directory / "molecules.json").read_text())[0]
    assert molecule["policy"] == "uniform_building_blocks_greedy_attachment"
    assert molecule["smiles"] == rows[-1]["smiles"]
    assert molecule["score"] == pytest.approx(float(rows[-1]["score"]))


def test_explicit_e2e_demo_still_uses_two_decision_phases(tmp_path):
    args = tutorial.parser().parse_args(
        [
            "demo",
            "--mode",
            "e2e",
            "--objective",
            "QED",
            "--max-steps",
            "2",
            "--output",
            str(tmp_path),
        ]
    )
    config, blocks, catalog = tutorial.make_config(args)
    assert config.environment.reward == "improvement"
    assert not config.environment.terminate_on_score_decrease
    directory = tutorial.run_demo(config, blocks, catalog)
    with (directory / "trajectory.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["phase"] for row in rows] == ["select_bb", "select_attach"]
    assert rows[-1]["truncated"] == "True"
