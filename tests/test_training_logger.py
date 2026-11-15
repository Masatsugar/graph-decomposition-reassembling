import csv
import json

import numpy as np

from moldr.utils import TrainingLogger, custom_log_creator


def test_training_logger_records_results_and_preserves_csv_schema_on_resume(tmp_path):
    logger = TrainingLogger({"array": np.array([1, 2]), "type": int}, tmp_path)
    logger.on_result(
        {
            "training_iteration": 1,
            "config": {"hidden": 32},
            "learner": {"loss": np.float32(0.5)},
            "values": np.array([1, 2]),
        }
    )
    logger.close()
    logger = TrainingLogger({"hidden": 32}, tmp_path)
    logger.on_result(
        {"training_iteration": 2, "learner": {"loss": 0.25}, "new_metric": 9}
    )
    logger.close()
    logger.close()

    records = [
        json.loads(line) for line in (tmp_path / "result.json").read_text().splitlines()
    ]
    assert records[0]["learner"]["loss"] == 0.5
    assert records[0]["values"] == [1, 2]
    assert records[1]["new_metric"] == 9
    with (tmp_path / "progress.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[1]["learner/loss"] == "0.25"
    assert "config/hidden" not in reader.fieldnames
    assert "new_metric" not in reader.fieldnames
    assert json.loads((tmp_path / "params.json").read_text()) == {"hidden": 32}


def test_custom_logger_records_tensorboard_and_serializes_custom_config(tmp_path):
    from ray import cloudpickle

    logger = custom_log_creator(tmp_path, "training")({"score": lambda x: x + 1})
    logger.on_result({"training_iteration": 1, "learner": {"loss": np.float32(0.5)}})
    logger.close()

    directory = next(tmp_path.iterdir())
    assert list(directory.glob("events.out.tfevents.*"))
    with (directory / "params.pkl").open("rb") as stream:
        assert cloudpickle.load(stream)["score"](2) == 3
    assert "score" in json.loads((directory / "params.json").read_text())
