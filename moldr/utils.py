import csv
import json
import numbers
import os
import pickle
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


def _json_default(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _flatten_result(result, prefix=""):
    flattened = {}
    for key, value in result.items():
        key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_result(value, key))
        else:
            flattened[key] = value
    return flattened


class TrainingLogger:
    """Record standalone RLlib training results without Tune's removed Logger API."""

    def __init__(self, config, logdir, *, tensorboard=False):
        self.logdir = str(logdir)
        directory = Path(logdir)
        directory.mkdir(parents=True, exist_ok=True)
        self._tensorboard = None
        if tensorboard:
            from tensorboardX import SummaryWriter

            self._tensorboard = SummaryWriter(self.logdir)
        self.update_config(config)
        self._json_file = (directory / "result.json").open("a", encoding="utf-8")
        csv_path = directory / "progress.csv"
        self._fieldnames = None
        if csv_path.exists() and csv_path.stat().st_size:
            with csv_path.open(newline="", encoding="utf-8") as stream:
                self._fieldnames = next(csv.reader(stream))
        self._csv_file = csv_path.open("a", newline="", encoding="utf-8")
        self._csv_writer = None

    def update_config(self, config):
        self.config = config.to_dict() if hasattr(config, "to_dict") else config
        directory = Path(self.logdir)
        with (directory / "params.json").open("w", encoding="utf-8") as stream:
            json.dump(self.config, stream, default=_json_default, indent=2)
        from ray import cloudpickle

        with (directory / "params.pkl").open("wb") as stream:
            cloudpickle.dump(self.config, stream)

    def on_result(self, result):
        json.dump(result, self._json_file, default=_json_default)
        self._json_file.write("\n")
        flattened = _flatten_result({k: v for k, v in result.items() if k != "config"})
        if self._csv_writer is None:
            write_header = self._fieldnames is None
            self._fieldnames = self._fieldnames or list(flattened)
            self._csv_writer = csv.DictWriter(
                self._csv_file, self._fieldnames, extrasaction="ignore"
            )
            if write_header:
                self._csv_writer.writeheader()
        self._csv_writer.writerow(flattened)
        if self._tensorboard is not None:
            step = result.get("timesteps_total") or result.get("training_iteration", 0)
            for key, value in flattened.items():
                if isinstance(value, numbers.Real):
                    self._tensorboard.add_scalar(f"ray/tune/{key}", value, step)
        self.flush()

    def flush(self):
        self._json_file.flush()
        self._csv_file.flush()
        if self._tensorboard is not None:
            self._tensorboard.flush()

    def close(self):
        self._json_file.close()
        self._csv_file.close()
        if self._tensorboard is not None:
            self._tensorboard.close()
            self._tensorboard = None


def save(fpath, obj):
    if not isinstance(fpath, Path):
        fpath = Path(fpath)
    if not fpath.is_dir():
        fpath.parent.mkdir(parents=True, exist_ok=True)
    with fpath.open("wb") as f:
        pickle.dump(obj, f)


def load(fpath):
    if not isinstance(fpath, Path):
        fpath = Path(fpath)
    with fpath.open("rb") as f:
        obj = pickle.load(f)
    return obj


def custom_log_creator(custom_path, custom_str):
    timestr = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = f"{custom_str}_{timestr}"

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = os.path.join(custom_path, logdir_prefix)
        os.makedirs(logdir, exist_ok=True)
        return TrainingLogger(config, logdir, tensorboard=True)

    return logger_creator
