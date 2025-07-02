import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

from ray.tune.logger import UnifiedLogger, DEFAULT_LOGGERS


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
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = f"{custom_str}_{timestr}"

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        # logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        logdir = os.path.join(custom_path, logdir_prefix)
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(config, logdir, loggers=DEFAULT_LOGGERS)

    return logger_creator
