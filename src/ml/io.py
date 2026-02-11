from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil


def create_run_dir(root: str | Path, prefix: str = "rf") -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_path / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_model(model, model_path: str | Path) -> Path:
    import joblib

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(model_path: str | Path):
    import joblib

    return joblib.load(model_path)


def save_json(data: dict, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return out


def copy_config(config_path: str | Path, dst_path: str | Path) -> Path:
    out = Path(dst_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out)
    return out
