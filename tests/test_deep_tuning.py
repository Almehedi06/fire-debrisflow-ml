from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep.tuning import expand_search_space, select_best_candidate, validate_unet_candidate_config


def _base_cfg() -> dict:
    return {
        "split": {
            "train_size": 0.6,
            "val_size": 0.2,
            "test_size": 0.2,
            "block_size": 128,
            "random_state": 42,
        },
        "model": {"type": "unet", "base_channels": 16},
        "training": {
            "patch_size": 64,
            "stride": 16,
            "min_valid_fraction": 0.1,
            "batch_size": 8,
            "epochs": 10,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "loss": "huber",
        },
        "tuning": {
            "selection_metric": "rmse",
            "search_space": {
                "model.base_channels": [16, 32],
                "training.learning_rate": [1e-3, 3e-4],
            },
        },
    }


def test_expand_search_space_preserves_overrides() -> None:
    candidates = expand_search_space(_base_cfg())
    assert len(candidates) == 4
    combos = {tuple(sorted(row["overrides"].items())) for row in candidates}
    assert len(combos) == 4


def test_validate_unet_candidate_config_rejects_bad_patch_geometry() -> None:
    cfg = _base_cfg()
    cfg["training"]["patch_size"] = 30
    with pytest.raises(ValueError, match="divisible by 8"):
        validate_unet_candidate_config(cfg)


def test_validate_unet_candidate_config_rejects_small_block() -> None:
    cfg = _base_cfg()
    cfg["split"]["block_size"] = 32
    with pytest.raises(ValueError, match="block_size"):
        validate_unet_candidate_config(cfg)


def test_select_best_candidate_respects_metric_direction() -> None:
    rows = [
        {"candidate_id": 1, "best_val_rmse": 0.5, "best_val_mae": 0.3, "best_val_r2": 0.1},
        {"candidate_id": 2, "best_val_rmse": 0.4, "best_val_mae": 0.35, "best_val_r2": 0.2},
    ]
    assert select_best_candidate(rows, "rmse")["candidate_id"] == 2
    assert select_best_candidate(rows, "mae")["candidate_id"] == 1
    assert select_best_candidate(rows, "r2")["candidate_id"] == 2
