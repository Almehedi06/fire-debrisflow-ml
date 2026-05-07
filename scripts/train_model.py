from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train raster ML model from aligned TIFFs.")
    parser.add_argument("--config", default="config/ml_rf.yaml", help="Training config path.")
    parser.add_argument("--data-dir", default=None, help="Override raster data directory.")
    parser.add_argument("--target", default=None, help="Override target TIFF filename.")
    parser.add_argument("--model-root", default=None, help="Override model output root directory.")
    parser.add_argument(
        "--model-type",
        choices=["rf", "xgb"],
        default=None,
        help="Override model type from config.",
    )
    return parser.parse_args()


def _base_model_params(model_cfg: dict) -> dict:
    excluded = {"type", "search_grid"}
    return {k: v for k, v in model_cfg.items() if k not in excluded}


def _expand_model_param_grid(model_cfg: dict) -> list[dict]:
    base = _base_model_params(model_cfg)
    grid = model_cfg.get("search_grid") or {}
    if not grid:
        return [base]

    keys = list(grid.keys())
    value_lists: list[list] = []
    for key in keys:
        values = grid[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"search_grid[{key!r}] must be a non-empty list.")
        value_lists.append(values)

    candidates: list[dict] = []
    for combo in itertools.product(*value_lists):
        candidate = base.copy()
        candidate.update(dict(zip(keys, combo)))
        candidates.append(candidate)
    return candidates


def _fit_model(model_type: str, x_train, y_train, params: dict):
    from ml.train import train_random_forest_regressor, train_xgboost_regressor

    if model_type == "rf":
        return train_random_forest_regressor(x_train, y_train, **params)
    if model_type == "xgb":
        return train_xgboost_regressor(x_train, y_train, **params)
    raise ValueError(f"Unsupported model type: {model_type}")


def _select_best_candidate(results: list[dict], metric: str) -> dict:
    if metric not in {"rmse", "mae", "r2"}:
        raise ValueError(f"Unsupported selection_metric: {metric}")
    key = f"mean_{metric}"
    reverse = metric == "r2"
    return sorted(results, key=lambda row: row[key], reverse=reverse)[0]


def main() -> None:
    args = _parse_args()

    from ml.dataset import build_xy_from_rasters, discover_feature_paths
    from ml.evaluate import regression_metrics
    from ml.io import copy_config, create_run_dir, save_json, save_model
    from ml.split import (
        build_spatial_block_groups,
        random_train_test_split,
        spatial_block_kfold_indices,
        spatial_block_train_test_split,
    )

    cfg = _load_yaml(args.config)

    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})
    model_cfg = cfg.get("model", {})
    output_cfg = cfg.get("output", {})
    model_type = (args.model_type or model_cfg.get("type", "rf")).lower()

    data_dir = Path(args.data_dir or data_cfg.get("dir"))
    target_name = args.target or data_cfg.get("target_name", "dem_diff.tif")
    target_path = data_dir / target_name
    if not target_path.exists():
        raise FileNotFoundError(f"Target raster not found: {target_path}")

    exclude = data_cfg.get("exclude_names", ["dem_pre.tif", "dem_post.tif"])
    exclude_contains = data_cfg.get("exclude_contains", ["pred"])
    feature_paths = discover_feature_paths(
        data_dir=data_dir,
        target_name=target_name,
        include_glob=data_cfg.get("include_glob", "*.tif"),
        include_names=data_cfg.get("include_names"),
        exclude_names=exclude,
        exclude_contains=exclude_contains,
    )

    bundle = build_xy_from_rasters(feature_paths, target_path)
    split_method = str(split_cfg.get("method", "random")).lower()
    random_state = int(split_cfg.get("random_state", 42))
    candidate_params = _expand_model_param_grid(model_cfg)
    if split_method == "random" and len(candidate_params) > 1:
        raise ValueError(
            "model.search_grid is only supported with split.method=spatial_block_cv."
        )

    cv_results: list[dict] | None = None
    best_params = candidate_params[0]

    if split_method == "random":
        split = random_train_test_split(
            bundle["X"],
            bundle["y"],
            test_size=float(split_cfg.get("test_size", 0.2)),
            random_state=random_state,
        )
        model = _fit_model(model_type, split["X_train"], split["y_train"], best_params)
        y_pred = model.predict(split["X_test"])
        metrics = regression_metrics(split["y_test"], y_pred)
        split_summary = {
            "method": "random",
            "test_size": float(split_cfg.get("test_size", 0.2)),
            "random_state": random_state,
            "n_train_samples": int(split["X_train"].shape[0]),
            "n_test_samples": int(split["X_test"].shape[0]),
        }
    elif split_method == "spatial_block_cv":
        block_size = int(split_cfg.get("block_size", 64))
        n_folds = int(split_cfg.get("n_folds", 5))
        selection_metric = str(split_cfg.get("selection_metric", "rmse")).lower()

        groups = build_spatial_block_groups(bundle["valid_mask"], block_size=block_size)
        split = spatial_block_train_test_split(
            bundle["X"],
            bundle["y"],
            groups=groups,
            test_size=float(split_cfg.get("test_size", 0.2)),
            random_state=random_state,
        )
        folds = spatial_block_kfold_indices(
            split["groups_train"],
            n_splits=n_folds,
            random_state=random_state,
        )

        cv_results = []
        for candidate in candidate_params:
            fold_metrics: list[dict] = []
            for fold_id, fold in enumerate(folds, start=1):
                fold_model = _fit_model(
                    model_type,
                    split["X_train"][fold["train_idx"]],
                    split["y_train"][fold["train_idx"]],
                    candidate,
                )
                y_val_pred = fold_model.predict(split["X_train"][fold["val_idx"]])
                fold_metric = regression_metrics(split["y_train"][fold["val_idx"]], y_val_pred)
                fold_metrics.append({"fold": fold_id, **fold_metric})

            cv_results.append(
                {
                    "params": candidate,
                    "fold_metrics": fold_metrics,
                    "mean_r2": float(np.mean([row["r2"] for row in fold_metrics])),
                    "mean_rmse": float(np.mean([row["rmse"] for row in fold_metrics])),
                    "mean_mae": float(np.mean([row["mae"] for row in fold_metrics])),
                }
            )

        best_cv = _select_best_candidate(cv_results, selection_metric)
        best_params = best_cv["params"]
        model = _fit_model(model_type, split["X_train"], split["y_train"], best_params)
        y_pred = model.predict(split["X_test"])
        metrics = regression_metrics(split["y_test"], y_pred)
        split_summary = {
            "method": "spatial_block_cv",
            "block_size": block_size,
            "n_folds": n_folds,
            "selection_metric": selection_metric,
            "random_state": random_state,
            "n_total_groups": int(np.unique(groups).size),
            "n_train_groups": int(np.unique(split["groups_train"]).size),
            "n_test_groups": int(np.unique(split["groups_test"]).size),
            "n_train_samples": int(split["X_train"].shape[0]),
            "n_test_samples": int(split["X_test"].shape[0]),
            "test_group_fraction": float(split_cfg.get("test_size", 0.2)),
        }
    else:
        raise ValueError(f"Unsupported split.method: {split_method}")

    model_root = Path(args.model_root or output_cfg.get("model_dir", f"models/{model_type}"))
    run_dir = create_run_dir(model_root, prefix=output_cfg.get("run_prefix", model_type))

    model_path = save_model(model, run_dir / "model.joblib")
    save_json(metrics, run_dir / "metrics.json")
    save_json(split_summary, run_dir / "split_summary.json")
    save_json(
        {
            "target_file": target_name,
            "model_type": model_type,
            "feature_files": bundle["feature_files"],
            "feature_names": bundle["feature_names"],
            "n_features": len(bundle["feature_files"]),
            "n_valid_pixels": int(bundle["X"].shape[0]),
            "split_method": split_method,
        },
        run_dir / "feature_order.json",
    )
    if cv_results is not None:
        save_json(
            {
                "selection_metric": split_summary["selection_metric"],
                "best_params": best_params,
                "candidates": cv_results,
            },
            run_dir / "cv_results.json",
        )
    save_json(cfg, run_dir / "resolved_train_config.json")
    copy_config(args.config, run_dir / "train_config.yaml")

    print("Saved model:", model_path)
    print("Saved metrics:", run_dir / "metrics.json")
    print("Saved split:", run_dir / "split_summary.json")
    print("Saved features:", run_dir / "feature_order.json")
    if cv_results is not None:
        print("Saved CV results:", run_dir / "cv_results.json")
        print("Best params:", best_params)
    print("Model type:", model_type)
    print("Test R2:", metrics["r2"])
    print("Test RMSE:", metrics["rmse"])


if __name__ == "__main__":
    main()
