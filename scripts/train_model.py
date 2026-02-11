from __future__ import annotations

import argparse
from pathlib import Path
import sys

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


def main() -> None:
    args = _parse_args()

    from ml.dataset import build_xy_from_rasters, discover_feature_paths
    from ml.evaluate import regression_metrics
    from ml.io import copy_config, create_run_dir, save_json, save_model
    from ml.split import random_train_test_split
    from ml.train import train_random_forest_regressor, train_xgboost_regressor

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
    split = random_train_test_split(
        bundle["X"],
        bundle["y"],
        test_size=float(split_cfg.get("test_size", 0.2)),
        random_state=int(split_cfg.get("random_state", 42)),
    )

    if model_type == "rf":
        model = train_random_forest_regressor(
            split["X_train"],
            split["y_train"],
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            max_depth=model_cfg.get("max_depth"),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
            random_state=int(model_cfg.get("random_state", 42)),
            n_jobs=int(model_cfg.get("n_jobs", -1)),
        )
    elif model_type == "xgb":
        model = train_xgboost_regressor(
            split["X_train"],
            split["y_train"],
            n_estimators=int(model_cfg.get("n_estimators", 500)),
            max_depth=int(model_cfg.get("max_depth", 8)),
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            subsample=float(model_cfg.get("subsample", 0.8)),
            colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
            reg_alpha=float(model_cfg.get("reg_alpha", 0.0)),
            reg_lambda=float(model_cfg.get("reg_lambda", 1.0)),
            random_state=int(model_cfg.get("random_state", 42)),
            n_jobs=int(model_cfg.get("n_jobs", -1)),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    y_pred = model.predict(split["X_test"])
    metrics = regression_metrics(split["y_test"], y_pred)

    model_root = Path(args.model_root or output_cfg.get("model_dir", f"models/{model_type}"))
    run_dir = create_run_dir(model_root, prefix=output_cfg.get("run_prefix", model_type))

    model_path = save_model(model, run_dir / "model.joblib")
    save_json(metrics, run_dir / "metrics.json")
    save_json(
        {
            "target_file": target_name,
            "model_type": model_type,
            "feature_files": bundle["feature_files"],
            "feature_names": bundle["feature_names"],
            "n_features": len(bundle["feature_files"]),
            "n_valid_pixels": int(bundle["X"].shape[0]),
            "split": {
                "test_size": float(split_cfg.get("test_size", 0.2)),
                "random_state": int(split_cfg.get("random_state", 42)),
            },
        },
        run_dir / "feature_order.json",
    )
    save_json(cfg, run_dir / "resolved_train_config.json")
    copy_config(args.config, run_dir / "train_config.yaml")

    print("Saved model:", model_path)
    print("Saved metrics:", run_dir / "metrics.json")
    print("Saved features:", run_dir / "feature_order.json")
    print("Model type:", model_type)
    print("Test R2:", metrics["r2"])
    print("Test RMSE:", metrics["rmse"])


if __name__ == "__main__":
    main()
