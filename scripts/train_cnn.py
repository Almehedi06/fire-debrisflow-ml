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
    parser = argparse.ArgumentParser(description="Train simple CNN regressor for dem_diff.")
    parser.add_argument("--config", default="config/ml_cnn.yaml", help="Training config path.")
    parser.add_argument("--data-dir", default=None, help="Override raster data directory.")
    parser.add_argument("--target", default=None, help="Override target TIFF filename.")
    parser.add_argument("--model-root", default=None, help="Override model output root directory.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from deep.data import load_feature_target_stack
    from deep.train import train_cnn_regression
    from ml.dataset import discover_feature_paths
    from ml.io import copy_config, create_run_dir, save_json

    cfg = _load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    output_cfg = cfg.get("output", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

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

    bundle = load_feature_target_stack(feature_paths, target_path)
    artifacts = train_cnn_regression(
        x=bundle["X"],
        y=bundle["y"],
        valid_mask=bundle["valid_mask"],
        config=cfg,
    )

    model_root = Path(args.model_root or output_cfg.get("model_dir", "models/cnn"))
    run_dir = create_run_dir(model_root, prefix=output_cfg.get("run_prefix", "cnn"))

    import torch

    checkpoint = {
        "model_state_dict": artifacts.model.state_dict(),
        "in_channels": int(bundle["X"].shape[0]),
        "hidden_channels": int(model_cfg.get("hidden_channels", 64)),
        "num_layers": int(model_cfg.get("num_layers", 5)),
        "patch_size": int(training_cfg.get("patch_size", 128)),
        "stride": int(training_cfg.get("stride", 64)),
        "norm_mean": artifacts.norm_mean.tolist(),
        "norm_std": artifacts.norm_std.tolist(),
    }
    model_path = run_dir / "model.pt"
    torch.save(checkpoint, model_path)

    save_json(artifacts.test_metrics, run_dir / "metrics_test.json")
    save_json({"history": artifacts.history}, run_dir / "history.json")
    save_json(artifacts.split_summary, run_dir / "split_summary.json")
    save_json(
        {
            "target_file": target_name,
            "feature_files": bundle["feature_files"],
            "feature_names": bundle["feature_names"],
            "n_features": len(bundle["feature_files"]),
            "model_type": "cnn",
        },
        run_dir / "feature_order.json",
    )
    save_json(cfg, run_dir / "resolved_train_config.json")
    copy_config(args.config, run_dir / "train_config.yaml")

    print("Saved model:", model_path)
    print("Saved metrics:", run_dir / "metrics_test.json")
    print("Saved history:", run_dir / "history.json")
    print("Saved split:", run_dir / "split_summary.json")
    print("Test R2:", artifacts.test_metrics["r2"])
    print("Test RMSE:", artifacts.test_metrics["rmse"])
    print("Test MAE:", artifacts.test_metrics["mae"])


if __name__ == "__main__":
    main()
