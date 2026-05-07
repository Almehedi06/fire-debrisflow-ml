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
    parser = argparse.ArgumentParser(description="Tune U-Net configs for dem_diff regression.")
    parser.add_argument("--config", default="config/ml_unet.yaml", help="Training config path.")
    parser.add_argument("--data-dir", default=None, help="Override raster data directory.")
    parser.add_argument("--target", default=None, help="Override target TIFF filename.")
    parser.add_argument("--model-root", default=None, help="Override tuning output root directory.")
    parser.add_argument("--device", default=None, help="Override runtime device: auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument(
        "--selection-metric",
        choices=["rmse", "mae", "r2"],
        default=None,
        help="Override tuning selection metric from config.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optionally limit the number of generated candidates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print candidate count and configs without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import torch

    from deep.data import load_feature_target_stack
    from deep.train import train_unet_regression
    from deep.tuning import expand_search_space, select_best_candidate, validate_unet_candidate_config
    from ml.dataset import discover_feature_paths
    from ml.io import copy_config, create_run_dir, save_json

    cfg = _load_yaml(args.config)
    cfg.setdefault("data", {})
    cfg.setdefault("output", {})
    cfg.setdefault("tuning", {})
    cfg.setdefault("runtime", {})

    if args.data_dir is not None:
        cfg["data"]["dir"] = args.data_dir
    if args.target is not None:
        cfg["data"]["target_name"] = args.target
    if args.device is not None:
        cfg["runtime"]["device"] = args.device

    data_cfg = cfg.get("data", {})
    output_cfg = cfg.get("output", {})
    tuning_cfg = cfg.get("tuning", {})

    data_dir = Path(data_cfg.get("dir"))
    target_name = data_cfg.get("target_name", "dem_diff.tif")
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

    candidates = expand_search_space(cfg)
    if not candidates:
        raise ValueError("No tuning.search_space found in config/ml_unet.yaml.")
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    selection_metric = str(
        args.selection_metric or tuning_cfg.get("selection_metric", "rmse")
    ).lower()
    if selection_metric not in {"rmse", "mae", "r2"}:
        raise ValueError("tuning.selection_metric must be one of: rmse, mae, r2.")

    for candidate in candidates:
        candidate["config"].setdefault("training", {})
        candidate["config"]["training"].setdefault("checkpoint_metric", selection_metric)
        validate_unet_candidate_config(candidate["config"])

    if args.dry_run:
        print("Selection metric:", selection_metric)
        print("Candidates:", len(candidates))
        for idx, candidate in enumerate(candidates, start=1):
            print(f"{idx:03d}:", candidate["overrides"])
        return

    tuning_root = Path(
        args.model_root
        or output_cfg.get("tuning_dir")
        or str(Path(output_cfg.get("model_dir", "models/unet")).with_name("unet_tuning"))
    )
    run_dir = create_run_dir(
        tuning_root,
        prefix=tuning_cfg.get("run_prefix", output_cfg.get("run_prefix", "unet_tune")),
    )
    candidates_root = run_dir / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)

    save_json(cfg, run_dir / "resolved_tuning_config.json")
    copy_config(args.config, run_dir / "input_config.yaml")
    save_json(
        {
            "target_file": target_name,
            "feature_files": bundle["feature_files"],
            "feature_names": bundle["feature_names"],
            "n_features": len(bundle["feature_files"]),
            "model_type": "unet",
            "selection_metric": selection_metric,
            "n_candidates": len(candidates),
            "device_requested": cfg["runtime"].get("device", "auto"),
        },
        run_dir / "feature_order.json",
    )

    results: list[dict] = []
    best_payload: dict | None = None
    best_artifacts = None
    best_cfg = None

    for idx, candidate in enumerate(candidates, start=1):
        candidate_cfg = candidate["config"]
        overrides = candidate["overrides"]
        candidate_dir = candidates_root / f"candidate_{idx:03d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(candidates)}] Training U-Net candidate:", overrides)
        artifacts = train_unet_regression(
            x=bundle["X"],
            y=bundle["y"],
            valid_mask=bundle["valid_mask"],
            config=candidate_cfg,
        )

        save_json(candidate_cfg, candidate_dir / "resolved_train_config.json")
        save_json(artifacts.best_val_metrics, candidate_dir / "metrics_val.json")
        save_json(artifacts.test_metrics, candidate_dir / "metrics_test.json")
        save_json({"history": artifacts.history}, candidate_dir / "history.json")
        save_json(artifacts.split_summary, candidate_dir / "split_summary.json")

        row = {
            "candidate_id": idx,
            "overrides": overrides,
            "best_epoch": artifacts.best_epoch,
            "device_requested": artifacts.device_requested,
            "device_resolved": artifacts.device_resolved,
            "best_val_r2": float(artifacts.best_val_metrics["r2"]),
            "best_val_rmse": float(artifacts.best_val_metrics["rmse"]),
            "best_val_mae": float(artifacts.best_val_metrics["mae"]),
            "test_r2": float(artifacts.test_metrics["r2"]),
            "test_rmse": float(artifacts.test_metrics["rmse"]),
            "test_mae": float(artifacts.test_metrics["mae"]),
            "test_loss": float(artifacts.test_metrics["test_loss"]),
            "candidate_dir": str(candidate_dir),
        }
        results.append(row)

        best_row = select_best_candidate(results, selection_metric)
        if best_row["candidate_id"] == idx:
            best_payload = row
            best_artifacts = artifacts
            best_cfg = candidate_cfg

        save_json(
            {
                "selection_metric": selection_metric,
                "best_candidate": select_best_candidate(results, selection_metric),
                "candidates": results,
            },
            run_dir / "tuning_results.json",
        )

    if best_payload is None or best_artifacts is None or best_cfg is None:
        raise RuntimeError("U-Net tuning produced no best candidate.")

    best_dir = run_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_artifacts.model.state_dict(),
        "in_channels": int(bundle["X"].shape[0]),
        "base_channels": int(best_cfg.get("model", {}).get("base_channels", 32)),
        "patch_size": int(best_cfg.get("training", {}).get("patch_size", 128)),
        "stride": int(best_cfg.get("training", {}).get("stride", 64)),
        "device_requested": best_artifacts.device_requested,
        "device_resolved": best_artifacts.device_resolved,
        "norm_mean": best_artifacts.norm_mean.tolist(),
        "norm_std": best_artifacts.norm_std.tolist(),
    }
    model_path = best_dir / "model.pt"
    torch.save(checkpoint, model_path)

    save_json(best_artifacts.best_val_metrics, best_dir / "metrics_val.json")
    save_json(best_artifacts.test_metrics, best_dir / "metrics_test.json")
    save_json({"history": best_artifacts.history}, best_dir / "history.json")
    save_json(best_artifacts.split_summary, best_dir / "split_summary.json")
    save_json(best_cfg, best_dir / "resolved_train_config.json")
    save_json(
        {
            "target_file": target_name,
            "feature_files": bundle["feature_files"],
            "feature_names": bundle["feature_names"],
            "n_features": len(bundle["feature_files"]),
            "model_type": "unet",
        },
        best_dir / "feature_order.json",
    )

    print("Saved tuning results:", run_dir / "tuning_results.json")
    print("Saved best model:", model_path)
    print("Best candidate:", best_payload["candidate_id"])
    print("Best overrides:", best_payload["overrides"])
    print("Device:", best_payload["device_resolved"], f"(requested: {best_payload['device_requested']})")
    print("Best Val RMSE:", best_payload["best_val_rmse"])
    print("Best Test RMSE:", best_payload["test_rmse"])


if __name__ == "__main__":
    main()
