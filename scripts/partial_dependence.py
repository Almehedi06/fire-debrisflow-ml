from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_yaml(path: str | Path | None) -> dict:
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _parse_feature_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    items = [x.strip() for x in raw.split(",")]
    return [x for x in items if x]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute partial dependence curves.")
    parser.add_argument("--config", default=None, help="Optional YAML config for interpretation.")
    parser.add_argument("--model-path", default=None, help="Path to model artifact.")
    parser.add_argument("--feature-order", default=None, help="Path to feature_order.json.")
    parser.add_argument("--data-dir", default=None, help="Directory containing feature/target rasters.")
    parser.add_argument("--target", default=None, help="Target raster filename.")
    parser.add_argument("--out-dir", default=None, help="Output directory for CSV/plots.")
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature names for PDP, e.g. topographic__elevation,soil__thickness",
    )
    parser.add_argument("--max-features", type=int, default=None, help="If --features omitted, use first N.")
    parser.add_argument("--grid-resolution", type=int, default=None, help="Grid points for PDP curve.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max evaluation samples.")
    parser.add_argument("--random-state", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def _cfg_get(cfg: dict, section: str, key: str, default):
    return cfg.get(section, {}).get(key, default)


def main() -> None:
    args = _parse_args()

    from ml.interpret import (
        build_test_split_for_interpretation,
        compute_partial_dependence_curves,
        default_interpret_out_dir,
        load_feature_order,
        plot_pdp_curve,
        resolve_feature_paths,
        save_pdp_csv,
    )
    from ml.io import load_model, save_json

    cfg = _load_yaml(args.config)

    model_path = args.model_path or _cfg_get(cfg, "model", "path", None)
    feature_order_path = args.feature_order or _cfg_get(cfg, "model", "feature_order", None)
    data_dir = args.data_dir or _cfg_get(cfg, "data", "dir", None)
    target_name = args.target or _cfg_get(cfg, "data", "target_name", "dem_diff.tif")

    if not model_path or not feature_order_path or not data_dir:
        raise ValueError("Provide model-path, feature-order, and data-dir (CLI or config).")

    max_features = int(args.max_features or _cfg_get(cfg, "partial_dependence", "max_features", 8))
    grid_resolution = int(
        args.grid_resolution or _cfg_get(cfg, "partial_dependence", "grid_resolution", 50)
    )
    max_samples = int(args.max_samples or _cfg_get(cfg, "partial_dependence", "max_samples", 50000))
    random_state = int(args.random_state or _cfg_get(cfg, "partial_dependence", "random_state", 42))

    out_dir = Path(args.out_dir or _cfg_get(cfg, "output", "dir", "")) if args.out_dir or _cfg_get(cfg, "output", "dir", None) else default_interpret_out_dir(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_order = load_feature_order(feature_order_path)
    feature_files = feature_order.get("feature_files")
    if not feature_files:
        raise ValueError("feature_order.json missing 'feature_files'.")
    feature_paths = resolve_feature_paths(data_dir, feature_files)

    split_cfg = feature_order.get("split", {})
    split_test_size = float(split_cfg.get("test_size", 0.2))
    split_random_state = int(split_cfg.get("random_state", 42))

    target_path = Path(data_dir) / target_name
    if not target_path.exists():
        raise FileNotFoundError(f"Target raster not found: {target_path}")

    model = load_model(model_path)
    eval_bundle = build_test_split_for_interpretation(
        feature_paths=feature_paths,
        target_path=target_path,
        split_test_size=split_test_size,
        split_random_state=split_random_state,
        max_samples=max_samples,
        sample_random_state=random_state,
    )

    selected = _parse_feature_list(args.features)
    if selected is None:
        selected = _cfg_get(cfg, "partial_dependence", "features", None)
    if not selected:
        selected = eval_bundle["feature_names"][:max_features]

    curves = compute_partial_dependence_curves(
        model=model,
        x_eval=eval_bundle["X_test"],
        feature_names=eval_bundle["feature_names"],
        selected_features=selected,
        grid_resolution=grid_resolution,
    )

    saved = []
    for feature_name, curve in curves.items():
        safe = feature_name.replace("/", "_")
        csv_path = save_pdp_csv(
            feature_name,
            curve["x"],
            curve["y"],
            out_dir / f"pdp_{safe}.csv",
        )
        fig_path = plot_pdp_curve(
            feature_name,
            curve["x"],
            curve["y"],
            out_dir / f"pdp_{safe}.png",
        )
        saved.append((csv_path, fig_path))

    save_json(
        {
            "model_path": str(model_path),
            "feature_order_path": str(feature_order_path),
            "data_dir": str(data_dir),
            "target_name": target_name,
            "selected_features": selected,
            "grid_resolution": grid_resolution,
            "max_samples": max_samples,
            "random_state": random_state,
            "split_test_size": split_test_size,
            "split_random_state": split_random_state,
        },
        out_dir / "pdp_meta.json",
    )

    for csv_path, fig_path in saved:
        print("Saved:", csv_path)
        print("Saved:", fig_path)


if __name__ == "__main__":
    main()

