from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_feature_order(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RF model on a raster stack.")
    parser.add_argument("--model-path", required=True, help="Path to trained model.joblib.")
    parser.add_argument(
        "--feature-order",
        required=True,
        help="Path to feature_order.json from training run.",
    )
    parser.add_argument("--data-dir", required=True, help="Directory with feature/target rasters.")
    parser.add_argument("--target", default="dem_diff.tif", help="Target raster filename.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from ml.dataset import build_xy_from_rasters
    from ml.evaluate import regression_metrics
    from ml.io import load_model

    model = load_model(args.model_path)
    order = _load_feature_order(Path(args.feature_order))

    data_dir = Path(args.data_dir)
    target_path = data_dir / args.target
    if not target_path.exists():
        raise FileNotFoundError(f"Target raster not found: {target_path}")

    feature_files = order.get("feature_files")
    if not feature_files:
        raise ValueError("feature_order file missing 'feature_files'.")
    feature_paths = [data_dir / name for name in feature_files]
    for fp in feature_paths:
        if not fp.exists():
            raise FileNotFoundError(f"Feature raster not found: {fp}")

    bundle = build_xy_from_rasters(feature_paths, target_path)
    y_pred = model.predict(bundle["X"])
    metrics = regression_metrics(bundle["y"], y_pred)

    print("Evaluation samples:", metrics["n_samples"])
    print("R2:", metrics["r2"])
    print("RMSE:", metrics["rmse"])
    print("MAE:", metrics["mae"])


if __name__ == "__main__":
    main()
