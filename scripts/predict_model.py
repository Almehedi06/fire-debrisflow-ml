from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raster inference with trained RF model.")
    parser.add_argument("--model-path", required=True, help="Path to model.joblib.")
    parser.add_argument(
        "--feature-order",
        required=True,
        help="Path to feature_order.json from training run.",
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing feature rasters.")
    parser.add_argument(
        "--out-path",
        required=True,
        help="Output GeoTIFF path for predictions.",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=-9999.0,
        help="Nodata value for output prediction raster.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from ml.io import load_model
    from ml.predict import predict_raster_from_features

    model = load_model(args.model_path)

    with open(args.feature_order, "r") as f:
        order = json.load(f)
    feature_files = order.get("feature_files")
    if not feature_files:
        raise ValueError("feature_order file missing 'feature_files'.")

    data_dir = Path(args.data_dir)
    feature_paths = [data_dir / name for name in feature_files]
    for fp in feature_paths:
        if not fp.exists():
            raise FileNotFoundError(f"Feature raster not found: {fp}")

    out_path = predict_raster_from_features(
        model=model,
        feature_paths=feature_paths,
        out_path=args.out_path,
        nodata_value=args.nodata,
    )
    print("Saved predictions:", out_path)


if __name__ == "__main__":
    main()
