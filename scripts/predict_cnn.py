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
    parser = argparse.ArgumentParser(description="Run full-raster inference with trained CNN.")
    parser.add_argument("--model-path", required=True, help="Path to trained model.pt.")
    parser.add_argument(
        "--feature-order",
        required=True,
        help="Path to feature_order.json from training run.",
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing feature rasters.")
    parser.add_argument("--out-path", required=True, help="Output prediction GeoTIFF path.")
    parser.add_argument("--patch-size", type=int, default=None, help="Override inference patch size.")
    parser.add_argument("--stride", type=int, default=None, help="Override inference stride.")
    parser.add_argument("--nodata", type=float, default=-9999.0, help="Output nodata value.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from deep.cnn import SimpleCNNRegressor
    from deep.data import load_feature_stack
    from deep.predict import predict_full_raster, save_prediction_tif

    import numpy as np
    import torch

    checkpoint = torch.load(args.model_path, map_location="cpu")
    in_channels = int(checkpoint["in_channels"])
    hidden_channels = int(checkpoint.get("hidden_channels", 64))
    num_layers = int(checkpoint.get("num_layers", 5))
    patch_size = int(args.patch_size or checkpoint.get("patch_size", 128))
    stride = int(args.stride or checkpoint.get("stride", 64))
    norm_mean = checkpoint.get("norm_mean")
    norm_std = checkpoint.get("norm_std")

    model = SimpleCNNRegressor(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(args.feature_order, "r") as f:
        feature_order = json.load(f)
    feature_files = feature_order.get("feature_files")
    if not feature_files:
        raise ValueError("feature_order file missing 'feature_files'.")

    data_dir = Path(args.data_dir)
    feature_paths = [data_dir / p for p in feature_files]
    for fp in feature_paths:
        if not fp.exists():
            raise FileNotFoundError(f"Feature raster not found: {fp}")

    bundle = load_feature_stack(feature_paths)
    if norm_mean is not None and norm_std is not None:
        mean = np.asarray(norm_mean, dtype="float32")
        std = np.asarray(norm_std, dtype="float32")
        std[std == 0] = 1.0
        x = (bundle["X"].astype("float64") - mean[:, None, None].astype("float64")) / std[
            :, None, None
        ].astype("float64")
        x = np.clip(x, -20.0, 20.0)
        bundle["X"] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    pred = predict_full_raster(
        model=model,
        x=bundle["X"],
        valid_mask=bundle["valid_mask"],
        patch_size=patch_size,
        stride=stride,
        device="cpu",
        nodata_value=float(args.nodata),
    )
    out_path = save_prediction_tif(
        pred=pred,
        profile=bundle["profile"],
        out_path=args.out_path,
        nodata_value=float(args.nodata),
    )
    print("Saved predictions:", out_path)


if __name__ == "__main__":
    main()
