from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from ml.dataset import build_prediction_matrix


def predict_raster_from_features(
    model,
    feature_paths: list[str | Path],
    out_path: str | Path,
    nodata_value: float = -9999.0,
) -> Path:
    bundle = build_prediction_matrix(feature_paths)

    x_valid = bundle["X_valid"]
    valid_mask = bundle["valid_mask"]
    profile = bundle["profile"]
    shape = bundle["shape"]

    preds_valid = model.predict(x_valid).astype("float32")
    pred_full = np.full(shape, nodata_value, dtype="float32")
    pred_full[valid_mask] = preds_valid

    out_profile = profile.copy()
    out_profile.update(
        {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "nodata": nodata_value,
        }
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **out_profile) as dst:
        dst.write(pred_full, 1)

    return out

