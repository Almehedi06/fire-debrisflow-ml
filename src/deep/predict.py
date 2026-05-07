from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def _tile_starts(size: int, patch_size: int, stride: int) -> list[int]:
    if size < patch_size:
        return [0]
    starts = list(range(0, size - patch_size + 1, stride))
    last = size - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _feather_axis_weights(length: int, min_weight: float = 0.05) -> np.ndarray:
    if length <= 2:
        return np.ones(length, dtype="float32")

    center_distance = (length - 1) // 2
    if center_distance <= 0:
        return np.ones(length, dtype="float32")

    coords = np.arange(length, dtype="float32")
    edge_distance = np.minimum(coords, (length - 1) - coords)
    weights = edge_distance / float(center_distance)
    return np.clip(weights, min_weight, 1.0).astype("float32")


def _feather_patch_weights(height: int, width: int, min_weight: float = 0.05) -> np.ndarray:
    wy = _feather_axis_weights(height, min_weight=min_weight)
    wx = _feather_axis_weights(width, min_weight=min_weight)
    return (wy[:, None] * wx[None, :]).astype("float32")


def predict_full_raster(
    model,
    x: np.ndarray,  # [C, H, W]
    valid_mask: np.ndarray,  # [H, W]
    patch_size: int = 128,
    stride: int = 64,
    device: str = "cpu",
    nodata_value: float = -9999.0,
) -> np.ndarray:
    import torch

    _, h, w = x.shape
    pred_sum = np.zeros((h, w), dtype="float32")
    pred_count = np.zeros((h, w), dtype="float32")
    base_weights = _feather_patch_weights(patch_size, patch_size)

    row_starts = _tile_starts(h, patch_size, stride)
    col_starts = _tile_starts(w, patch_size, stride)

    model.eval()
    with torch.no_grad():
        for r0 in row_starts:
            r1 = min(r0 + patch_size, h)
            for c0 in col_starts:
                c1 = min(c0 + patch_size, w)
                patch = x[:, r0:r1, c0:c1]

                # Pad edge tiles to fixed patch size for model input.
                padded = np.zeros((x.shape[0], patch_size, patch_size), dtype="float32")
                padded[:, : patch.shape[1], : patch.shape[2]] = patch

                xb = torch.from_numpy(padded[None, :, :, :]).to(device)
                out = model(xb).detach().cpu().numpy()[0, 0]
                out = out[: patch.shape[1], : patch.shape[2]]
                if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
                    weights = base_weights
                else:
                    weights = _feather_patch_weights(patch.shape[1], patch.shape[2])

                pred_sum[r0:r1, c0:c1] += out * weights
                pred_count[r0:r1, c0:c1] += weights

    pred_count[pred_count == 0.0] = 1.0
    pred = pred_sum / pred_count
    pred[~valid_mask] = nodata_value
    return pred.astype("float32")


def save_prediction_tif(
    pred: np.ndarray,
    profile: dict,
    out_path: str | Path,
    nodata_value: float = -9999.0,
) -> Path:
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
        dst.write(pred, 1)
    return out
