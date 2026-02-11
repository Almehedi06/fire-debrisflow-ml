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

                pred_sum[r0:r1, c0:c1] += out
                pred_count[r0:r1, c0:c1] += 1.0

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

