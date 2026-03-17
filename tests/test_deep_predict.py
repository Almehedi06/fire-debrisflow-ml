from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep.predict import _feather_patch_weights, predict_full_raster


def test_feather_patch_weights_prioritize_tile_center() -> None:
    weights = _feather_patch_weights(8, 8)

    assert weights.shape == (8, 8)
    assert np.all(weights > 0.0)
    assert float(weights[3, 3]) > float(weights[0, 0])
    assert np.isclose(float(weights[0, 0]), float(weights[0, -1]))
    assert np.isclose(float(weights[0, 0]), float(weights[-1, 0]))


def test_predict_full_raster_preserves_constant_prediction_with_overlap() -> None:
    torch = pytest.importorskip("torch")

    class ConstantModel(torch.nn.Module):
        def forward(self, xb):
            batch, _, height, width = xb.shape
            return torch.full((batch, 1, height, width), 7.5, dtype=xb.dtype, device=xb.device)

    x = np.zeros((2, 9, 9), dtype="float32")
    valid_mask = np.ones((9, 9), dtype=bool)
    valid_mask[0, 0] = False

    pred = predict_full_raster(
        model=ConstantModel(),
        x=x,
        valid_mask=valid_mask,
        patch_size=4,
        stride=2,
        nodata_value=-9999.0,
    )

    assert pred.shape == (9, 9)
    assert pred[0, 0] == -9999.0
    assert np.allclose(pred[valid_mask], 7.5)
