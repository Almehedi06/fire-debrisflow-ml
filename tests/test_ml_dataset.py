from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.dataset import discover_feature_paths


def _write_tif(path: Path, value: float = 1.0) -> None:
    arr = np.full((2, 2), value, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:32610",
        transform=from_origin(500000.0, 4100000.0, 30.0, 30.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(arr, 1)


def test_discover_feature_paths_preserves_include_names_order(tmp_path: Path) -> None:
    _write_tif(tmp_path / "b.tif", value=2.0)
    _write_tif(tmp_path / "a.tif", value=1.0)
    _write_tif(tmp_path / "dem_diff.tif", value=0.0)

    paths = discover_feature_paths(
        data_dir=tmp_path,
        target_name="dem_diff.tif",
        include_names=["b.tif", "a.tif"],
    )

    assert [p.name for p in paths] == ["b.tif", "a.tif"]


def test_discover_feature_paths_rejects_duplicate_include_names(tmp_path: Path) -> None:
    _write_tif(tmp_path / "a.tif", value=1.0)
    _write_tif(tmp_path / "dem_diff.tif", value=0.0)

    with pytest.raises(ValueError, match="Duplicate feature"):
        discover_feature_paths(
            data_dir=tmp_path,
            target_name="dem_diff.tif",
            include_names=["a.tif", "a.tif"],
        )


def test_discover_feature_paths_rejects_excluded_explicit_feature(tmp_path: Path) -> None:
    _write_tif(tmp_path / "dem_pre.tif", value=1.0)
    _write_tif(tmp_path / "dem_diff.tif", value=0.0)

    with pytest.raises(ValueError, match="excluded or reserved as target"):
        discover_feature_paths(
            data_dir=tmp_path,
            target_name="dem_diff.tif",
            include_names=["dem_pre.tif"],
            exclude_names=["dem_pre.tif", "dem_post.tif"],
        )
