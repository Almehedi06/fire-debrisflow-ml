from __future__ import annotations

import json
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

from soil_data.core import (
    DEFAULT_SOIL_SPECS,
    SoilVarSpec,
    harmonize_soil_layers,
    parse_soil_keys,
    resolve_soil_specs,
)


def _write_tif(path: Path, data: np.ndarray, transform, crs: str, nodata: float = -9999.0) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def test_parse_soil_keys_default() -> None:
    assert parse_soil_keys(None) == list(DEFAULT_SOIL_SPECS.keys())


def test_parse_soil_keys_subset() -> None:
    keys = parse_soil_keys("cec7_0_cm,claytotal_0_cm")
    assert keys == ["cec7_0_cm", "claytotal_0_cm"]


def test_parse_soil_keys_unknown() -> None:
    with pytest.raises(ValueError):
        parse_soil_keys("does_not_exist")


def test_resolve_soil_specs_override() -> None:
    cfg = {
        "feature_sources": {
            "rasters": {
                "cec7_0_cm": {
                    "url": "file:///tmp/custom.tif",
                    "resampling": "nearest",
                }
            }
        }
    }
    specs = resolve_soil_specs(["cec7_0_cm"], cfg)
    assert len(specs) == 1
    assert specs[0].url == "file:///tmp/custom.tif"
    assert specs[0].resampling == "nearest"


def test_harmonize_grid_invariants(tmp_path: Path) -> None:
    pytest.importorskip("fiona")
    geopandas = pytest.importorskip("geopandas")
    shapely_geometry = pytest.importorskip("shapely.geometry")

    crs = "EPSG:32610"
    template_transform = from_origin(500000.0, 4100000.0, 30.0, 30.0)
    template_shape = (6, 6)

    template_tif = tmp_path / "template.tif"
    template_arr = np.arange(template_shape[0] * template_shape[1], dtype="float32").reshape(
        template_shape
    )
    _write_tif(template_tif, template_arr, template_transform, crs)

    source_dir = tmp_path / "raw"
    source_dir.mkdir(parents=True, exist_ok=True)
    src_arr = np.ones((8, 8), dtype="float32")
    src_transform = from_origin(499940.0, 4100060.0, 20.0, 20.0)
    _write_tif(source_dir / "cec7_0_cm.tif", src_arr, src_transform, crs)

    west, north = template_transform.c, template_transform.f
    east = west + template_shape[1] * template_transform.a
    south = north + template_shape[0] * template_transform.e
    polygon = shapely_geometry.box(
        min(west, east), min(south, north), max(west, east), max(south, north)
    )
    gdf = geopandas.GeoDataFrame({"id": [1]}, geometry=[polygon], crs=crs)
    aoi_path = tmp_path / "aoi.shp"
    gdf.to_file(aoi_path, driver="ESRI Shapefile")

    out_dir = tmp_path / "harmonized"
    spec = SoilVarSpec(
        key="cec7_0_cm",
        field_name="cation__exchange_capacity",
        url="unused_for_this_test",
        resampling="bilinear",
    )
    manifest_path = harmonize_soil_layers(
        aoi_path=aoi_path,
        output_dir=out_dir,
        specs=[spec],
        source_dir=source_dir,
        template_path=template_tif,
        target_res=30.0,
        output_format="both",
        overwrite=True,
        keep_intermediates=False,
    )

    out_tif = out_dir / "cation__exchange_capacity.tif"
    assert out_tif.exists()
    with rasterio.open(template_tif) as ref, rasterio.open(out_tif) as out:
        assert out.crs == ref.crs
        assert out.transform == ref.transform
        assert out.width == ref.width
        assert out.height == ref.height

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    assert manifest["manifest_version"] == "1.0"
    assert manifest["stage"] == "harmonize"
    assert manifest["grid"]["width"] == template_shape[1]
    assert manifest["grid"]["height"] == template_shape[0]
