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

from pipeline import run_landlab_pipeline, run_raster_pipeline
from reproject_and_resample import clip_raster_to_shape, convert_to_ascii, read_ascii_header


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


def _write_aoi(path: Path, bounds, crs: str) -> None:
    geopandas = pytest.importorskip("geopandas")
    shapely_geometry = pytest.importorskip("shapely.geometry")

    polygon = shapely_geometry.box(*bounds)
    gdf = geopandas.GeoDataFrame({"id": [1]}, geometry=[polygon], crs=crs)
    gdf.to_file(path, driver="ESRI Shapefile")


def test_clip_raster_to_shape_preserves_template_grid(tmp_path: Path) -> None:
    pytest.importorskip("fiona")

    crs = "EPSG:32610"
    transform = from_origin(500000.0, 4100000.0, 30.0, 30.0)
    arr = np.arange(36, dtype="float32").reshape(6, 6)
    tif_path = tmp_path / "source.tif"
    _write_tif(tif_path, arr, transform, crs)

    with rasterio.open(tif_path) as src:
        template_meta = src.meta.copy()
        bounds = src.bounds

    # A smaller AOI inside the raster should mask values, not shrink the grid,
    # when a template grid is provided.
    aoi_bounds = (
        bounds.left + 30.0,
        bounds.bottom + 30.0,
        bounds.right - 30.0,
        bounds.top - 30.0,
    )
    aoi_path = tmp_path / "aoi.shp"
    _write_aoi(aoi_path, aoi_bounds, crs)

    clipped_path = Path(clip_raster_to_shape(str(tif_path), str(aoi_path), template_meta=template_meta))
    with rasterio.open(clipped_path) as src:
        clipped = src.read(1)
        assert src.width == 6
        assert src.height == 6
        assert src.transform == transform
        assert np.count_nonzero(clipped == src.nodata) > 0

    asc_path = Path(convert_to_ascii(str(clipped_path), str(tmp_path), template_meta=template_meta))
    header = read_ascii_header(str(asc_path))
    assert header["ncols"] == 6
    assert header["nrows"] == 6
    assert header["cellsize"] == 30.0

    lines = asc_path.read_text().strip().splitlines()
    assert len(lines) == 12


def test_main_pipeline_local_smoke(tmp_path: Path) -> None:
    pytest.importorskip("fiona")
    pytest.importorskip("geopandas")
    pytest.importorskip("landlab")

    crs = "EPSG:32610"
    transform = from_origin(500000.0, 4100000.0, 30.0, 30.0)
    shape = (6, 6)

    dem = np.arange(shape[0] * shape[1], dtype="float32").reshape(shape) + 100.0
    burn = np.full(shape, 3.0, dtype="float32")
    landcover = np.full(shape, 42.0, dtype="float32")

    data_dir = tmp_path / "inputs"
    out_dir = tmp_path / "outputs"
    data_dir.mkdir()
    out_dir.mkdir()

    dem_path = data_dir / "dem.tif"
    burn_path = data_dir / "burn.tif"
    landcover_path = data_dir / "landcover.tif"
    _write_tif(dem_path, dem, transform, crs)
    _write_tif(burn_path, burn, transform, crs)
    _write_tif(landcover_path, landcover, transform, crs)

    soil_keys = [
        "cec7_0_cm",
        "anylithicdpt_cm",
        "claytotal_0_cm",
        "ph1to1h2o_0_cm",
        "sandtotal_0_cm",
        "silttotal_0_cm",
        "dbovendry_0_cm",
    ]
    soil_values = {
        "cec7_0_cm": 10.0,
        "anylithicdpt_cm": 200.0,
        "claytotal_0_cm": 25.0,
        "ph1to1h2o_0_cm": 650.0,
        "sandtotal_0_cm": 40.0,
        "silttotal_0_cm": 35.0,
        "dbovendry_0_cm": 140.0,
    }
    for key in soil_keys:
        _write_tif(data_dir / f"{key}.tif", np.full(shape, soil_values[key], dtype="float32"), transform, crs)

    with rasterio.open(dem_path) as src:
        bounds = src.bounds
    aoi_path = data_dir / "aoi.shp"
    _write_aoi(aoi_path, (bounds.left, bounds.bottom, bounds.right, bounds.top), crs)

    cfg = {
        "aoi": {"aoi": str(aoi_path)},
        "paths": {
            "input_dir": str(data_dir),
            "output_dir": str(out_dir),
        },
        "raster": {
            "target_res": 30.0,
            "resampling_method": "bilinear",
        },
        "fire": {
            "name": "Test Fire",
            "id": "TEST000",
        },
        "dem": {
            "source": "local",
            "path": str(dem_path),
        },
        "burn_severity": {
            "source": "local",
            "local": {
                "path": str(data_dir),
                "filename": "burn.tif",
                "resampling": "nearest",
            },
        },
        "feature_sources": {
            "rasters": {
                key: {
                    "url": str(data_dir / f"{key}.tif"),
                    "resampling": "bilinear",
                }
                for key in soil_keys
            },
            "landcover": {
                "nlcd_local": {
                    "url": str(landcover_path),
                    "resampling": "nearest",
                    "unzip": False,
                }
            },
        },
    }

    outputs = run_raster_pipeline(cfg, cleanup_intermediates=False)
    grid = run_landlab_pipeline(cfg, outputs)

    assert "soil__thickness" in grid.at_node
    assert np.isclose(float(np.mean(grid.at_node["soil__thickness"][grid.core_nodes])), 2.0)

    expected_files = [
        out_dir / "topographic__elevation.asc",
        out_dir / "soil__thickness.asc",
        out_dir / "landcover.asc",
        out_dir / "soil__transmissivity.asc",
        out_dir / "vegetation__plant_functional_type.asc",
    ]
    for path in expected_files:
        assert path.exists(), f"Missing output: {path}"
