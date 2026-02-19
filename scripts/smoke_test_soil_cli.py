from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import importlib.util

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import yaml


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


def main() -> None:
    if importlib.util.find_spec("fiona") is None:
        raise RuntimeError(
            "Missing dependency 'fiona'. Create env first: "
            "'conda env create -f environment.yml && conda activate fire-debrisflow-ml'"
        )

    root = Path(__file__).resolve().parents[1]
    cli_path = root / "scripts" / "soil_run.py"
    if not cli_path.exists():
        raise FileNotFoundError(f"Missing soil CLI: {cli_path}")

    soil_map = {
        "cec7_0_cm": "cation__exchange_capacity",
        "anylithicdpt_cm": "soil__thickness",
        "claytotal_0_cm": "clay__total",
        "ph1to1h2o_0_cm": "pH",
        "sandtotal_0_cm": "sand__total",
        "silttotal_0_cm": "silt__total",
        "dbovendry_0_cm": "dry__bulk_density",
    }

    with tempfile.TemporaryDirectory(prefix="soil_cli_smoke_") as tmpdir:
        tmp = Path(tmpdir)
        input_dir = tmp / "inputs"
        output_dir = tmp / "outputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        crs = "EPSG:32610"
        transform = from_origin(500000.0, 4100000.0, 30.0, 30.0)
        shape = (6, 6)

        template_arr = np.arange(shape[0] * shape[1], dtype="float32").reshape(shape)
        template_path = input_dir / "template.tif"
        _write_tif(template_path, template_arr, transform, crs=crs)

        for i, key in enumerate(soil_map.keys(), start=1):
            arr = np.full(shape, float(i), dtype="float32")
            _write_tif(input_dir / f"{key}.tif", arr, transform, crs=crs)

        west, north = transform.c, transform.f
        east = west + shape[1] * transform.a
        south = north + shape[0] * transform.e
        polygon = box(min(west, east), min(south, north), max(west, east), max(south, north))
        aoi_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs=crs)
        aoi_path = input_dir / "aoi.shp"
        aoi_gdf.to_file(aoi_path, driver="ESRI Shapefile")

        cfg = {
            "aoi": {"aoi": str(aoi_path)},
            "paths": {"output_dir": str(output_dir)},
            "raster": {"target_res": 30},
            "feature_sources": {"rasters": {}},
        }
        for key in soil_map.keys():
            cfg["feature_sources"]["rasters"][key] = {
                "url": str(input_dir / f"{key}.tif"),
                "resampling": "bilinear",
            }

        cfg_path = tmp / "smoke_config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        cmd = [
            sys.executable,
            str(cli_path),
            "--config",
            str(cfg_path),
            "--template",
            str(template_path),
            "--output-dir",
            str(output_dir),
            "--format",
            "both",
            "--overwrite",
        ]
        subprocess.run(cmd, check=True, cwd=root)

        manifest_path = output_dir / "soil_collection_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Missing manifest: {manifest_path}")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        layers = manifest.get("layers", [])
        if len(layers) != len(soil_map):
            raise RuntimeError(f"Unexpected layer count {len(layers)} != {len(soil_map)}")

        for field_name in soil_map.values():
            asc = output_dir / f"{field_name}.asc"
            tif = output_dir / f"{field_name}.tif"
            if not asc.exists():
                raise RuntimeError(f"Missing ASC output: {asc}")
            if not tif.exists():
                raise RuntimeError(f"Missing TIF output: {tif}")

        print("Smoke test passed:")
        print(f"- Manifest: {manifest_path}")
        print(f"- Layers: {len(layers)}")
        print(f"- Output dir: {output_dir}")


if __name__ == "__main__":
    main()
