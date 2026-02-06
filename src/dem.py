from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from bmi_topography import Topography


def fetch_dem(aoi_shapefile: str, dem_cfg: dict, output_dir: str) -> Path:
    bounds = gpd.read_file(aoi_shapefile).to_crs(epsg=4326).total_bounds
    west, south, east, north = bounds
    buffer_deg = dem_cfg.get("buffer_deg", 0.05)

    topo = Topography(
        dem_type=dem_cfg.get("dem_type", "USGS10m"),
        south=south - buffer_deg,
        north=north + buffer_deg,
        west=west - buffer_deg,
        east=east + buffer_deg,
        output_format=dem_cfg.get("output_format", "GTiff"),
        cache_dir=dem_cfg.get("cache_dir", output_dir),
        api_key=dem_cfg.get("api_key"),
    )
    return Path(topo.fetch())