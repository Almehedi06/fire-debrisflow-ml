from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio


def ensure_output_dir_writable(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = out_dir / ".write_test"
    with open(probe, "w") as f:
        f.write("ok\n")
    probe.unlink(missing_ok=True)
    return out_dir


def load_and_validate_aoi(aoi_path: str | Path) -> gpd.GeoDataFrame:
    path = Path(aoi_path)
    if not path.exists():
        raise FileNotFoundError(f"AOI not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"AOI has no features: {path}")
    if gdf.crs is None:
        raise ValueError(f"AOI has no CRS: {path}")
    if gdf.geometry.is_empty.all():
        raise ValueError(f"AOI geometries are empty: {path}")
    return gdf


def validate_raster_path(
    raster_path: str | Path,
    *,
    label: str = "Raster",
    require_crs: bool = True,
) -> Path:
    path = Path(raster_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    with rasterio.open(path) as src:
        if src.count < 1:
            raise ValueError(f"{label} has no bands: {path}")
        if src.width < 1 or src.height < 1:
            raise ValueError(f"{label} has invalid dimensions: {path}")
        if require_crs and src.crs is None:
            raise ValueError(f"{label} has no CRS: {path}")
    return path


def bounds_overlap(bounds_a, bounds_b) -> bool:
    left = max(bounds_a[0], bounds_b[0])
    bottom = max(bounds_a[1], bounds_b[1])
    right = min(bounds_a[2], bounds_b[2])
    top = min(bounds_a[3], bounds_b[3])
    return left < right and bottom < top


def validate_aoi_overlaps_raster(
    aoi_path: str | Path,
    raster_path: str | Path,
    *,
    label: str = "Raster",
) -> None:
    gdf = load_and_validate_aoi(aoi_path)
    validate_raster_path(raster_path, label=label, require_crs=True)

    with rasterio.open(raster_path) as src:
        raster_bounds = tuple(src.bounds)
        aoi_bounds = tuple(gdf.to_crs(src.crs).total_bounds)

    if not bounds_overlap(aoi_bounds, raster_bounds):
        raise ValueError(
            f"AOI does not overlap {label}: AOI={aoi_path} {aoi_bounds}, "
            f"{label}={raster_path} {raster_bounds}"
        )

