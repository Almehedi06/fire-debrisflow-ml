from __future__ import annotations

from typing import Iterable

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, transform_geom
import fiona


def _get_resampling_enum(method: str) -> Resampling:
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "mode": Resampling.mode,
    }
    return mapping.get((method or "nearest").lower(), Resampling.nearest)


def _read_shapes(
    shapefile_path: str,
    raster_crs,
    reproject_shapes: bool = True,
) -> list[dict]:
    with fiona.open(shapefile_path, "r") as src:
        shapes = [feature["geometry"] for feature in src]
        if not reproject_shapes:
            return shapes

        src_crs = src.crs_wkt or src.crs
        if not src_crs or raster_crs is None:
            return shapes

        src_crs_obj = CRS.from_user_input(src_crs)
        dst_crs_obj = CRS.from_user_input(raster_crs)
        if src_crs_obj == dst_crs_obj:
            return shapes

        return [transform_geom(src_crs_obj, dst_crs_obj, geom) for geom in shapes]


def reproject_raster_to_match_crs(
    src_path: str,
    target_crs_epsg: str | int,
    resampling_method: str,
    template_meta: dict | None = None,
) -> str:
    resampling_enum = _get_resampling_enum(resampling_method)

    with rasterio.open(src_path) as src:
        dst_crs = f"EPSG:{target_crs_epsg}"
        if template_meta is None:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
        else:
            # Snap to template grid.
            transform = template_meta["transform"]
            width = template_meta["width"]
            height = template_meta["height"]

        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        reprojected_path = src_path.replace(".tif", f"_reproj_{target_crs_epsg}.tif")
        with rasterio.open(reprojected_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_enum,
                )
    return reprojected_path


def clip_raster_to_shape(
    raster_path: str,
    shapefile_path: str,
    template_meta: dict | None = None,
    reproject_shapes: bool = True,
) -> str:
    with rasterio.open(raster_path) as src:
        shapes = _read_shapes(shapefile_path, src.crs, reproject_shapes=reproject_shapes)
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()

    if template_meta is not None:
        # Snap clipped raster to DEM grid exactly.
        out_meta.update(
            {
                "driver": "GTiff",
                "height": template_meta["height"],
                "width": template_meta["width"],
                "transform": template_meta["transform"],
            }
        )
    else:
        # For DEM (first raster).
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

    clipped_path = raster_path.replace(".tif", "_clipped.tif")
    with rasterio.open(clipped_path, "w", **out_meta) as dest:
        dest.write(out_image)
    return clipped_path


def resample_raster(
    raster_path: str,
    template_meta: dict | None = None,
    resampling_method: str = "nearest",
    target_resolution: float | None = None,
) -> str:
    resampling_enum = _get_resampling_enum(resampling_method)

    with rasterio.open(raster_path) as src:
        kwargs = src.meta.copy()

        if template_meta is not None:
            # Snap resampled grid to DEM alignment.
            transform = template_meta["transform"]
            width = template_meta["width"]
            height = template_meta["height"]
        else:
            transform = src.transform
            width = src.width
            height = src.height
            if target_resolution is not None:
                scale_x = src.res[0] / target_resolution
                scale_y = src.res[1] / target_resolution
                width = int(src.width * scale_x)
                height = int(src.height * scale_y)
                transform = rasterio.Affine(
                    target_resolution,
                    transform.b,
                    transform.c,
                    transform.d,
                    -target_resolution,
                    transform.f,
                )

        kwargs.update({"transform": transform, "width": width, "height": height})

        resampled_path = raster_path.replace(".tif", "_resampled.tif")
        with rasterio.open(resampled_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=resampling_enum,
                )
    return resampled_path


def convert_to_ascii(tif_path: str, out_dir: str, template_meta: dict | None = None) -> str:
    import os

    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(tif_path) as src:
        array = src.read(1)
        meta = template_meta if template_meta is not None else src.meta

        transform = meta["transform"]
        width = meta["width"]
        height = meta["height"]

        xllcorner = transform[2]
        yllcorner = (
            transform[5] + transform[4] * (height - 1)
            if transform[4] < 0
            else transform[5] - transform[4] * (height - 1)
        )
        cellsize = abs(transform[0])

        nodata_value = src.nodata if src.nodata is not None else -9999.0

        ascii_path = os.path.join(out_dir, os.path.basename(tif_path).replace(".tif", ".asc"))
        with open(ascii_path, "w") as f:
            f.write(f"ncols         {width}\n")
            f.write(f"nrows         {height}\n")
            f.write(f"xllcorner     {xllcorner}\n")
            f.write(f"yllcorner     {yllcorner}\n")
            f.write(f"cellsize      {cellsize}\n")
            f.write(f"NODATA_value  {nodata_value}\n")
            for row in array:
                row_out = [str(nodata_value) if np.isnan(v) else str(v) for v in row]
                f.write(" ".join(row_out) + "\n")
    return ascii_path


def read_ascii_header(path: str) -> dict[str, float]:
    header: dict[str, float] = {}
    with open(path, "r") as f:
        for _ in range(6):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) >= 2:
                header[parts[0].lower()] = float(parts[1])
    return header


def sanity_check_ascii(paths: Iterable[str], tol: float = 1e-6, strict: bool = True) -> None:
    paths = list(paths)
    if not paths:
        raise ValueError("No ASCII paths provided.")
    headers = {p: read_ascii_header(p) for p in paths}

    ref_path = paths[0]
    ref = headers[ref_path]
    keys = ["ncols", "nrows", "cellsize", "xllcorner", "yllcorner"]

    mismatches = []
    for p, h in headers.items():
        for k in keys:
            if k not in h or k not in ref:
                mismatches.append((p, k, "missing", "missing"))
                break
            v = h[k]
            rv = ref[k]
            if abs(v - rv) > tol:
                mismatches.append((p, k, v, rv))
                break

    if mismatches:
        print(f"Grid mismatch vs template: {ref_path}")
        for p, k, v, rv in mismatches:
            print(f"- {p} -> {k}: {v} (ref {rv})")
        if strict:
            raise ValueError("ASCII grid mismatch")
    else:
        print(f"All {len(paths)} ASCII files match the template grid.")
