from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import rasterio
import yaml

from reproject_and_resample import (
    clip_raster_to_shape,
    convert_to_ascii,
    reproject_raster_to_match_crs,
)


def _load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_aoi(config_path: str | None, aoi_arg: str | None) -> str | None:
    if aoi_arg:
        return aoi_arg
    cfg = _load_config(config_path)
    return cfg.get("aoi", {}).get("aoi")


def _resolve_template(config_path: str | None, template_arg: str | None) -> Path | None:
    if template_arg:
        return Path(template_arg)
    if not config_path:
        return None
    cfg = _load_config(config_path)
    out_dir = cfg.get("paths", {}).get("output_dir")
    if not out_dir:
        return None
    candidates = [
        Path(out_dir) / "topographic__elevation.tif",
        Path(out_dir) / "topographic__elevation.asc",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _copy_or_clip(src: Path, dst: Path, aoi_path: str | None) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(src, dst)
    if not aoi_path:
        return

    clipped = clip_raster_to_shape(str(dst), aoi_path)
    # Replace the copied raster with the clipped raster.
    os.replace(clipped, dst)


def _validate_alignment(pre: rasterio.DatasetReader, post: rasterio.DatasetReader) -> None:
    if pre.crs != post.crs:
        raise ValueError(f"CRS mismatch: pre={pre.crs} post={post.crs}")
    if pre.transform != post.transform:
        raise ValueError("Transform mismatch between pre and post rasters.")
    if pre.width != post.width or pre.height != post.height:
        raise ValueError("Dimension mismatch between pre and post rasters.")


def _compute_diff(pre_tif: Path, post_tif: Path, diff_tif: Path) -> None:
    with rasterio.open(pre_tif) as pre, rasterio.open(post_tif) as post:
        _validate_alignment(pre, post)

        pre_arr = pre.read(1)
        post_arr = post.read(1)

        nodata = post.nodata if post.nodata is not None else pre.nodata
        if nodata is None:
            nodata = -9999.0

        mask = np.zeros(pre_arr.shape, dtype=bool)
        if pre.nodata is not None:
            mask |= pre_arr == pre.nodata
        if post.nodata is not None:
            mask |= post_arr == post.nodata
        mask |= np.isnan(pre_arr) | np.isnan(post_arr)

        diff = post_arr.astype("float32") - pre_arr.astype("float32")
        diff = np.where(mask, nodata, diff).astype("float32")

        meta = post.meta.copy()
        meta.update({"dtype": "float32", "count": 1, "nodata": nodata})
        with rasterio.open(diff_tif, "w", **meta) as dst:
            dst.write(diff, 1)


def _align_to_template(src_path: Path, template_path: Path, out_path: Path) -> None:
    with rasterio.open(template_path) as tmpl:
        target_epsg = tmpl.crs.to_epsg() if tmpl.crs else None
        if target_epsg is None:
            raise ValueError(f"Template has no EPSG CRS: {template_path}")
        template_meta = {
            "transform": tmpl.transform,
            "width": tmpl.width,
            "height": tmpl.height,
        }

    aligned = reproject_raster_to_match_crs(
        str(src_path),
        target_crs_epsg=target_epsg,
        resampling_method="bilinear",
        template_meta=template_meta,
    )
    os.replace(aligned, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute DEM difference (post - pre) and export ASCII grids."
    )
    parser.add_argument("--pre", required=True, help="Path to pre-event DEM GeoTIFF.")
    parser.add_argument("--post", required=True, help="Path to post-event DEM GeoTIFF.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config YAML; used to pull AOI if --aoi not provided.",
    )
    parser.add_argument(
        "--aoi",
        default=None,
        help="Optional AOI shapefile. If omitted and --config is provided, "
        "uses cfg['aoi']['aoi'] when available.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Optional template raster (e.g., topographic__elevation.tif/.asc) to "
        "snap pre/post DEMs to a common grid.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pre_src = Path(args.pre)
    post_src = Path(args.post)
    aoi_path = _resolve_aoi(args.config, args.aoi)
    template_path = _resolve_template(args.config, args.template)

    pre_tif = out_dir / "dem_pre.tif"
    post_tif = out_dir / "dem_post.tif"
    diff_tif = out_dir / "dem_diff.tif"

    _copy_or_clip(pre_src, pre_tif, aoi_path)
    _copy_or_clip(post_src, post_tif, aoi_path)

    if template_path is not None:
        _align_to_template(pre_tif, template_path, pre_tif)
        _align_to_template(post_tif, template_path, post_tif)

    _compute_diff(pre_tif, post_tif, diff_tif)

    pre_asc = Path(convert_to_ascii(str(pre_tif), str(out_dir)))
    post_asc = Path(convert_to_ascii(str(post_tif), str(out_dir)))
    diff_asc = Path(convert_to_ascii(str(diff_tif), str(out_dir)))

    print("Saved:", diff_tif)
    print("Saved:", diff_asc)
    print("Saved:", pre_asc, post_asc)


if __name__ == "__main__":
    main()
