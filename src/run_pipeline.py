from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd

from export_ascii_to_tif import export_ascii_dir_to_tifs
from pipeline import load_config, run_landlab_pipeline, run_raster_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run debris-flow raster/landlab pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. config/base.yaml).",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep intermediate .tif/.zip files in the output directory.",
    )
    parser.add_argument(
        "--raster-only",
        action="store_true",
        help="Only run raster processing; skip Landlab feature generation.",
    )
    parser.add_argument(
        "--export-final-tifs",
        action="store_true",
        help="Export clean GeoTIFFs for DEM and landcover layers.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING).",
    )
    return parser.parse_args()


def _resolve_output_crs(cfg: dict) -> str | None:
    aoi_path = cfg.get("aoi", {}).get("aoi")
    if not aoi_path:
        return None
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        return None
    epsg = gdf.crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"
    return gdf.crs.to_wkt()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    cfg = load_config(args.config)
    outputs = run_raster_pipeline(
        cfg,
        cleanup_intermediates=not args.keep_intermediates,
    )
    if args.raster_only:
        if args.export_final_tifs:
            crs = _resolve_output_crs(cfg)
            export_ascii_dir_to_tifs(
                Path(cfg["paths"]["output_dir"]),
                overwrite=True,
                crs=crs,
            )
        return
    run_landlab_pipeline(cfg, outputs)
    if args.export_final_tifs:
        crs = _resolve_output_crs(cfg)
        export_ascii_dir_to_tifs(
            Path(cfg["paths"]["output_dir"]),
            overwrite=True,
            crs=crs,
        )


if __name__ == "__main__":
    main()
