from __future__ import annotations

import argparse
from pathlib import Path

from soil_data.core import (
    harmonize_soil_layers,
    load_yaml,
    parse_soil_keys,
    resolve_aoi_path,
    resolve_output_dir,
    resolve_soil_specs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Harmonize soil rasters to a single grid (same CRS/resolution/transform/shape) "
            "and export ASC/TIF outputs."
        )
    )
    parser.add_argument(
        "--config",
        default="config/base.yaml",
        help="Optional config path used for defaults.",
    )
    parser.add_argument("--aoi", default=None, help="AOI shapefile path.")
    parser.add_argument("--output-dir", default=None, help="Output directory for harmonized layers.")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional directory containing fetched source rasters named <soil_key>.tif.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Optional template raster to align against (preferred when available).",
    )
    parser.add_argument(
        "--target-res",
        type=float,
        default=None,
        help="Target resolution in meters when template is not provided.",
    )
    parser.add_argument(
        "--soil-keys",
        default=None,
        help="Comma-separated subset of soil keys.",
    )
    parser.add_argument(
        "--format",
        choices=["asc", "tif", "both"],
        default="both",
        help="Output format for harmonized layers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing harmonized outputs.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep temporary reproject/resample/clip files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_yaml(args.config)
    aoi = resolve_aoi_path(args.aoi, cfg)
    output_dir = resolve_output_dir(args.output_dir, cfg)

    keys = parse_soil_keys(args.soil_keys)
    specs = resolve_soil_specs(keys, cfg)
    target_res = float(args.target_res or cfg.get("raster", {}).get("target_res", 10.0))

    manifest_path = harmonize_soil_layers(
        aoi_path=aoi,
        output_dir=Path(output_dir),
        specs=specs,
        source_dir=Path(args.source_dir) if args.source_dir else None,
        template_path=Path(args.template) if args.template else None,
        target_res=target_res,
        dem_cfg=cfg.get("dem", {}),
        output_format=args.format,
        overwrite=args.overwrite,
        keep_intermediates=args.keep_intermediates,
    )
    print(f"Saved harmonize manifest: {manifest_path}")
    print(f"Soil layers harmonized: {len(specs)}")


if __name__ == "__main__":
    main()
