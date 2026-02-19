from __future__ import annotations

import argparse
from pathlib import Path

from soil_data.core import (
    fetch_soil_layers,
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
            "Run end-to-end soil workflow: fetch (+ AOI clip) then harmonize "
            "to one target grid for downstream analysis."
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
        "--raw-dir",
        default=None,
        help="Optional directory for fetched intermediate rasters. Default: <output-dir>/soil_raw",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Optional template raster to align against.",
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
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep temporary files in both fetch and harmonize stages.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_yaml(args.config)
    aoi = resolve_aoi_path(args.aoi, cfg)
    output_dir = resolve_output_dir(args.output_dir, cfg)
    raw_dir = Path(args.raw_dir) if args.raw_dir else (Path(output_dir) / "soil_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    keys = parse_soil_keys(args.soil_keys)
    specs = resolve_soil_specs(keys, cfg)
    target_res = float(args.target_res or cfg.get("raster", {}).get("target_res", 10.0))

    fetch_manifest = fetch_soil_layers(
        aoi_path=aoi,
        output_dir=raw_dir,
        specs=specs,
        clip_to_aoi=True,
        overwrite=args.overwrite,
        keep_intermediates=args.keep_intermediates,
    )
    print(f"Saved fetch manifest: {fetch_manifest}")

    harmonize_manifest = harmonize_soil_layers(
        aoi_path=aoi,
        output_dir=Path(output_dir),
        specs=specs,
        source_dir=raw_dir,
        template_path=Path(args.template) if args.template else None,
        target_res=target_res,
        dem_cfg=cfg.get("dem", {}),
        output_format=args.format,
        overwrite=args.overwrite,
        keep_intermediates=args.keep_intermediates,
    )
    print(f"Saved harmonize manifest: {harmonize_manifest}")
    print(f"Soil layers processed: {len(specs)}")


if __name__ == "__main__":
    main()
