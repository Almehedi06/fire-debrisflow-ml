from __future__ import annotations

import argparse
from pathlib import Path

from soil_data.core import (
    fetch_soil_layers,
    load_yaml,
    parse_soil_keys,
    resolve_aoi_path,
    resolve_output_dir,
    resolve_soil_specs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SOLUS soil layers and optionally clip them to AOI (no grid harmonization)."
    )
    parser.add_argument(
        "--config",
        default="config/base.yaml",
        help="Optional config path used for defaults.",
    )
    parser.add_argument("--aoi", default=None, help="AOI shapefile path.")
    parser.add_argument("--output-dir", default=None, help="Output directory for fetched TIFF files.")
    parser.add_argument(
        "--soil-keys",
        default=None,
        help="Comma-separated subset of soil keys.",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Do not clip to AOI after download/local resolve.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fetched TIFFs.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep temporary downloaded/intermediate files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_yaml(args.config)
    aoi = resolve_aoi_path(args.aoi, cfg)
    output_dir = resolve_output_dir(args.output_dir, cfg)

    keys = parse_soil_keys(args.soil_keys)
    specs = resolve_soil_specs(keys, cfg)

    manifest_path = fetch_soil_layers(
        aoi_path=aoi,
        output_dir=Path(output_dir),
        specs=specs,
        clip_to_aoi=not args.no_clip,
        overwrite=args.overwrite,
        keep_intermediates=args.keep_intermediates,
    )
    print(f"Saved fetch manifest: {manifest_path}")
    print(f"Soil layers fetched: {len(specs)}")


if __name__ == "__main__":
    main()
