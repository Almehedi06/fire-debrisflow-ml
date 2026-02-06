from __future__ import annotations

import argparse
import logging

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
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    cfg = load_config(args.config)
    outputs = run_raster_pipeline(cfg, cleanup_intermediates=not args.keep_intermediates)
    if args.raster_only:
        return
    run_landlab_pipeline(cfg, outputs)


if __name__ == "__main__":
    main()
