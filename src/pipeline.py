from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Callable

import geopandas as gpd
import numpy as np
import rasterio
import yaml

from dem import fetch_dem
from downloads import download_file, extract_first_tif
from landlab_io import add_ascii_field, load_grid, read_nodata_value, write_ascii_field
from reproject_and_resample import (
    clip_raster_to_shape,
    convert_to_ascii,
    reproject_raster_to_match_crs,
    resample_raster,
)
from soil_features import (
    compute_fc_wp_arrays,
    compute_ksat,
    compute_saturated_water_content,
    compute_soil_density,
    compute_soil_texture,
    compute_transmissivity,
)
from vegetation_features import (
    adjust_internal_friction_angle,
    compute_cohesion,
    rootcohesion,
    vegtype,
)


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceSpec:
    key: str
    uri: str | list[str]
    resampling: str
    unzip: bool = False


@dataclass(frozen=True)
class FieldSpec:
    source_key: str
    field_name: str
    scale: float = 1.0
    offset: float = 0.0
    close_nodata: bool = True
    extra_close_values: list[float] | None = None
    rename_file: bool = True
    transform: Callable | None = None


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_utm_aoi(aoi_path: str) -> tuple[str, str]:
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        raise ValueError(f"AOI has no CRS: {aoi_path}")

    crs_wkt = gdf.crs.to_wkt()
    if "UTM zone" in crs_wkt:
        epsg_code = gdf.crs.to_epsg()
        crs = f"EPSG:{epsg_code}"
        LOG.info("AOI already in UTM: %s", crs)
        return aoi_path, crs

    centroid = gdf.to_crs(epsg=4326).geometry.unary_union.centroid
    zone = int((centroid.x + 180) / 6) + 1
    crs = f"EPSG:326{zone}"

    # Preserve notebook behavior: overwrite AOI on disk.
    gdf = gdf.to_crs(crs)
    gdf.to_file(aoi_path, driver="ESRI Shapefile")
    LOG.info("Reprojected AOI to %s and overwrote %s", crs, aoi_path)
    return aoi_path, crs


def build_sources_from_config(cfg: dict) -> list[SourceSpec]:
    sources: list[SourceSpec] = []

    for key, info in cfg.get("feature_sources", {}).get("rasters", {}).items():
        sources.append(
            SourceSpec(
                key=key,
                uri=info["url"],
                resampling=info.get("resampling", cfg["raster"]["resampling_method"]),
            )
        )

    for key, info in cfg.get("feature_sources", {}).get("landcover", {}).items():
        sources.append(
            SourceSpec(
                key=key,
                uri=info["url"],
                resampling=info.get("resampling", "nearest"),
                unzip=info.get("unzip", True),
            )
        )

    bs = cfg.get("burn_severity", {})
    src = bs.get("source", "local").lower()
    if src == "local":
        local_path = os.path.join(bs["local"]["path"], bs["local"]["filename"])
        sources.append(
            SourceSpec(
                key="burn_severity",
                uri=local_path,
                resampling=bs["local"].get("resampling", "nearest"),
            )
        )
    else:
        base_url = bs["remote"]["base_url"]
        fire_name_fmt = cfg["fire"]["name"].lower().replace(" ", "_")
        fire_id_fmt = cfg["fire"]["id"].lower()
        candidates = [
            pattern.format(
                base_url=base_url,
                fire_name_fmt=fire_name_fmt,
                fire_id_fmt=fire_id_fmt,
            )
            for pattern in bs["remote"]["candidates"]
        ]
        sources.append(
            SourceSpec(
                key="burn_severity",
                uri=candidates,
                resampling=bs["remote"].get("resampling", "nearest"),
                unzip=True,
            )
        )

    return sources


def _resolve_single_source(spec: SourceSpec, output_dir: str) -> str:
    uri = spec.uri
    if isinstance(uri, list):
        raise TypeError("Expected a single URI, got list")

    if uri.startswith("http"):
        local_path = os.path.join(output_dir, os.path.basename(uri))
        download_file(uri, local_path)
        if uri.lower().endswith(".zip") or spec.unzip:
            extract_dir = os.path.join(output_dir, f"unzipped_{spec.key}")
            return extract_first_tif(local_path, extract_dir)
        return local_path

    if not os.path.exists(uri):
        raise FileNotFoundError(f"Source not found: {uri}")

    if uri.lower().endswith(".zip"):
        extract_dir = os.path.join(output_dir, f"unzipped_{spec.key}")
        return extract_first_tif(uri, extract_dir)

    return uri


def resolve_source_to_tif(spec: SourceSpec, output_dir: str) -> str:
    if isinstance(spec.uri, list):
        last_err = None
        for candidate in spec.uri:
            try:
                return _resolve_single_source(
                    SourceSpec(
                        key=spec.key,
                        uri=candidate,
                        resampling=spec.resampling,
                        unzip=spec.unzip,
                    ),
                    output_dir,
                )
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(f"Failed to resolve {spec.key}. Last error: {last_err}")

    return _resolve_single_source(spec, output_dir)


def process_dem(
    dem_path: str,
    aoi_path: str,
    target_crs: str,
    target_resolution: float,
    output_dir: str,
) -> tuple[str, dict]:
    dem_reproj = reproject_raster_to_match_crs(
        str(dem_path),
        target_crs_epsg=target_crs.split(":")[1],
        resampling_method="cubic",
    )
    dem_clipped = clip_raster_to_shape(dem_reproj, aoi_path)
    dem_resampled = resample_raster(
        dem_clipped,
        template_meta=None,
        resampling_method="cubic",
        target_resolution=target_resolution,
    )
    dem_ascii = convert_to_ascii(dem_resampled, output_dir, template_meta=None)

    with rasterio.open(dem_resampled) as src:
        template_meta = src.meta.copy()

    return dem_ascii, template_meta


def _cleanup_intermediates(output_dir: str) -> None:
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        try:
            if os.path.isfile(item_path) and item_path.lower().endswith((".tif", ".zip")):
                os.remove(item_path)
            elif os.path.isdir(item_path) and item.startswith("unzipped_"):
                import shutil

                shutil.rmtree(item_path)
        except Exception as exc:
            LOG.warning("Could not delete %s: %s", item_path, exc)


def process_source(
    spec: SourceSpec,
    aoi_path: str,
    target_crs: str,
    template_meta: dict,
    target_resolution: float,
    output_dir: str,
    cleanup_intermediates: bool = True,
) -> str:
    src_path = resolve_source_to_tif(spec, output_dir)

    reprojected = reproject_raster_to_match_crs(
        src_path,
        target_crs_epsg=target_crs.split(":")[1],
        resampling_method=spec.resampling,
        template_meta=template_meta,
    )
    resampled = resample_raster(
        reprojected,
        template_meta=template_meta,
        resampling_method=spec.resampling,
        target_resolution=target_resolution,
    )
    clipped = clip_raster_to_shape(resampled, aoi_path, template_meta=template_meta)
    ascii_path = convert_to_ascii(clipped, output_dir, template_meta=template_meta)

    if cleanup_intermediates:
        _cleanup_intermediates(output_dir)

    return ascii_path


def run_raster_pipeline(cfg: dict, cleanup_intermediates: bool = True) -> dict:
    aoi_path, target_crs = ensure_utm_aoi(cfg["aoi"]["aoi"])
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    dem_path = fetch_dem(aoi_path, cfg.get("dem", {}), output_dir)
    dem_ascii, template_meta = process_dem(
        dem_path,
        aoi_path,
        target_crs,
        cfg["raster"]["target_res"],
        output_dir,
    )

    outputs = {"dem": dem_ascii}
    for spec in build_sources_from_config(cfg):
        try:
            outputs[spec.key] = process_source(
                spec,
                aoi_path,
                target_crs,
                template_meta,
                cfg["raster"]["target_res"],
                output_dir,
                cleanup_intermediates=cleanup_intermediates,
            )
        except Exception as exc:
            LOG.warning("Failed processing %s: %s", spec.key, exc)

    return outputs


def _burn_transform(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).copy()
    values[~np.isin(values, [2, 3, 4])] = 1
    return values


def run_landlab_pipeline(cfg: dict, outputs: dict, strict: bool = True):
    output_dir = cfg["paths"]["output_dir"]

    if "dem" not in outputs:
        raise ValueError("Missing DEM output; cannot build master grid.")

    grid = load_grid(outputs["dem"], "topographic__elevation")
    nodata_val = read_nodata_value(outputs["dem"])
    grid.set_nodata_nodes_to_closed(grid.at_node["topographic__elevation"], nodata_val)
    os.rename(outputs["dem"], os.path.join(output_dir, "topographic__elevation.asc"))

    field_map = {
        "cec7_0_cm": FieldSpec("cec7_0_cm", "cation__exchange_capacity", scale=0.1),
        "anylithicdpt_cm": FieldSpec("anylithicdpt_cm", "soil__thickness", scale=0.01),
        "claytotal_0_cm": FieldSpec("claytotal_0_cm", "clay__total"),
        "ph1to1h2o_0_cm": FieldSpec("ph1to1h2o_0_cm", "pH", scale=0.01),
        "sandtotal_0_cm": FieldSpec("sandtotal_0_cm", "sand__total"),
        "silttotal_0_cm": FieldSpec("silttotal_0_cm", "silt__total"),
        "dbovendry_0_cm": FieldSpec("dbovendry_0_cm", "dry__bulk_density", scale=0.01),
        "burn_severity": FieldSpec(
            "burn_severity",
            "burn__severity",
            close_nodata=False,
            transform=_burn_transform,
        ),
    }

    for source_key, spec in field_map.items():
        if source_key not in outputs:
            if strict:
                raise ValueError(f"Missing required source: {source_key}")
            LOG.warning("Skipping missing source: %s", source_key)
            continue
        add_ascii_field(
            grid,
            outputs[source_key],
            spec.field_name,
            scale=spec.scale,
            offset=spec.offset,
            close_nodata=spec.close_nodata,
            extra_close_values=spec.extra_close_values,
            rename_file=spec.rename_file,
            transform=spec.transform,
        )

    landcover_keys = list(cfg.get("feature_sources", {}).get("landcover", {}).keys())
    for key in landcover_keys:
        if key in outputs:
            add_ascii_field(
                grid,
                outputs[key],
                "landcover",
                extra_close_values=[11],
            )
            break
    else:
        if strict and landcover_keys:
            raise ValueError("Missing landcover source output.")

    required = [
        "pH",
        "clay__total",
        "silt__total",
        "cation__exchange_capacity",
        "soil__thickness",
        "dry__bulk_density",
        "sand__total",
    ]
    for name in required:
        if name not in grid.at_node:
            raise ValueError(f"Missing required field: {name}")

    ksat = compute_ksat(
        grid.at_node["pH"],
        grid.at_node["clay__total"],
        grid.at_node["silt__total"],
        grid.at_node["cation__exchange_capacity"],
    )
    grid.add_field(
        "soil__saturated_hydraulic_conductivity",
        (ksat / 100) * 10,
        at="node",
        clobber=True,
    )
    write_ascii_field(
        os.path.join(output_dir, "soil__saturated_hydraulic_conductivity.asc"),
        grid,
        "soil__saturated_hydraulic_conductivity",
    )

    transmissivity = compute_transmissivity(
        grid.at_node["soil__saturated_hydraulic_conductivity"],
        grid.at_node["soil__thickness"],
    )
    grid.add_field("soil__transmissivity", transmissivity, at="node", clobber=True)
    write_ascii_field(
        os.path.join(output_dir, "soil__transmissivity.asc"),
        grid,
        "soil__transmissivity",
    )

    wsat = compute_saturated_water_content(
        grid.at_node["dry__bulk_density"],
        grid.at_node["clay__total"],
        grid.at_node["silt__total"],
    )
    grid.add_field("saturated__water_content", wsat, at="node", clobber=True)
    write_ascii_field(
        os.path.join(output_dir, "saturated__water_content.asc"),
        grid,
        "saturated__water_content",
    )

    soil_texture = compute_soil_texture(
        grid.at_node["sand__total"],
        grid.at_node["silt__total"],
        grid.at_node["clay__total"],
    )
    grid.add_field("soil__texture", soil_texture, at="node", clobber=True)
    write_ascii_field(os.path.join(output_dir, "soil__texture.asc"), grid, "soil__texture")

    porosity, theta_fc, theta_wp, phi = compute_fc_wp_arrays(
        grid.at_node["soil__texture"],
        grid.at_node["saturated__water_content"],
    )
    grid.add_field("field__capacity", theta_fc, at="node", clobber=True)
    grid.add_field("wilting__point", theta_wp, at="node", clobber=True)
    grid.add_field("porosity", porosity, at="node", clobber=True)
    grid.add_field("soil__internal_friction_angle", phi, at="node", clobber=True)

    if "landcover" in grid.at_node:
        adjusted_phi = adjust_internal_friction_angle(
            grid.at_node["landcover"],
            grid.at_node["soil__internal_friction_angle"],
        )
        grid.at_node["soil__internal_friction_angle"] = adjusted_phi

    write_ascii_field(os.path.join(output_dir, "field__capacity.asc"), grid, "field__capacity")
    write_ascii_field(os.path.join(output_dir, "wilting__point.asc"), grid, "wilting__point")
    write_ascii_field(os.path.join(output_dir, "porosity.asc"), grid, "porosity")
    write_ascii_field(
        os.path.join(output_dir, "soil__internal_friction_angle.asc"),
        grid,
        "soil__internal_friction_angle",
    )

    density = compute_soil_density(
        grid.at_node["dry__bulk_density"], grid.at_node["porosity"]
    )
    grid.add_field("soil__density", density, at="node", clobber=True)
    write_ascii_field(os.path.join(output_dir, "soil__density.asc"), grid, "soil__density")

    if "landcover" in grid.at_node:
        landcover = grid.at_node["landcover"]
        landcover_color = rootcohesion(landcover, 0, 1, 2, 3, 4, 5)
        grid.add_field("landcovercolor", landcover_color, at="node", clobber=True)

        c_min, c_mode, c_max = compute_cohesion(landcover)
        grid.add_field("soil__minimum_total_cohesion", c_min, at="node", clobber=True)
        grid.add_field("soil__maximum_total_cohesion", c_max, at="node", clobber=True)
        grid.add_field("soil__mode_total_cohesion", c_mode, at="node", clobber=True)

        write_ascii_field(
            os.path.join(output_dir, "soil__minimum_total_cohesion.asc"),
            grid,
            "soil__minimum_total_cohesion",
        )
        write_ascii_field(
            os.path.join(output_dir, "soil__maximum_total_cohesion.asc"),
            grid,
            "soil__maximum_total_cohesion",
        )
        write_ascii_field(
            os.path.join(output_dir, "soil__mode_total_cohesion.asc"),
            grid,
            "soil__mode_total_cohesion",
        )

        vegetation_type = vegtype(landcover, -9999.0, 3, 2, 1, 0)
        grid.add_field(
            "vegetation__plant_functional_type",
            vegetation_type,
            at="node",
            clobber=True,
        )
        write_ascii_field(
            os.path.join(output_dir, "vegetation__plant_functional_type.asc"),
            grid,
            "vegetation__plant_functional_type",
        )

    return grid


def run_pipeline(config_path: str, cleanup_intermediates: bool = True):
    cfg = load_config(config_path)
    outputs = run_raster_pipeline(cfg, cleanup_intermediates=cleanup_intermediates)
    grid = run_landlab_pipeline(cfg, outputs)
    return outputs, grid
