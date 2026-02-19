from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import geopandas as gpd
import rasterio
import yaml


MANIFEST_VERSION = "1.0"


@dataclass(frozen=True)
class SoilVarSpec:
    key: str
    field_name: str
    url: str
    resampling: str = "bilinear"


@dataclass(frozen=True)
class RasterSourceSpec:
    key: str
    uri: str | list[str]
    resampling: str
    unzip: bool = False


DEFAULT_SOIL_SPECS: dict[str, SoilVarSpec] = {
    "cec7_0_cm": SoilVarSpec(
        key="cec7_0_cm",
        field_name="cation__exchange_capacity",
        url="https://storage.googleapis.com/solus100pub/cec7_0_cm_p.tif",
        resampling="bilinear",
    ),
    "anylithicdpt_cm": SoilVarSpec(
        key="anylithicdpt_cm",
        field_name="soil__thickness",
        url="https://storage.googleapis.com/solus100pub/anylithicdpt_cm_p.tif",
        resampling="bilinear",
    ),
    "claytotal_0_cm": SoilVarSpec(
        key="claytotal_0_cm",
        field_name="clay__total",
        url="https://storage.googleapis.com/solus100pub/claytotal_0_cm_p.tif",
        resampling="bilinear",
    ),
    "ph1to1h2o_0_cm": SoilVarSpec(
        key="ph1to1h2o_0_cm",
        field_name="pH",
        url="https://storage.googleapis.com/solus100pub/ph1to1h2o_0_cm_p.tif",
        resampling="bilinear",
    ),
    "sandtotal_0_cm": SoilVarSpec(
        key="sandtotal_0_cm",
        field_name="sand__total",
        url="https://storage.googleapis.com/solus100pub/sandtotal_0_cm_p.tif",
        resampling="bilinear",
    ),
    "silttotal_0_cm": SoilVarSpec(
        key="silttotal_0_cm",
        field_name="silt__total",
        url="https://storage.googleapis.com/solus100pub/silttotal_0_cm_p.tif",
        resampling="bilinear",
    ),
    "dbovendry_0_cm": SoilVarSpec(
        key="dbovendry_0_cm",
        field_name="dry__bulk_density",
        url="https://storage.googleapis.com/solus100pub/dbovendry_0_cm_p.tif",
        resampling="bilinear",
    ),
}


def load_yaml(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def parse_soil_keys(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_SOIL_SPECS.keys())
    items = [x.strip() for x in raw.split(",")]
    keys = [x for x in items if x]
    if not keys:
        raise ValueError("No keys provided after parsing --soil-keys.")
    unknown = [k for k in keys if k not in DEFAULT_SOIL_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown soil keys: {unknown}. Valid keys: {list(DEFAULT_SOIL_SPECS.keys())}"
        )
    return keys


def resolve_soil_specs(keys: list[str], cfg: dict) -> list[SoilVarSpec]:
    cfg_rasters = cfg.get("feature_sources", {}).get("rasters", {})
    resolved: list[SoilVarSpec] = []
    for key in keys:
        base = DEFAULT_SOIL_SPECS[key]
        cfg_info = cfg_rasters.get(key, {})
        resolved.append(
            SoilVarSpec(
                key=base.key,
                field_name=base.field_name,
                url=cfg_info.get("url", base.url),
                resampling=cfg_info.get("resampling", base.resampling),
            )
        )
    return resolved


def resolve_aoi_path(aoi_arg: str | None, cfg: dict) -> Path:
    aoi_raw = aoi_arg or cfg.get("aoi", {}).get("aoi")
    if not aoi_raw:
        raise ValueError("AOI path is required via --aoi or config['aoi']['aoi'].")
    aoi = Path(aoi_raw)
    if not aoi.exists():
        raise FileNotFoundError(f"AOI not found: {aoi}")
    return aoi


def resolve_output_dir(output_dir_arg: str | None, cfg: dict) -> Path:
    out_dir_raw = output_dir_arg or cfg.get("paths", {}).get("output_dir")
    if not out_dir_raw:
        raise ValueError(
            "Output dir is required via --output-dir or config['paths']['output_dir']."
        )
    out_dir = Path(out_dir_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _infer_utm_crs(aoi_path: Path) -> str:
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        raise ValueError(f"AOI has no CRS: {aoi_path}")

    centroid = gdf.to_crs(epsg=4326).geometry.unary_union.centroid
    zone = int((centroid.x + 180) / 6) + 1
    epsg_base = 326 if centroid.y >= 0 else 327
    return f"EPSG:{epsg_base}{zone:02d}"


def _resolve_single_source(spec: RasterSourceSpec, output_dir: Path) -> Path:
    from downloads import download_file, extract_first_tif

    uri = spec.uri
    if isinstance(uri, list):
        raise TypeError("Expected a single URI, got list.")

    if uri.startswith("http"):
        local_path = output_dir / Path(uri).name
        download_file(uri, str(local_path))
        if uri.lower().endswith(".zip") or spec.unzip:
            extract_dir = output_dir / f"unzipped_{spec.key}"
            return Path(extract_first_tif(str(local_path), str(extract_dir)))
        return local_path

    local = Path(uri)
    if not local.exists():
        raise FileNotFoundError(f"Source not found: {local}")
    if local.suffix.lower() == ".zip":
        extract_dir = output_dir / f"unzipped_{spec.key}"
        return Path(extract_first_tif(str(local), str(extract_dir)))
    return local


def _resolve_source_to_tif(spec: RasterSourceSpec, output_dir: Path) -> Path:
    if isinstance(spec.uri, list):
        last_error: Exception | None = None
        for candidate in spec.uri:
            try:
                return _resolve_single_source(
                    RasterSourceSpec(
                        key=spec.key,
                        uri=candidate,
                        resampling=spec.resampling,
                        unzip=spec.unzip,
                    ),
                    output_dir,
                )
            except Exception as exc:  # pragma: no cover - best-effort fallback chain
                last_error = exc
                continue
        raise RuntimeError(f"Failed resolving source {spec.key}. Last error: {last_error}")
    return _resolve_single_source(spec, output_dir)


def _cleanup_intermediates(output_dir: Path, keep_paths: set[Path]) -> None:
    keep_resolved = {p.resolve() for p in keep_paths if p.exists()}
    for item in output_dir.iterdir():
        resolved = item.resolve()
        if resolved in keep_resolved:
            continue

        if item.is_dir() and item.name.startswith("unzipped_"):
            shutil.rmtree(item, ignore_errors=True)
            continue

        if not item.is_file():
            continue

        name = item.name.lower()
        is_intermediate = (
            name.endswith(".zip")
            or "_reproj_" in name
            or "_resampled" in name
            or "_clipped" in name
            or name.startswith("usgs")
        )
        if is_intermediate:
            item.unlink(missing_ok=True)


def _build_template_from_dem(
    aoi_path: Path,
    output_dir: Path,
    target_crs: str,
    target_res: float,
    dem_cfg: dict,
    write_template_asc: bool,
) -> tuple[Path, dict]:
    from dem import fetch_dem
    from reproject_and_resample import (
        clip_raster_to_shape,
        convert_to_ascii,
        reproject_raster_to_match_crs,
        resample_raster,
    )

    dem_path = fetch_dem(str(aoi_path), dem_cfg, str(output_dir))
    epsg_code = target_crs.split(":")[1]

    dem_reproj = reproject_raster_to_match_crs(
        str(dem_path),
        target_crs_epsg=epsg_code,
        resampling_method="cubic",
    )
    dem_clipped = clip_raster_to_shape(dem_reproj, str(aoi_path))
    dem_resampled = resample_raster(
        dem_clipped,
        template_meta=None,
        resampling_method="cubic",
        target_resolution=target_res,
    )

    template_tif = output_dir / "topographic__elevation.tif"
    shutil.copyfile(dem_resampled, template_tif)
    with rasterio.open(template_tif) as src:
        template_meta = src.meta.copy()

    if write_template_asc:
        convert_to_ascii(str(template_tif), str(output_dir), template_meta=template_meta)

    return template_tif, template_meta


def _load_template_meta(template_path: Path) -> tuple[str, dict]:
    with rasterio.open(template_path) as src:
        epsg = src.crs.to_epsg() if src.crs else None
        if epsg is None:
            raise ValueError(f"Template CRS must include EPSG code: {template_path}")
        target_crs = f"EPSG:{epsg}"
        template_meta = src.meta.copy()
    return target_crs, template_meta


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def _manifest_common(stage: str, aoi: Path, output_dir: Path, soil_keys: list[str], options: dict) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "aoi": str(aoi),
        "output_dir": str(output_dir),
        "soil_keys": soil_keys,
        "options": options,
    }


def fetch_soil_layers(
    *,
    aoi_path: Path,
    output_dir: Path,
    specs: list[SoilVarSpec],
    clip_to_aoi: bool = True,
    overwrite: bool = False,
    keep_intermediates: bool = False,
) -> Path:
    try:
        from reproject_and_resample import clip_raster_to_shape
    except ModuleNotFoundError as exc:
        if exc.name == "fiona":
            raise ImportError(
                "Missing dependency 'fiona'. Install dependencies first "
                "(recommended: conda env create -f environment.yml)."
            ) from exc
        raise

    output_dir.mkdir(parents=True, exist_ok=True)

    keep_paths: set[Path] = set()
    layers: list[dict] = []

    for spec in specs:
        final_tif = output_dir / f"{spec.key}.tif"
        if final_tif.exists() and not overwrite:
            keep_paths.add(final_tif)
            layers.append(
                {
                    "key": spec.key,
                    "field_name": spec.field_name,
                    "source_uri": spec.url,
                    "resampling": spec.resampling,
                    "fetched_tif": str(final_tif),
                    "status": "skipped_existing",
                }
            )
            continue

        src_tif = _resolve_source_to_tif(
            RasterSourceSpec(key=spec.key, uri=spec.url, resampling=spec.resampling),
            output_dir,
        )
        out_tif = src_tif
        if clip_to_aoi:
            out_tif = Path(clip_raster_to_shape(str(src_tif), str(aoi_path)))

        if final_tif.exists():
            final_tif.unlink()
        out_tif.replace(final_tif)
        keep_paths.add(final_tif)

        layers.append(
            {
                "key": spec.key,
                "field_name": spec.field_name,
                "source_uri": spec.url,
                "resampling": spec.resampling,
                "fetched_tif": str(final_tif),
                "status": "created",
            }
        )

    if not keep_intermediates:
        _cleanup_intermediates(output_dir, keep_paths)

    manifest = _manifest_common(
        stage="fetch",
        aoi=aoi_path,
        output_dir=output_dir,
        soil_keys=[s.key for s in specs],
        options={
            "clip_to_aoi": clip_to_aoi,
            "overwrite": overwrite,
            "keep_intermediates": keep_intermediates,
        },
    )
    manifest["layers"] = layers
    return _write_json(output_dir / "soil_fetch_manifest.json", manifest)


def _asc_to_tif(
    asc_path: Path,
    tif_path: Path,
    crs: str | None = None,
    overwrite: bool = True,
) -> None:
    if tif_path.exists() and not overwrite:
        return

    with rasterio.open(asc_path) as src:
        data = src.read(1)
        meta = src.meta.copy()

    meta.update(driver="GTiff", count=1)
    if crs:
        meta.update(crs=crs)

    with rasterio.open(tif_path, "w", **meta) as dst:
        dst.write(data, 1)


def _harmonize_source_to_ascii(
    source_tif: Path,
    *,
    aoi_path: Path,
    target_crs: str,
    template_meta: dict,
    target_res: float,
    work_dir: Path,
    resampling: str,
) -> Path:
    from reproject_and_resample import (
        clip_raster_to_shape,
        convert_to_ascii,
        reproject_raster_to_match_crs,
        resample_raster,
    )

    epsg_code = target_crs.split(":")[1]
    reprojected = reproject_raster_to_match_crs(
        str(source_tif),
        target_crs_epsg=epsg_code,
        resampling_method=resampling,
        template_meta=template_meta,
    )
    resampled = resample_raster(
        reprojected,
        template_meta=template_meta,
        resampling_method=resampling,
        target_resolution=target_res,
    )
    clipped = clip_raster_to_shape(resampled, str(aoi_path), template_meta=template_meta)
    return Path(convert_to_ascii(clipped, str(work_dir), template_meta=template_meta))


def harmonize_soil_layers(
    *,
    aoi_path: Path,
    output_dir: Path,
    specs: list[SoilVarSpec],
    source_dir: Path | None = None,
    template_path: Path | None = None,
    target_res: float = 10.0,
    dem_cfg: dict | None = None,
    output_format: str = "both",
    overwrite: bool = False,
    keep_intermediates: bool = False,
) -> Path:
    if output_format not in {"asc", "tif", "both"}:
        raise ValueError(f"Unsupported output_format: {output_format}")

    try:
        from reproject_and_resample import reproject_raster_to_match_crs  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "fiona":
            raise ImportError(
                "Missing dependency 'fiona'. Install dependencies first "
                "(recommended: conda env create -f environment.yml)."
            ) from exc
        raise

    output_dir.mkdir(parents=True, exist_ok=True)
    export_asc = output_format in {"asc", "both"}
    export_tif = output_format in {"tif", "both"}

    created_template = False
    if template_path is not None:
        if not template_path.exists():
            raise FileNotFoundError(f"Template raster not found: {template_path}")
        target_crs, template_meta = _load_template_meta(template_path)
    else:
        target_crs = _infer_utm_crs(aoi_path)
        template_path, template_meta = _build_template_from_dem(
            aoi_path=aoi_path,
            output_dir=output_dir,
            target_crs=target_crs,
            target_res=target_res,
            dem_cfg=dem_cfg or {},
            write_template_asc=export_asc,
        )
        created_template = True

    keep_paths: set[Path] = set()
    layers: list[dict] = []

    for spec in specs:
        source_tif = None
        if source_dir is not None:
            candidate = source_dir / f"{spec.key}.tif"
            if candidate.exists():
                source_tif = candidate
        if source_tif is None:
            source_tif = _resolve_source_to_tif(
                RasterSourceSpec(key=spec.key, uri=spec.url, resampling=spec.resampling),
                output_dir,
            )

        out_asc = output_dir / f"{spec.field_name}.asc"
        out_tif = output_dir / f"{spec.field_name}.tif"

        if not overwrite:
            if export_asc and export_tif and out_asc.exists() and out_tif.exists():
                keep_paths.update({out_asc, out_tif})
                layers.append(
                    {
                        "key": spec.key,
                        "field_name": spec.field_name,
                        "source_uri": str(source_tif),
                        "resampling": spec.resampling,
                        "asc": str(out_asc),
                        "tif": str(out_tif),
                        "status": "skipped_existing",
                    }
                )
                continue
            if export_asc and not export_tif and out_asc.exists():
                keep_paths.add(out_asc)
                layers.append(
                    {
                        "key": spec.key,
                        "field_name": spec.field_name,
                        "source_uri": str(source_tif),
                        "resampling": spec.resampling,
                        "asc": str(out_asc),
                        "tif": None,
                        "status": "skipped_existing",
                    }
                )
                continue
            if export_tif and not export_asc and out_tif.exists():
                keep_paths.add(out_tif)
                layers.append(
                    {
                        "key": spec.key,
                        "field_name": spec.field_name,
                        "source_uri": str(source_tif),
                        "resampling": spec.resampling,
                        "asc": None,
                        "tif": str(out_tif),
                        "status": "skipped_existing",
                    }
                )
                continue

        raw_asc = _harmonize_source_to_ascii(
            source_tif,
            aoi_path=aoi_path,
            target_crs=target_crs,
            template_meta=template_meta,
            target_res=target_res,
            work_dir=output_dir,
            resampling=spec.resampling,
        )

        if out_asc.exists() and overwrite:
            out_asc.unlink()
        if raw_asc != out_asc:
            raw_asc.replace(out_asc)

        if export_tif:
            _asc_to_tif(out_asc, out_tif, crs=target_crs, overwrite=True)
            keep_paths.add(out_tif)

        if export_asc:
            keep_paths.add(out_asc)
            manifest_asc = str(out_asc)
        else:
            out_asc.unlink(missing_ok=True)
            manifest_asc = None

        layers.append(
            {
                "key": spec.key,
                "field_name": spec.field_name,
                "source_uri": str(source_tif),
                "resampling": spec.resampling,
                "asc": manifest_asc,
                "tif": str(out_tif) if export_tif else None,
                "status": "created",
            }
        )

    if created_template:
        template_tif_out = output_dir / "topographic__elevation.tif"
        if template_tif_out.exists():
            keep_paths.add(template_tif_out)
        template_asc_out = output_dir / "topographic__elevation.asc"
        if export_asc and template_asc_out.exists():
            keep_paths.add(template_asc_out)

    if not keep_intermediates:
        _cleanup_intermediates(output_dir, keep_paths)

    with rasterio.open(template_path) as src:
        grid = {
            "width": int(src.width),
            "height": int(src.height),
            "transform": [float(v) for v in tuple(src.transform)],
        }
        resolution = [float(src.res[0]), float(src.res[1])]
        target_crs = src.crs.to_string() if src.crs else target_crs

    manifest = _manifest_common(
        stage="harmonize",
        aoi=aoi_path,
        output_dir=output_dir,
        soil_keys=[s.key for s in specs],
        options={
            "output_format": output_format,
            "overwrite": overwrite,
            "keep_intermediates": keep_intermediates,
            "template_raster": str(template_path),
            "source_dir": str(source_dir) if source_dir else None,
        },
    )
    manifest.update(
        {
            "target_crs": target_crs,
            "target_resolution": resolution,
            "grid": grid,
            "layers": layers,
        }
    )
    return _write_json(output_dir / "soil_collection_manifest.json", manifest)
