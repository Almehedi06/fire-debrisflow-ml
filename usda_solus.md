# USDA SOLUS Soil Data Workflow

This document is the dedicated guide for SOLUS soil retrieval and homogenization.
Use this when you need pixel-aligned soil layers for downstream Landlab/ML/statistical tasks.

## Install

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate fire-debrisflow-ml
```

Or use existing `ml_debris` environment:

```bash
conda activate ml_debris
```

`ml_debris` currently includes required soil-workflow dependencies:
- `python 3.10.12`
- `numpy 2.2.6`
- `pyyaml 6.0.3`
- `requests 2.32.5`
- `rasterio 1.4.4`
- `fiona 1.10.1`
- `geopandas 1.1.1`
- `shapely 2.1.2`
- `pyproj 3.7.1`
- `landlab 2.10.0`
- `bmi-topography 0.9.0`

Optional for local tests:

```bash
conda install -n ml_debris -c conda-forge pytest
```

Quick dependency check in `ml_debris`:

```bash
CONDA_NO_PLUGINS=true conda run -n ml_debris python -c "import importlib.util as u;mods=['yaml','requests','rasterio','fiona','geopandas','shapely','pyproj','landlab','bmi_topography'];print({m: bool(u.find_spec(m)) for m in mods})"
```

Pip (if your system already has geospatial prerequisites):

```bash
pip install -r requirements.txt
```

## Workflow Design

There are two explicit stages plus one convenience wrapper:

1. `soil-fetch`: collect source rasters and clip to AOI (minimal transformation).
2. `soil-harmonize`: align all selected layers to one target grid.
3. `soil-run`: execute both stages in sequence.

CLIs are available as scripts and package entry points:
- Script wrappers: `scripts/soil_fetch.py`, `scripts/soil_harmonize.py`, `scripts/soil_run.py`
- Backward-compatible alias: `scripts/download_soil_data.py` -> `soil-run`

## Soil Variables (Current Set)

| Source key | Output field name | Public URL |
| --- | --- | --- |
| `cec7_0_cm` | `cation__exchange_capacity` | https://storage.googleapis.com/solus100pub/cec7_0_cm_p.tif |
| `anylithicdpt_cm` | `soil__thickness` | https://storage.googleapis.com/solus100pub/anylithicdpt_cm_p.tif |
| `claytotal_0_cm` | `clay__total` | https://storage.googleapis.com/solus100pub/claytotal_0_cm_p.tif |
| `ph1to1h2o_0_cm` | `pH` | https://storage.googleapis.com/solus100pub/ph1to1h2o_0_cm_p.tif |
| `sandtotal_0_cm` | `sand__total` | https://storage.googleapis.com/solus100pub/sandtotal_0_cm_p.tif |
| `silttotal_0_cm` | `silt__total` | https://storage.googleapis.com/solus100pub/silttotal_0_cm_p.tif |
| `dbovendry_0_cm` | `dry__bulk_density` | https://storage.googleapis.com/solus100pub/dbovendry_0_cm_p.tif |

## Stage 1: Fetch

Download/resolve rasters and clip to AOI (no common-grid harmonization):

```bash
python scripts/soil_fetch.py \
  --aoi /path/to/your_aoi.shp \
  --output-dir /path/to/soil_raw
```

Output manifest: `soil_fetch_manifest.json`

## Stage 2: Harmonize

Snap selected soil layers to one homogeneous grid (same CRS/transform/resolution/shape):

```bash
python scripts/soil_harmonize.py \
  --aoi /path/to/your_aoi.shp \
  --source-dir /path/to/soil_raw \
  --template /path/to/topographic__elevation.tif \
  --output-dir /path/to/soil_harmonized \
  --format both
```

If `--template` is omitted, DEM is fetched and used as grid anchor.

Output manifest: `soil_collection_manifest.json`

## One-Step Run (Fetch + Harmonize)

```bash
python scripts/soil_run.py \
  --aoi /path/to/your_aoi.shp \
  --output-dir /path/to/soil_harmonized \
  --format both
```

`scripts/download_soil_data.py` provides the same behavior for compatibility.

## Useful Options

- `--config config/base.yaml` to reuse AOI/output/source defaults.
- `--soil-keys cec7_0_cm,claytotal_0_cm,sandtotal_0_cm` for subsets.
- `--overwrite` to regenerate outputs.
- `--keep-intermediates` for debugging intermediate rasters.

## Data Contract

Harmonized outputs guarantee:
- shared CRS,
- shared transform,
- shared raster dimensions,
- deterministic field naming (`<field_name>.asc`, `<field_name>.tif`),
- provenance manifests with `manifest_version`, stage, layer metadata, and grid metadata.

## Smoke Test (Local/CI)

```bash
python scripts/smoke_test_soil_cli.py
```

This creates synthetic AOI/template/source rasters, runs `soil-run`, and validates:
- manifest creation,
- expected ASC/TIF outputs,
- basic end-to-end CLI behavior.

Optional unit test:

```bash
pytest -q tests/test_soil_data_core.py
```
