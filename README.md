# fire-debrisflow-ml

Modular pipeline for postfire raster processing (DEM, soils, burn severity, landcover) and ML training for `dem_diff` prediction.

## What This Repo Can Do

- Build an aligned raster stack for an AOI from DEM, soil, burn severity, and landcover inputs.
- Use a remote DEM (`bmi-topography`) or a local DEM raster as the elevation source.
- Fetch USDA SOLUS soil rasters, clip them to AOI, and harmonize them to a common grid.
- Generate Landlab-ready ASCII layers and exported GeoTIFF layers from the same processed stack.
- Create a `dem_diff.tif` target raster from aligned pre/post DEMs.
- Train raster ML regressors for `dem_diff` with Random Forest and XGBoost.
- Train deep raster models for `dem_diff` with U-Net and a simple CNN.
- Run full-raster prediction from trained ML/deep models.
- Run model-interpretation utilities for permutation feature importance and partial dependence.
- Run local smoke tests and CI checks for the soil workflow and raster processing path.

## Install

`environment.yml` is the canonical project environment for this repo.
Use it for full repo setup. `environment.ci.yml` is a lean CI environment, not the recommended user environment.

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate fire-debrisflow-ml
```

Use existing `ml_debris` environment (already present on this machine):

```bash
conda activate ml_debris
```

Known-good snapshot in `ml_debris`:
- `python 3.10.12`
- `numpy 2.2.6`
- `rasterio 1.4.4`
- `fiona 1.10.1`
- `geopandas 1.1.1`
- `shapely 2.1.2`
- `pyproj 3.7.1`
- `landlab 2.10.0`
- `bmi-topography 0.9.0`
- `scikit-learn 1.7.2`
- `xgboost 3.1.3`
- `torch 2.9.1+cpu`

If you want to run repository tests/smoke checks from `ml_debris`, install:

```bash
conda install -n ml_debris -c conda-forge pytest
```

Quick dependency check:

```bash
CONDA_NO_PLUGINS=true conda run -n ml_debris python -c "import importlib.util as u;mods=['numpy','yaml','requests','rasterio','fiona','geopandas','shapely','pyproj','landlab','bmi_topography','sklearn','joblib','xgboost','torch'];print({m: bool(u.find_spec(m)) for m in mods})"
```

Pip (if your system geospatial stack is already available):

```bash
pip install -r requirements.txt
```

Editable pip install from the repo root:

```bash
pip install -e ".[soil,ml,deep,viz,dev]"
```

This repo's base `pyproject.toml` dependency set is intentionally minimal. For a full local install, use `environment.yml`, `requirements.txt`, or the pip extras command above.

## Getting Started

Recommended setup for a new machine:

```bash
git clone <your-repo-url>
cd fire-debrisflow-ml
conda env create -f environment.yml
conda activate fire-debrisflow-ml
python -m pytest -q tests
cp config/base.example.yaml config/base.yaml
```

Then edit `config/base.yaml` for your machine-specific paths and data sources.

## Quick Start

Use an environment with required geospatial packages (`geopandas`, `rasterio`, `fiona`, `landlab`, `bmi_topography`, `requests`, `pyyaml`).

Start from `config/base.example.yaml`, make a local working copy, update the paths, then run:

```bash
python src/run_pipeline.py --config config/base.yaml --export-final-tifs
```

This builds aligned final `.asc` and `.tif` layers in `paths.output_dir`.

For dedicated USDA SOLUS soil data workflow (fetch/harmonize/run CLIs), see `usda_solus.md`.

Quick local verification after environment setup:

```bash
python -m pytest -q tests
```

## Which Workflow To Use

Use the main pipeline when you want the repo's end-to-end feature engineering and modeling workflow:

- build aligned DEM, soil, burn severity, and landcover layers,
- export Landlab-ready `.asc` and `.tif` outputs,
- create modeling-ready rasters for `dem_diff`,
- train or run ML/deep models on the prepared stack.

Use the dedicated soil CLI when you only need soil collection/harmonization as a standalone task:

- fetch USDA SOLUS soil rasters,
- clip them to an AOI,
- harmonize them to a chosen template grid,
- hand the outputs to another workflow or another user.

Relationship between them:

- they are related, because both deal with raster preprocessing,
- they are not the same thing,
- the soil CLI is a standalone sub-workflow,
- the main pipeline is the broader end-to-end workflow for feature generation and modeling.

## Burn Severity Source

`config/base.yaml` supports:

- `source: local`
- `source: remote`
- `source: remote_then_local` (try fire-name/id remote first, then local)
- `source: auto` (same behavior as `remote_then_local`)

## DEM Source

`config/base.yaml` supports:

- `source: bmi-topography`
- `source: local` with `dem.path` pointing to a local raster

## DEM Difference Target

Generate target raster (`dem_diff.tif`) aligned to the pipeline grid:

```bash
python src/dem_difference.py \
  --pre /path/to/pre_dem.tif \
  --post /path/to/post_dem.tif \
  --out-dir /path/to/output \
  --config config/base.yaml \
  --template /path/to/output/topographic__elevation.tif
```

## Train Models

Model configs in `config/ml_*.yaml` should declare the exact training rasters under `data.include_names`.
That explicit list is the authoritative feature set for the run. Training saves the same ordered list in `feature_order.json`, and prediction reuses it.

For `RF` and `XGB`, `split.method: spatial_block_cv` enables a held-out spatial test set plus spatial block cross-validation on the remaining blocks.
Key fields are `split.block_size`, `split.n_folds`, `split.test_size`, and `split.selection_metric`.
If you want to tune tree-model hyperparameters, add `model.search_grid` in the config. Without that block, the script evaluates the single configured parameter set.

Random Forest:

```bash
python scripts/train_model.py --config config/ml_rf.yaml
```

XGBoost:

```bash
python scripts/train_model.py --config config/ml_xgb.yaml
```

U-Net (CPU starter setup):

```bash
python scripts/train_unet.py --config config/ml_unet.yaml
```

U-Net tuning:

```bash
python scripts/tune_unet.py --config config/ml_unet.yaml
```

`train_unet.py` trains one configured run. `tune_unet.py` expands `tuning.search_space`, ranks candidates by validation metric, writes per-candidate histories/metrics, and saves the best model under `models/unet_tuning/<run_id>/best_model/`.
Deep configs support `runtime.device: auto|cpu|cuda`, and the train/predict scripts also accept `--device` for an explicit override.

Simple CNN (CPU starter setup):

```bash
python scripts/train_cnn.py --config config/ml_cnn.yaml
```

## Predict

XGBoost (latest run):

```bash
python scripts/predict_model.py \
  --model-path "$(ls -1dt models/xgb/* | head -n 1)/model.joblib" \
  --feature-order "$(ls -1dt models/xgb/* | head -n 1)/feature_order.json" \
  --data-dir /path/to/output \
  --out-path /path/to/output/dem_diff_pred_xgb.tif
```

U-Net (latest run):

```bash
python scripts/predict_unet.py \
  --model-path "$(ls -1dt models/unet/* | head -n 1)/model.pt" \
  --feature-order "$(ls -1dt models/unet/* | head -n 1)/feature_order.json" \
  --data-dir /path/to/output \
  --out-path /path/to/output/dem_diff_pred_unet.tif
```

Simple CNN (latest run):

```bash
python scripts/predict_cnn.py \
  --model-path "$(ls -1dt models/cnn/* | head -n 1)/model.pt" \
  --feature-order "$(ls -1dt models/cnn/* | head -n 1)/feature_order.json" \
  --data-dir /path/to/output \
  --out-path /path/to/output/dem_diff_pred_cnn.tif
```

## Interpretation

Permutation feature importance:

```bash
python scripts/feature_importance.py --config config/interpret.yaml
```

Partial dependence:

```bash
python scripts/partial_dependence.py --config config/interpret.yaml
```

## Testing

Run the full local test suite:

```bash
python -m pytest -q tests
```

Run the soil workflow smoke test directly:

```bash
python scripts/smoke_test_soil_cli.py
```

## Notes

- Training requires feature rasters and `dem_diff.tif` on the same grid (CRS, transform, shape).
- Pipeline needs internet access for remote DEM/feature downloads unless all sources are local/cached.
- Keep secrets (API keys, personal paths) out of commits.
