# fire-debrisflow-ml

Modular pipeline for postfire raster processing (DEM, soils, burn severity, landcover) and ML training for `dem_diff` prediction.

## Install

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

## Quick Start

Use an environment with required geospatial packages (`geopandas`, `rasterio`, `fiona`, `landlab`, `bmi_topography`, `requests`, `pyyaml`).

Update paths in `config/base.yaml`, then run:

```bash
python src/run_pipeline.py --config config/base.yaml --export-final-tifs
```

This builds aligned final `.asc` and `.tif` layers in `paths.output_dir`.

For dedicated USDA SOLUS soil data workflow (fetch/harmonize/run CLIs), see `usda_solus.md`.

## Burn Severity Source

`config/base.yaml` supports:

- `source: local`
- `source: remote`
- `source: remote_then_local` (try fire-name/id remote first, then local)
- `source: auto` (same behavior as `remote_then_local`)

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

## Notes

- Training requires feature rasters and `dem_diff.tif` on the same grid (CRS, transform, shape).
- Pipeline needs internet access for remote DEM/feature downloads unless all sources are local/cached.
- Keep secrets (API keys, personal paths) out of commits.
