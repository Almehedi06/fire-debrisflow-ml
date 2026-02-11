# fire-debrisflow-ml

Modular pipeline for postfire raster processing (DEM, soils, burn severity, landcover) and ML training for `dem_diff` prediction.

## Quick Start

Update paths in `config/base.yaml`, then run:

```bash
python src/run_pipeline.py --config config/base.yaml --export-final-tifs
```

This builds aligned final `.asc` and `.tif` layers in `paths.output_dir`.

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

## Notes

- Training requires feature rasters and `dem_diff.tif` on the same grid (CRS, transform, shape).
- Keep secrets (API keys, personal paths) out of commits.
