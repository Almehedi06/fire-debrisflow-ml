# Soil Property Data Collection (Public Sources)

This guide shows how to collect the soil property rasters used in this repo from public URLs, clip them to an AOI, align them to the DEM grid, and export Landlab-ready ASCII files.

The current public soil rasters come from the SOLUS public bucket:
- https://storage.googleapis.com/solus100pub/

## Soil properties and public URLs

These match `config/base.yaml` and the mapping in `src/pipeline.py`.

| Source key | Model field name | Scale applied | Public URL |
| --- | --- | ---: | --- |
| `cec7_0_cm` | `cation__exchange_capacity` | `0.1` | https://storage.googleapis.com/solus100pub/cec7_0_cm_p.tif |
| `anylithicdpt_cm` | `soil__thickness` | `0.01` | https://storage.googleapis.com/solus100pub/anylithicdpt_cm_p.tif |
| `claytotal_0_cm` | `clay__total` | `1.0` | https://storage.googleapis.com/solus100pub/claytotal_0_cm_p.tif |
| `ph1to1h2o_0_cm` | `pH` | `0.01` | https://storage.googleapis.com/solus100pub/ph1to1h2o_0_cm_p.tif |
| `sandtotal_0_cm` | `sand__total` | `1.0` | https://storage.googleapis.com/solus100pub/sandtotal_0_cm_p.tif |
| `silttotal_0_cm` | `silt__total` | `1.0` | https://storage.googleapis.com/solus100pub/silttotal_0_cm_p.tif |
| `dbovendry_0_cm` | `dry__bulk_density` | `0.01` | https://storage.googleapis.com/solus100pub/dbovendry_0_cm_p.tif |

Notes:
- The scale factors are applied inside `run_landlab_pipeline` in `src/pipeline.py:314`.
- These base properties are then used to derive additional soil variables such as:
  - `soil__saturated_hydraulic_conductivity`
  - `soil__transmissivity`
  - `saturated__water_content`
  - `soil__texture`
  - `field__capacity`, `wilting__point`, `porosity`
  - `soil__internal_friction_angle`, `soil__density`

## Recommended workflow (uses the existing pipeline)

The simplest and most consistent way is to run the repo pipeline, which already:
- Ensures the AOI is in UTM (meters),
- Downloads the public rasters,
- Reprojects, resamples, and clips them to the DEM grid,
- Exports ASCII files for Landlab.

### 1) Configure the AOI and output directory

Edit these fields in `config/base.yaml`:

```yaml
aoi:
  aoi: /path/to/your_aoi.shp

paths:
  output_dir: /path/to/output_dir

raster:
  target_res: 10
```

Important:
- Use a Shapefile AOI (`.shp`). The current `ensure_utm_aoi` function overwrites the AOI on disk.
- If your AOI is a GeoPackage, make a Shapefile copy first.

### 2) Run the pipeline

From a notebook or Python session:

```python
from src.pipeline import run_pipeline

outputs, grid = run_pipeline("config/base.yaml", cleanup_intermediates=True)
```

This produces:
- Base soil ASCII files in `output_dir` (for example `clay__total.asc`),
- Derived soil ASCII files (for example `soil__saturated_hydraulic_conductivity.asc`).

## Quick verification checklist

After running, confirm:
- The ASCII files exist in `output_dir`,
- They all share the same grid shape and cellsize as the DEM,
- Values look reasonable when plotted.

Example quick check:

```python
import os
import rasterio

out_dir = "/path/to/output_dir"
check_files = [
    "topographic__elevation.asc",
    "clay__total.asc",
    "sand__total.asc",
    "silt__total.asc",
    "soil__saturated_hydraulic_conductivity.asc",
]

for name in check_files:
    path = os.path.join(out_dir, name)
    with rasterio.open(path) as src:
        print(name, src.crs, src.res, src.width, src.height)
```

---

If you want this guide adapted for the GAIA JupyterBook, this file is ready to copy with small edits.
