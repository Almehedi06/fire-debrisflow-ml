from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def discover_feature_paths(
    data_dir: str | Path,
    target_name: str = "dem_diff.tif",
    include_glob: str = "*.tif",
    include_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
    exclude_contains: list[str] | None = None,
) -> list[Path]:
    root = Path(data_dir)
    excluded = {target_name}
    if exclude_names:
        excluded.update(exclude_names)
    exclude_terms = [t.lower() for t in (exclude_contains or [])]

    if include_names:
        paths = []
        for name in include_names:
            p = root / name
            if not p.exists():
                raise FileNotFoundError(f"Requested feature not found: {p}")
            if p.name in excluded:
                continue
            lname = p.name.lower()
            if exclude_terms and any(term in lname for term in exclude_terms):
                continue
            paths.append(p)
        paths = sorted(paths)
        if not paths:
            raise ValueError("No feature rasters left after include/exclude filtering.")
        return paths

    paths = []
    for p in root.glob(include_glob):
        name = p.name
        if name in excluded:
            continue
        lname = name.lower()
        if exclude_terms and any(term in lname for term in exclude_terms):
            continue
        paths.append(p)
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No feature rasters found in {root} with pattern {include_glob}.")
    return paths


def assert_same_grid(reference_path: str | Path, candidate_paths: list[str | Path]) -> None:
    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)

    for p in candidate_paths:
        with rasterio.open(p) as src:
            if src.crs != ref_crs:
                raise ValueError(f"CRS mismatch for {p}: {src.crs} != {ref_crs}")
            if src.transform != ref_transform:
                raise ValueError(f"Transform mismatch for {p}")
            if (src.height, src.width) != ref_shape:
                raise ValueError(f"Shape mismatch for {p}: {(src.height, src.width)} != {ref_shape}")


def _read_band_and_invalid_mask(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata

    invalid = ~np.isfinite(arr)
    if nodata is not None:
        invalid |= arr == nodata
    # Handle common sentinels when nodata metadata is absent.
    invalid |= arr == -9999.0
    invalid |= np.abs(arr) > 1e20
    return arr, invalid


def build_xy_from_rasters(
    feature_paths: list[str | Path],
    target_path: str | Path,
) -> dict:
    if not feature_paths:
        raise ValueError("feature_paths is empty.")

    assert_same_grid(target_path, feature_paths)

    target_arr, target_invalid = _read_band_and_invalid_mask(target_path)

    feature_arrays: list[np.ndarray] = []
    invalid = target_invalid.copy()
    for fp in feature_paths:
        arr, arr_invalid = _read_band_and_invalid_mask(fp)
        feature_arrays.append(arr)
        invalid |= arr_invalid

    valid = ~invalid
    if not np.any(valid):
        raise ValueError("No valid pixels remain after nodata masking.")

    x = np.stack([arr[valid] for arr in feature_arrays], axis=1).astype("float32")
    y = target_arr[valid].astype("float32")

    with rasterio.open(target_path) as src:
        profile = src.profile.copy()

    return {
        "X": x,
        "y": y,
        "valid_mask": valid,
        "profile": profile,
        "shape": target_arr.shape,
        "feature_names": [Path(p).stem for p in feature_paths],
        "feature_files": [Path(p).name for p in feature_paths],
    }


def build_prediction_matrix(feature_paths: list[str | Path]) -> dict:
    if not feature_paths:
        raise ValueError("feature_paths is empty.")

    assert_same_grid(feature_paths[0], feature_paths[1:])

    feature_arrays: list[np.ndarray] = []
    invalid = None
    for fp in feature_paths:
        arr, arr_invalid = _read_band_and_invalid_mask(fp)
        feature_arrays.append(arr)
        if invalid is None:
            invalid = arr_invalid.copy()
        else:
            invalid |= arr_invalid

    if invalid is None:
        raise ValueError("No feature arrays were loaded.")

    valid = ~invalid
    if not np.any(valid):
        raise ValueError("No valid pixels remain after nodata masking.")

    x_valid = np.stack([arr[valid] for arr in feature_arrays], axis=1).astype("float32")

    with rasterio.open(feature_paths[0]) as src:
        profile = src.profile.copy()
        shape = (src.height, src.width)

    return {
        "X_valid": x_valid,
        "valid_mask": valid,
        "profile": profile,
        "shape": shape,
        "feature_names": [Path(p).stem for p in feature_paths],
        "feature_files": [Path(p).name for p in feature_paths],
    }
