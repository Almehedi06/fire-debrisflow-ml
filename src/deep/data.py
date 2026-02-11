from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio

from ml.dataset import assert_same_grid

@dataclass(frozen=True)
class BlockWindow:
    row0: int
    row1: int
    col0: int
    col1: int


def _read_band(path: str | Path) -> tuple[np.ndarray, float | None]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
    return arr, nodata


def load_feature_target_stack(
    feature_paths: list[str | Path],
    target_path: str | Path,
) -> dict:
    if not feature_paths:
        raise ValueError("feature_paths is empty.")

    assert_same_grid(target_path, feature_paths)

    target_arr, target_nodata = _read_band(target_path)
    invalid = np.isnan(target_arr)
    if target_nodata is not None:
        invalid |= target_arr == target_nodata

    feature_arrays: list[np.ndarray] = []
    for fp in feature_paths:
        arr, nodata = _read_band(fp)
        feature_arrays.append(arr)
        invalid |= np.isnan(arr)
        if nodata is not None:
            invalid |= arr == nodata

    x = np.stack(feature_arrays, axis=0).astype("float32")  # [C, H, W]
    y = target_arr.astype("float32")  # [H, W]
    valid_mask = ~invalid

    with rasterio.open(target_path) as src:
        profile = src.profile.copy()

    return {
        "X": x,
        "y": y,
        "valid_mask": valid_mask,
        "shape": y.shape,
        "profile": profile,
        "feature_files": [Path(p).name for p in feature_paths],
        "feature_names": [Path(p).stem for p in feature_paths],
    }


def load_feature_stack(feature_paths: list[str | Path]) -> dict:
    if not feature_paths:
        raise ValueError("feature_paths is empty.")

    assert_same_grid(feature_paths[0], feature_paths[1:])

    feature_arrays: list[np.ndarray] = []
    invalid = None
    for fp in feature_paths:
        arr, nodata = _read_band(fp)
        feature_arrays.append(arr)
        arr_invalid = np.isnan(arr)
        if nodata is not None:
            arr_invalid |= arr == nodata
        invalid = arr_invalid.copy() if invalid is None else (invalid | arr_invalid)

    if invalid is None:
        raise ValueError("Failed to build valid mask from feature rasters.")

    with rasterio.open(feature_paths[0]) as src:
        profile = src.profile.copy()
        shape = (src.height, src.width)

    return {
        "X": np.stack(feature_arrays, axis=0).astype("float32"),
        "valid_mask": ~invalid,
        "shape": shape,
        "profile": profile,
        "feature_files": [Path(p).name for p in feature_paths],
        "feature_names": [Path(p).stem for p in feature_paths],
    }


def build_block_windows(height: int, width: int, block_size: int) -> list[BlockWindow]:
    windows: list[BlockWindow] = []
    for r0 in range(0, height, block_size):
        r1 = min(r0 + block_size, height)
        for c0 in range(0, width, block_size):
            c1 = min(c0 + block_size, width)
            windows.append(BlockWindow(r0, r1, c0, c1))
    return windows


def split_block_windows(
    windows: list[BlockWindow],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> dict[str, list[BlockWindow]]:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    rng = np.random.default_rng(random_state)
    idx = np.arange(len(windows))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val
    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError("Block split produced an empty split. Reduce block_size or adjust fractions.")

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return {
        "train": [windows[i] for i in train_idx],
        "val": [windows[i] for i in val_idx],
        "test": [windows[i] for i in test_idx],
    }


def _start_positions(start: int, stop: int, patch_size: int, stride: int) -> list[int]:
    if stop - start < patch_size:
        return []
    out = list(range(start, stop - patch_size + 1, stride))
    last = stop - patch_size
    if not out or out[-1] != last:
        out.append(last)
    return out


def extract_patch_origins(
    valid_mask: np.ndarray,
    windows: Iterable[BlockWindow],
    patch_size: int = 128,
    stride: int = 64,
    min_valid_fraction: float = 1.0,
) -> list[tuple[int, int]]:
    origins: list[tuple[int, int]] = []
    for win in windows:
        row_starts = _start_positions(win.row0, win.row1, patch_size, stride)
        col_starts = _start_positions(win.col0, win.col1, patch_size, stride)
        for r in row_starts:
            for c in col_starts:
                valid_patch = valid_mask[r : r + patch_size, c : c + patch_size]
                valid_fraction = float(valid_patch.mean())
                if valid_fraction >= min_valid_fraction:
                    origins.append((r, c))
    return origins


class RasterPatchDataset:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        origins: list[tuple[int, int]],
        patch_size: int,
    ) -> None:
        self.x = x
        self.y = y
        self.origins = origins
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int):
        import torch

        r, c = self.origins[index]
        ps = self.patch_size
        x_patch = self.x[:, r : r + ps, c : c + ps]
        y_patch = self.y[r : r + ps, c : c + ps]
        return (
            torch.from_numpy(x_patch).float(),
            torch.from_numpy(y_patch[None, :, :]).float(),
        )

