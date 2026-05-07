from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.split import (
    build_spatial_block_groups,
    spatial_block_kfold_indices,
    spatial_block_train_test_split,
)


def test_build_spatial_block_groups_matches_valid_pixel_order() -> None:
    valid_mask = np.array(
        [
            [True, False, True, True],
            [True, True, False, False],
            [False, True, True, False],
            [True, False, True, True],
        ],
        dtype=bool,
    )

    groups = build_spatial_block_groups(valid_mask, block_size=2)

    # 2x2 blocks over a 4x4 raster produce block ids:
    # [[0,0,1,1],
    #  [0,0,1,1],
    #  [2,2,3,3],
    #  [2,2,3,3]]
    expected = np.array([0, 1, 1, 0, 0, 2, 3, 2, 3, 3], dtype=np.int64)
    assert np.array_equal(groups, expected)


def test_spatial_block_train_test_split_keeps_groups_disjoint() -> None:
    x = np.arange(24, dtype=float).reshape(12, 2)
    y = np.arange(12, dtype=float)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)

    split = spatial_block_train_test_split(x, y, groups, test_size=0.34, random_state=7)

    train_groups = set(split["train_group_ids"])
    test_groups = set(split["test_group_ids"])
    assert train_groups
    assert test_groups
    assert train_groups.isdisjoint(test_groups)
    assert split["X_train"].shape[0] + split["X_test"].shape[0] == x.shape[0]


def test_spatial_block_kfold_indices_cover_all_groups_without_overlap() -> None:
    groups = np.repeat(np.arange(6, dtype=np.int64), 3)
    folds = spatial_block_kfold_indices(groups, n_splits=3, random_state=42)

    seen_val_groups: set[int] = set()
    for fold in folds:
        train_idx = fold["train_idx"]
        val_idx = fold["val_idx"]
        assert train_idx.size > 0
        assert val_idx.size > 0

        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        assert train_groups.isdisjoint(val_groups)
        seen_val_groups.update(val_groups)

    assert seen_val_groups == set(np.unique(groups).tolist())
