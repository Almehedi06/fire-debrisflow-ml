from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def random_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return {
        "X_train": x_train,
        "X_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def build_spatial_block_groups(valid_mask: np.ndarray, block_size: int) -> np.ndarray:
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must be 2D, got shape {valid_mask.shape}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    n_rows, n_cols = valid_mask.shape
    n_block_cols = int(np.ceil(n_cols / block_size))
    rows, cols = np.indices((n_rows, n_cols))
    block_ids = (rows // block_size) * n_block_cols + (cols // block_size)
    groups = block_ids[valid_mask]
    if groups.size == 0:
        raise ValueError("No valid pixels available to assign spatial block groups.")
    return groups.astype("int64", copy=False)


def spatial_block_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    if x.shape[0] != y.shape[0] or x.shape[0] != groups.shape[0]:
        raise ValueError("x, y, and groups must have the same first dimension.")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Need at least two unique spatial groups for train/test split.")

    rng = np.random.default_rng(random_state)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    n_test_groups = int(np.ceil(unique_groups.size * test_size))
    n_test_groups = max(1, min(n_test_groups, unique_groups.size - 1))
    test_groups = shuffled_groups[:n_test_groups]
    train_groups = shuffled_groups[n_test_groups:]

    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Spatial train/test split produced an empty split.")

    return {
        "X_train": x[train_mask],
        "X_test": x[test_mask],
        "y_train": y[train_mask],
        "y_test": y[test_mask],
        "groups_train": groups[train_mask],
        "groups_test": groups[test_mask],
        "train_group_ids": train_groups.astype("int64").tolist(),
        "test_group_ids": test_groups.astype("int64").tolist(),
    }


def spatial_block_kfold_indices(
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[dict[str, np.ndarray]]:
    if groups.ndim != 1:
        raise ValueError(f"groups must be 1D, got shape {groups.shape}")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    unique_groups = np.unique(groups)
    if unique_groups.size < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique spatial groups, got {unique_groups.size}."
        )

    rng = np.random.default_rng(random_state)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    fold_groups = np.array_split(shuffled_groups, n_splits)

    folds: list[dict[str, np.ndarray]] = []
    for val_groups in fold_groups:
        val_mask = np.isin(groups, val_groups)
        train_mask = ~val_mask
        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError("Spatial block CV produced an empty train or validation fold.")
        folds.append(
            {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "val_group_ids": val_groups.astype("int64"),
            }
        )
    return folds
