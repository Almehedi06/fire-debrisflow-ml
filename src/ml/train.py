from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def train_random_forest_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)
    return model


def train_xgboost_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 1200,
    max_depth: int = 6,
    learning_rate: float = 0.03,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    min_child_weight: float = 5.0,
    gamma: float = 0.1,
    reg_alpha: float = 0.0,
    reg_lambda: float = 2.0,
    tree_method: str = "hist",
    max_bin: int = 256,
    verbosity: int = 0,
    random_state: int = 42,
    n_jobs: int = -1,
):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Run: pip install xgboost"
        ) from exc

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method=tree_method,
        max_bin=max_bin,
        verbosity=verbosity,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)
    return model
