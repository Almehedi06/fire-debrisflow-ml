from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import train_test_split

from ml.dataset import build_xy_from_rasters


def load_feature_order(path: str | Path) -> dict:
    import json

    with open(path, "r") as f:
        return json.load(f)


def resolve_feature_paths(data_dir: str | Path, feature_files: list[str]) -> list[Path]:
    root = Path(data_dir)
    paths = [root / name for name in feature_files]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature rasters: {missing}")
    return paths


def default_interpret_out_dir(model_path: str | Path) -> Path:
    run_id = Path(model_path).resolve().parent.name
    return Path("experiments") / "interpret" / run_id


def build_test_split_for_interpretation(
    feature_paths: list[str | Path],
    target_path: str | Path,
    split_test_size: float = 0.2,
    split_random_state: int = 42,
    max_samples: int | None = 50000,
    sample_random_state: int = 42,
) -> dict:
    bundle = build_xy_from_rasters(feature_paths, target_path)
    x = bundle["X"]
    y = bundle["y"]

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=split_test_size,
        random_state=split_random_state,
    )

    if max_samples is not None and x_test.shape[0] > max_samples:
        rng = np.random.default_rng(sample_random_state)
        idx = rng.choice(x_test.shape[0], size=max_samples, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    return {
        "X_test": x_test,
        "y_test": y_test,
        "feature_names": bundle["feature_names"],
        "feature_files": bundle["feature_files"],
    }


def compute_permutation_importance_rows(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
    scoring: str = "r2",
) -> list[dict]:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring=scoring,
    )

    rows = []
    for i, name in enumerate(feature_names):
        rows.append(
            {
                "feature": name,
                "importance_mean": float(result.importances_mean[i]),
                "importance_std": float(result.importances_std[i]),
            }
        )
    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    return rows


def _parse_partial_dependence_result(pd_result) -> tuple[np.ndarray, np.ndarray]:
    if "grid_values" in pd_result:
        grid = np.asarray(pd_result["grid_values"][0], dtype="float64")
    elif "values" in pd_result:
        grid = np.asarray(pd_result["values"][0], dtype="float64")
    else:
        raise KeyError("partial_dependence result missing grid values.")

    avg = np.asarray(pd_result["average"][0], dtype="float64").reshape(-1)
    return grid, avg


def compute_partial_dependence_curves(
    model,
    x_eval: np.ndarray,
    feature_names: list[str],
    selected_features: list[str],
    grid_resolution: int = 50,
) -> dict[str, dict]:
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    curves: dict[str, dict] = {}
    for name in selected_features:
        if name not in name_to_idx:
            raise ValueError(f"Feature not found for PDP: {name}")
        idx = name_to_idx[name]
        pd_res = partial_dependence(
            model,
            x_eval,
            features=[idx],
            grid_resolution=grid_resolution,
            kind="average",
        )
        x_grid, y_avg = _parse_partial_dependence_result(pd_res)
        curves[name] = {
            "x": x_grid,
            "y": y_avg,
        }
    return curves


def save_permutation_csv(rows: list[dict], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["feature", "importance_mean", "importance_std"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out


def save_pdp_csv(feature_name: str, x: np.ndarray, y: np.ndarray, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([feature_name, "prediction"])
        for xv, yv in zip(x, y):
            writer.writerow([float(xv), float(yv)])
    return out


def plot_permutation_importance(rows: list[dict], out_path: str | Path, top_k: int = 20) -> Path:
    import matplotlib.pyplot as plt

    top = rows[:top_k]
    features = [r["feature"] for r in top][::-1]
    means = [r["importance_mean"] for r in top][::-1]
    stds = [r["importance_std"] for r in top][::-1]

    fig_h = max(4, 0.35 * len(features) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(features, means, xerr=stds, color="#3b82f6", alpha=0.9)
    ax.set_xlabel("Permutation Importance")
    ax.set_title("Permutation Feature Importance")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_pdp_curve(feature_name: str, x: np.ndarray, y: np.ndarray, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, color="#111827", linewidth=2)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Predicted dem_diff")
    ax.set_title(f"Partial Dependence: {feature_name}")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

