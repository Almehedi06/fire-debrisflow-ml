from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_yaml(path: str | Path | None) -> dict:
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute permutation feature importance.")
    parser.add_argument("--config", default=None, help="Optional YAML config for interpretation.")
    parser.add_argument("--model-path", default=None, help="Path to model.joblib.")
    parser.add_argument("--feature-order", default=None, help="Path to feature_order.json.")
    parser.add_argument("--data-dir", default=None, help="Directory containing feature/target rasters.")
    parser.add_argument("--target", default=None, help="Target raster filename.")
    parser.add_argument("--out-dir", default=None, help="Output directory for CSV/plots.")
    parser.add_argument("--n-repeats", type=int, default=None, help="Permutation repeats.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples.")
    parser.add_argument("--scoring", default=None, help="Scoring metric (default: r2).")
    parser.add_argument("--random-state", type=int, default=None, help="Random seed.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for permutation.")
    parser.add_argument("--top-k", type=int, default=None, help="Top K features in bar plot.")
    return parser.parse_args()


def _cfg_get(cfg: dict, section: str, key: str, default):
    return cfg.get(section, {}).get(key, default)


def main() -> None:
    args = _parse_args()

    from ml.interpret import (
        build_test_split_for_interpretation,
        compute_permutation_importance_rows,
        default_interpret_out_dir,
        load_feature_order,
        plot_permutation_importance,
        resolve_feature_paths,
        save_permutation_csv,
    )
    from ml.io import load_model, save_json

    cfg = _load_yaml(args.config)

    model_path = args.model_path or _cfg_get(cfg, "model", "path", None)
    feature_order_path = args.feature_order or _cfg_get(cfg, "model", "feature_order", None)
    data_dir = args.data_dir or _cfg_get(cfg, "data", "dir", None)
    target_name = args.target or _cfg_get(cfg, "data", "target_name", "dem_diff.tif")

    if not model_path or not feature_order_path or not data_dir:
        raise ValueError("Provide model-path, feature-order, and data-dir (CLI or config).")

    n_repeats = int(args.n_repeats or _cfg_get(cfg, "permutation", "n_repeats", 10))
    max_samples = int(args.max_samples or _cfg_get(cfg, "permutation", "max_samples", 50000))
    scoring = args.scoring or _cfg_get(cfg, "permutation", "scoring", "r2")
    random_state = int(args.random_state or _cfg_get(cfg, "permutation", "random_state", 42))
    n_jobs = int(args.n_jobs or _cfg_get(cfg, "permutation", "n_jobs", -1))
    top_k = int(args.top_k or _cfg_get(cfg, "permutation", "top_k", 20))

    out_dir = Path(args.out_dir or _cfg_get(cfg, "output", "dir", "")) if args.out_dir or _cfg_get(cfg, "output", "dir", None) else default_interpret_out_dir(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_order = load_feature_order(feature_order_path)
    feature_files = feature_order.get("feature_files")
    if not feature_files:
        raise ValueError("feature_order.json missing 'feature_files'.")
    feature_paths = resolve_feature_paths(data_dir, feature_files)

    split_cfg = feature_order.get("split", {})
    split_test_size = float(split_cfg.get("test_size", 0.2))
    split_random_state = int(split_cfg.get("random_state", 42))
    target_path = Path(data_dir) / target_name
    if not target_path.exists():
        raise FileNotFoundError(f"Target raster not found: {target_path}")

    model = load_model(model_path)
    eval_bundle = build_test_split_for_interpretation(
        feature_paths=feature_paths,
        target_path=target_path,
        split_test_size=split_test_size,
        split_random_state=split_random_state,
        max_samples=max_samples,
        sample_random_state=random_state,
    )

    effective_n_jobs = n_jobs
    try:
        rows = compute_permutation_importance_rows(
            model=model,
            x_test=eval_bundle["X_test"],
            y_test=eval_bundle["y_test"],
            feature_names=eval_bundle["feature_names"],
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=effective_n_jobs,
            scoring=scoring,
        )
    except Exception as exc:
        msg = str(exc).lower()
        is_pickling = "pickle" in msg or "picklingerror" in msg or "could not pickle" in msg
        if not is_pickling or effective_n_jobs == 1:
            raise
        print(
            "Permutation importance parallel execution failed. "
            "Retrying with n_jobs=1 for compatibility."
        )
        effective_n_jobs = 1
        rows = compute_permutation_importance_rows(
            model=model,
            x_test=eval_bundle["X_test"],
            y_test=eval_bundle["y_test"],
            feature_names=eval_bundle["feature_names"],
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=effective_n_jobs,
            scoring=scoring,
        )

    csv_path = save_permutation_csv(rows, out_dir / "permutation_importance.csv")
    plot_path = plot_permutation_importance(rows, out_dir / "permutation_importance.png", top_k=top_k)
    save_json(
        {
            "model_path": str(model_path),
            "feature_order_path": str(feature_order_path),
            "data_dir": str(data_dir),
            "target_name": target_name,
            "n_repeats": n_repeats,
            "max_samples": max_samples,
            "scoring": scoring,
            "random_state": random_state,
            "n_jobs_requested": n_jobs,
            "n_jobs_used": effective_n_jobs,
            "split_test_size": split_test_size,
            "split_random_state": split_random_state,
            "n_features": len(rows),
        },
        out_dir / "permutation_meta.json",
    )

    print("Saved:", csv_path)
    print("Saved:", plot_path)


if __name__ == "__main__":
    main()
