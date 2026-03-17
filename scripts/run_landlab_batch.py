from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RunConfig:
    run_id: str
    dem_path: str
    total_t: float = 5_000_000.0
    dt: float = 1_000.0
    uplift_rate: float = 0.0003
    k_d: float = 0.01
    K_sp: float = 0.00001
    m_sp: float = 0.5
    n_sp: float = 1.0
    threshold_sp: float = 0.0
    flow_director: str = "FlowDirectorD8"
    roughness_amplitude: float = 0.0
    seed: int | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent Landlab diffusion+fluvial simulations in parallel "
            "across CPU cores."
        )
    )
    parser.add_argument(
        "--dem-path",
        default="data/topographic__elevation.asc",
        help="Path to DEM ESRI ASCII file.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/landlab_batch",
        help="Output directory for run metrics.",
    )
    parser.add_argument(
        "--run-specs",
        default=None,
        help=(
            "Optional JSON file with a list of per-run config overrides. "
            "If provided, --n-runs is ignored."
        ),
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=8,
        help="Number of independent runs to launch when --run-specs is not set.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Worker processes. Default: min(n_runs, os.cpu_count()).",
    )

    # Defaults matched to the notebook.
    parser.add_argument("--total-t", type=float, default=5_000_000.0)
    parser.add_argument("--dt", type=float, default=1_000.0)
    parser.add_argument("--uplift-rate", type=float, default=0.0003)
    parser.add_argument("--k-d", type=float, default=0.01)
    parser.add_argument("--K-sp", type=float, default=0.00001)
    parser.add_argument("--m-sp", type=float, default=0.5)
    parser.add_argument("--n-sp", type=float, default=1.0)
    parser.add_argument("--threshold-sp", type=float, default=0.0)
    parser.add_argument("--flow-director", default="FlowDirectorD8")
    parser.add_argument(
        "--roughness-amplitude",
        type=float,
        default=0.0,
        help="Optional random perturbation amplitude [m] added to DEM core nodes per run.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed for reproducible per-run random perturbations.",
    )

    return parser.parse_args()


def _load_run_specs(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("--run-specs JSON must contain a list of run config objects.")
    return data


def _build_run_configs(args: argparse.Namespace) -> list[RunConfig]:
    dem_path = str(Path(args.dem_path).resolve())

    if args.run_specs:
        specs = _load_run_specs(Path(args.run_specs))
        runs: list[RunConfig] = []
        for idx, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise ValueError(f"Run spec at index {idx} is not a JSON object.")
            merged: dict[str, Any] = {
                "run_id": spec.get("run_id", f"run_{idx:03d}"),
                "dem_path": dem_path,
                "total_t": args.total_t,
                "dt": args.dt,
                "uplift_rate": args.uplift_rate,
                "k_d": args.k_d,
                "K_sp": args.K_sp,
                "m_sp": args.m_sp,
                "n_sp": args.n_sp,
                "threshold_sp": args.threshold_sp,
                "flow_director": args.flow_director,
                "roughness_amplitude": args.roughness_amplitude,
                "seed": args.seed_base + idx,
            }
            merged.update(spec)
            merged["dem_path"] = dem_path
            runs.append(RunConfig(**merged))
        return runs

    if args.n_runs < 1:
        raise ValueError("--n-runs must be >= 1")

    return [
        RunConfig(
            run_id=f"run_{i:03d}",
            dem_path=dem_path,
            total_t=args.total_t,
            dt=args.dt,
            uplift_rate=args.uplift_rate,
            k_d=args.k_d,
            K_sp=args.K_sp,
            m_sp=args.m_sp,
            n_sp=args.n_sp,
            threshold_sp=args.threshold_sp,
            flow_director=args.flow_director,
            roughness_amplitude=args.roughness_amplitude,
            seed=args.seed_base + i,
        )
        for i in range(args.n_runs)
    ]


def _run_one(cfg: RunConfig) -> dict[str, Any]:
    # Avoid thread oversubscription when using process-level parallelism.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    from landlab.components import FastscapeEroder, FlowAccumulator, LinearDiffuser
    from landlab.io import esri_ascii

    start = time.perf_counter()

    with open(cfg.dem_path, "r") as f:
        mg = esri_ascii.load(f, name="topographic__elevation")

    mg.set_closed_boundaries_at_grid_edges(
        right_is_closed=True,
        top_is_closed=False,
        left_is_closed=True,
        bottom_is_closed=False,
    )

    z = mg.at_node["topographic__elevation"]

    if cfg.roughness_amplitude > 0.0:
        rng = np.random.default_rng(cfg.seed)
        z[mg.core_nodes] += rng.random(mg.core_nodes.size) * cfg.roughness_amplitude

    nt = int(cfg.total_t // cfg.dt)
    uplift_per_step = cfg.uplift_rate * cfg.dt

    lin_diffuse = LinearDiffuser(mg, linear_diffusivity=cfg.k_d, deposit=True)
    fr = FlowAccumulator(mg, flow_director=cfg.flow_director)
    sp = FastscapeEroder(
        mg,
        K_sp=cfg.K_sp,
        m_sp=cfg.m_sp,
        n_sp=cfg.n_sp,
        threshold_sp=cfg.threshold_sp,
    )

    core = mg.core_nodes
    mean_elev_history: list[float] = []

    for _ in range(nt):
        lin_diffuse.run_one_step(cfg.dt)
        fr.run_one_step()
        sp.run_one_step(cfg.dt)

        z[core] += uplift_per_step
        mean_elev_history.append(float(np.mean(z[core])))

    runtime_sec = time.perf_counter() - start

    out: dict[str, Any] = asdict(cfg)
    out.update(
        {
            "nt": nt,
            "uplift_per_step": uplift_per_step,
            "runtime_sec": runtime_sec,
            "runtime_min": runtime_sec / 60.0,
            "mean_elev_final": mean_elev_history[-1] if mean_elev_history else float(np.mean(z[core])),
            "mean_elev_min": float(np.min(mean_elev_history)) if mean_elev_history else float(np.mean(z[core])),
            "mean_elev_max": float(np.max(mean_elev_history)) if mean_elev_history else float(np.mean(z[core])),
            "elev_min": float(np.min(z[core])),
            "elev_max": float(np.max(z[core])),
            "n_nodes": int(mg.number_of_nodes),
            "n_core_nodes": int(core.size),
            "dx": float(mg.dx),
        }
    )
    return out


def _save_results(results: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-run JSON for easy reruns/debugging.
    for result in results:
        run_id = result["run_id"]
        with (out_dir / f"{run_id}.json").open("w") as f:
            json.dump(result, f, indent=2)

    # Flat CSV for quick comparison.
    keys: list[str] = sorted({k for r in results for k in r.keys()})
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = _parse_args()

    try:
        import landlab  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "landlab is not installed in the active Python environment. "
            "Activate your project conda env (environment.yml) or install landlab first."
        ) from exc

    dem_path = Path(args.dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    runs = _build_run_configs(args)
    workers = args.max_workers or min(len(runs), (os.cpu_count() or 1))
    out_dir = Path(args.out_dir)

    print(f"DEM: {dem_path.resolve()}")
    print(f"Runs: {len(runs)}")
    print(f"Workers: {workers}")
    if args.roughness_amplitude == 0.0 and len(runs) > 1 and args.run_specs is None:
        print(
            "Note: runs are identical because roughness_amplitude=0 and no run-specific overrides were provided."
        )

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        future_to_id = {pool.submit(_run_one, cfg): cfg.run_id for cfg in runs}
        for idx, future in enumerate(as_completed(future_to_id), start=1):
            run_id = future_to_id[future]
            result = future.result()
            results.append(result)
            print(
                f"[{idx}/{len(runs)}] {run_id} done in {result['runtime_min']:.2f} min "
                f"(final mean elev={result['mean_elev_final']:.3f} m)"
            )

    wall_sec = time.perf_counter() - t0
    results.sort(key=lambda r: str(r["run_id"]))
    _save_results(results, out_dir)

    print(f"Saved: {out_dir / 'summary.csv'}")
    print(f"Total wall time: {wall_sec / 60.0:.2f} min")


if __name__ == "__main__":
    main()
