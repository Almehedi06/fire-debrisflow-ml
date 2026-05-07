from __future__ import annotations

import copy
import itertools


def _set_nested(cfg: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def expand_search_space(cfg: dict) -> list[dict]:
    tuning_cfg = cfg.get("tuning", {})
    search_space = tuning_cfg.get("search_space") or {}
    if not search_space:
        return []

    keys = list(search_space.keys())
    value_lists: list[list] = []
    for key in keys:
        values = search_space[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"tuning.search_space[{key!r}] must be a non-empty list.")
        value_lists.append(values)

    candidates: list[dict] = []
    for combo in itertools.product(*value_lists):
        overrides = dict(zip(keys, combo))
        candidate_cfg = copy.deepcopy(cfg)
        for dotted_key, value in overrides.items():
            _set_nested(candidate_cfg, dotted_key, value)
        candidates.append({"overrides": overrides, "config": candidate_cfg})
    return candidates


def validate_unet_candidate_config(cfg: dict) -> None:
    split_cfg = cfg.get("split", {})
    train_cfg = cfg.get("training", {})

    patch_size = int(train_cfg.get("patch_size", 128))
    stride = int(train_cfg.get("stride", 64))
    block_size = int(split_cfg.get("block_size", 256))
    min_valid_fraction = float(train_cfg.get("min_valid_fraction", 1.0))

    if patch_size <= 0 or stride <= 0 or block_size <= 0:
        raise ValueError("patch_size, stride, and block_size must be positive.")
    if patch_size % 8 != 0:
        raise ValueError("U-Net patch_size must be divisible by 8.")
    if block_size < patch_size:
        raise ValueError("split.block_size must be >= training.patch_size.")
    if stride > patch_size:
        raise ValueError("training.stride must be <= training.patch_size.")
    if not (0.0 < min_valid_fraction <= 1.0):
        raise ValueError("training.min_valid_fraction must be in (0, 1].")


def select_best_candidate(results: list[dict], metric: str) -> dict:
    if metric not in {"rmse", "mae", "r2"}:
        raise ValueError(f"Unsupported tuning.selection_metric: {metric}")
    key = f"best_val_{metric}"
    reverse = metric == "r2"
    return sorted(results, key=lambda row: row[key], reverse=reverse)[0]
