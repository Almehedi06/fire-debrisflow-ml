from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from deep.cnn import SimpleCNNRegressor
from deep.data import (
    RasterPatchDataset,
    build_block_windows,
    extract_patch_origins,
    split_block_windows,
)
from deep.unet import UNetRegressor
from ml.evaluate import regression_metrics


@dataclass(frozen=True)
class TrainArtifacts:
    model: object
    history: list[dict]
    test_metrics: dict
    split_summary: dict
    norm_mean: np.ndarray
    norm_std: np.ndarray


def _set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_channel_norm_stats(
    x: np.ndarray,  # [C, H, W]
    mask: np.ndarray,  # [H, W]
) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(x.shape[0], dtype="float32")
    stds = np.ones(x.shape[0], dtype="float32")
    for c in range(x.shape[0]):
        vals = x[c][mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            means[c] = 0.0
            stds[c] = 1.0
            continue
        means[c] = float(vals.mean())
        s = float(vals.std())
        stds[c] = s if s > 1e-6 else 1.0
    return means, stds


def _apply_channel_norm(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    x64 = x.astype("float64", copy=False)
    m64 = means.astype("float64", copy=False)
    s64 = stds.astype("float64", copy=False)
    x_norm = (x64 - m64[:, None, None]) / s64[:, None, None]
    # Prevent rare extreme values from dominating optimization.
    x_norm = np.clip(x_norm, -20.0, 20.0)
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return x_norm.astype("float32")


def _masked_mse_loss(pred, target, mask):
    import torch

    diff2 = (pred - target) ** 2
    masked = diff2 * mask
    denom = mask.sum()
    if denom.item() <= 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return masked.sum() / denom


def _masked_huber_loss(pred, target, mask, delta: float):
    import torch

    abs_diff = torch.abs(pred - target)
    # Smooth L1 with configurable transition point.
    loss = torch.where(abs_diff < delta, 0.5 * (abs_diff**2) / delta, abs_diff - 0.5 * delta)
    masked = loss * mask
    denom = mask.sum()
    if denom.item() <= 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return masked.sum() / denom


def _evaluate_loader(model, loader, device: str) -> tuple[float, dict]:
    import torch

    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb, mb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            out = model(xb)
            loss = _masked_mse_loss(out, yb, mb)
            losses.append(float(loss.item()))

            out_np = out.detach().cpu().numpy().reshape(-1)
            y_np = yb.detach().cpu().numpy().reshape(-1)
            m_np = mb.detach().cpu().numpy().reshape(-1) > 0.5
            if np.any(m_np):
                preds.append(out_np[m_np])
                truths.append(y_np[m_np])

    if preds:
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(truths)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[finite]
        y_pred = y_pred[finite]
    else:
        y_true = np.array([], dtype="float32")
        y_pred = np.array([], dtype="float32")

    if y_true.size == 0:
        metrics = {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"), "n_samples": 0}
    else:
        metrics = regression_metrics(y_true, y_pred)
    return float(np.mean(losses)) if losses else float("nan"), metrics


def _prepare_patch_splits(
    y: np.ndarray,
    valid_mask: np.ndarray,
    split_cfg: dict,
    train_cfg: dict,
) -> dict:
    random_state = int(split_cfg.get("random_state", 42))
    patch_size = int(train_cfg.get("patch_size", 128))
    stride = int(train_cfg.get("stride", 64))
    block_size = int(split_cfg.get("block_size", 256))
    min_valid_fraction = float(train_cfg.get("min_valid_fraction", 1.0))

    windows = build_block_windows(y.shape[0], y.shape[1], block_size=block_size)
    split_windows = split_block_windows(
        windows,
        train_frac=float(split_cfg.get("train_size", 0.7)),
        val_frac=float(split_cfg.get("val_size", 0.15)),
        test_frac=float(split_cfg.get("test_size", 0.15)),
        random_state=random_state,
    )

    train_origins = extract_patch_origins(
        valid_mask,
        split_windows["train"],
        patch_size=patch_size,
        stride=stride,
        min_valid_fraction=min_valid_fraction,
    )
    val_origins = extract_patch_origins(
        valid_mask,
        split_windows["val"],
        patch_size=patch_size,
        stride=stride,
        min_valid_fraction=min_valid_fraction,
    )
    test_origins = extract_patch_origins(
        valid_mask,
        split_windows["test"],
        patch_size=patch_size,
        stride=stride,
        min_valid_fraction=min_valid_fraction,
    )

    if not train_origins:
        raise ValueError("No training patches were created. Lower min_valid_fraction or patch size.")
    if not val_origins:
        raise ValueError("No validation patches were created. Adjust block/patch settings.")
    if not test_origins:
        raise ValueError("No test patches were created. Adjust block/patch settings.")

    return {
        "patch_size": patch_size,
        "stride": stride,
        "block_size": block_size,
        "min_valid_fraction": min_valid_fraction,
        "split_windows": split_windows,
        "train_origins": train_origins,
        "val_origins": val_origins,
        "test_origins": test_origins,
    }


def _train_patch_regression(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    config: dict,
    model_builder: Callable[[int, dict], object],
) -> TrainArtifacts:
    import torch
    from torch.utils.data import DataLoader

    split_cfg = config.get("split", {})
    train_cfg = config.get("training", {})

    random_state = int(split_cfg.get("random_state", 42))
    _set_seed(random_state)

    split_setup = _prepare_patch_splits(y, valid_mask, split_cfg, train_cfg)
    patch_size = split_setup["patch_size"]
    stride = split_setup["stride"]
    block_size = split_setup["block_size"]
    min_valid_fraction = split_setup["min_valid_fraction"]
    split_windows = split_setup["split_windows"]
    train_origins = split_setup["train_origins"]
    val_origins = split_setup["val_origins"]
    test_origins = split_setup["test_origins"]

    train_region_mask = np.zeros_like(valid_mask, dtype=bool)
    for w in split_windows["train"]:
        train_region_mask[w.row0 : w.row1, w.col0 : w.col1] = True
    norm_mask = train_region_mask & valid_mask
    if not np.any(norm_mask):
        norm_mask = valid_mask

    norm_mean, norm_std = _compute_channel_norm_stats(x, norm_mask)
    x_norm = _apply_channel_norm(x, norm_mean, norm_std)

    train_ds = RasterPatchDataset(x_norm, y, valid_mask, train_origins, patch_size=patch_size)
    val_ds = RasterPatchDataset(x_norm, y, valid_mask, val_origins, patch_size=patch_size)
    test_ds = RasterPatchDataset(x_norm, y, valid_mask, test_origins, patch_size=patch_size)

    batch_size = int(train_cfg.get("batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = "cpu"
    model_cfg = config.get("model", {})
    model = model_builder(int(x.shape[0]), model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    loss_name = str(train_cfg.get("loss", "mse")).lower()
    huber_delta = float(train_cfg.get("huber_delta", 1.0))

    def _train_loss(out, yb, mb):
        if loss_name == "huber":
            return _masked_huber_loss(out, yb, mb, delta=huber_delta)
        return _masked_mse_loss(out, yb, mb)

    best_state = None
    best_val_rmse = float("inf")
    history: list[dict] = []

    epochs = int(train_cfg.get("epochs", 20))
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []

        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = _train_loss(out, yb, mb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss, val_metrics = _evaluate_loader(model, val_loader, device)

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_r2": val_metrics["r2"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
        }
        history.append(epoch_row)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics = _evaluate_loader(model, test_loader, device)
    test_metrics = {
        **test_metrics,
        "test_loss": test_loss,
    }

    split_summary = {
        "patch_size": patch_size,
        "stride": stride,
        "block_size": block_size,
        "min_valid_fraction": min_valid_fraction,
        "n_train_patches": len(train_ds),
        "n_val_patches": len(val_ds),
        "n_test_patches": len(test_ds),
        "n_train_blocks": len(split_windows["train"]),
        "n_val_blocks": len(split_windows["val"]),
        "n_test_blocks": len(split_windows["test"]),
    }

    return TrainArtifacts(
        model=model,
        history=history,
        test_metrics=test_metrics,
        split_summary=split_summary,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )


def train_unet_regression(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    config: dict,
) -> TrainArtifacts:
    def _builder(in_channels: int, model_cfg: dict) -> UNetRegressor:
        return UNetRegressor(
            in_channels=in_channels,
            base_channels=int(model_cfg.get("base_channels", 32)),
        )

    return _train_patch_regression(
        x=x,
        y=y,
        valid_mask=valid_mask,
        config=config,
        model_builder=_builder,
    )


def train_cnn_regression(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    config: dict,
) -> TrainArtifacts:
    def _builder(in_channels: int, model_cfg: dict) -> SimpleCNNRegressor:
        return SimpleCNNRegressor(
            in_channels=in_channels,
            hidden_channels=int(model_cfg.get("hidden_channels", 64)),
            num_layers=int(model_cfg.get("num_layers", 5)),
        )

    return _train_patch_regression(
        x=x,
        y=y,
        valid_mask=valid_mask,
        config=config,
        model_builder=_builder,
    )
