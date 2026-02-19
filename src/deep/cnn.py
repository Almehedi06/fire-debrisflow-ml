from __future__ import annotations

import torch
import torch.nn as nn


def _group_count(channels: int) -> int:
    if channels % 8 == 0:
        return 8
    if channels % 4 == 0:
        return 4
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNNRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 5,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        layers: list[nn.Module] = [ConvBlock(in_channels, hidden_channels)]
        for _ in range(num_layers - 2):
            layers.append(ConvBlock(hidden_channels, hidden_channels))
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
