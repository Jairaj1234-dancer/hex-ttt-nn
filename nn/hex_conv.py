"""Hex-aware convolution utilities for brick-wall hex grid layout.

On a brick-wall (even-r offset) layout, a standard 3x3 convolution kernel
captures 8 cells around the center.  Six of these correspond to true hex
neighbors; the remaining two are the diagonally-opposite "non-neighbor"
corners.  This is a well-known and accepted approximation -- the slight
contamination from two non-neighbor cells is negligible in practice and
avoids the complexity of custom sparse kernels.

The module provides:
    - :class:`HexResBlock`: standard pre-activation residual block.
    - :class:`HexResNet`: full ResNet backbone with KataGo-style global
      pooling so that the network can reason about whole-board features
      (e.g. material count, global threat level) alongside local patterns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HexResBlock(nn.Module):
    """Standard residual block for hex-grid feature maps.

    Architecture::

        x ─┬─ Conv3x3 ─ BN ─ ReLU ─ Conv3x3 ─ BN ─┬─ ReLU ─ out
           └────────────── skip ────────────────────┘

    Padding is ``same`` so spatial dimensions are preserved.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, C, H, W)`` feature map.

        Returns:
            ``(B, C, H, W)`` feature map with residual connection.
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class HexResNet(nn.Module):
    """ResNet backbone for hex grid with KataGo-style global pooling.

    Architecture::

        input ─ Conv3x3 ─ BN ─ ReLU ─ [HexResBlock] x N ─┬─ trunk output
                                                            │
                Global pooling branch:                      │
                  GAP + GMP ─ concat ─ FC ─ ReLU ─ FC ─ broadcast ─ ADD ─ output

    The global pooling branch lets the network incorporate whole-board
    statistics (e.g. stone counts, overall threat level) into every cell's
    representation.

    Args:
        in_channels: number of input feature planes (typically 12).
        num_blocks: number of residual blocks.
        channels: number of internal channels.
    """

    def __init__(self, in_channels: int, num_blocks: int, channels: int) -> None:
        super().__init__()
        self.channels = channels

        # Initial convolution: project input planes to internal channel count.
        self.init_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(channels)

        # Stack of residual blocks.
        self.blocks = nn.ModuleList([HexResBlock(channels) for _ in range(num_blocks)])

        # KataGo-style global pooling branch.
        # GAP + GMP -> (B, 2*channels) -> FC -> ReLU -> FC -> (B, channels)
        self.global_fc1 = nn.Linear(2 * channels, channels)
        self.global_fc2 = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, in_channels, H, W)`` input feature tensor.

        Returns:
            ``(B, channels, H, W)`` output feature tensor.
        """
        # Initial convolution.
        out = F.relu(self.init_bn(self.init_conv(x)))  # (B, C, H, W)

        # Residual tower.
        for block in self.blocks:
            out = block(out)  # (B, C, H, W)

        # Global pooling branch.
        gap = out.mean(dim=(2, 3))  # (B, C) -- global average pooling
        gmp = out.amax(dim=(2, 3))  # (B, C) -- global max pooling
        pooled = torch.cat([gap, gmp], dim=1)  # (B, 2*C)
        global_feat = F.relu(self.global_fc1(pooled))  # (B, C)
        global_feat = self.global_fc2(global_feat)  # (B, C)

        # Broadcast back to spatial dimensions and add.
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        out = out + global_feat  # (B, C, H, W) -- broadcast addition

        return out
