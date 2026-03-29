"""Hex-aware convolution utilities for brick-wall hex grid layout.

On a brick-wall (even-r offset) layout, a standard 3x3 convolution kernel
captures 8 cells around the center.  Six of these correspond to true hex
neighbors; the remaining two are the diagonally-opposite "non-neighbor"
corners.

This module provides :class:`HexConv2d`, a Conv2d subclass that masks out
the two non-hex-neighbor corners of the 3x3 kernel (positions [0,0] and
[2,2] in the kernel tensor).  This ensures the network operates strictly
on the Z[omega] lattice neighborhood, encoding geometry (structural
substrate) without leaking information from non-adjacent cells.

Inspired by the hexgo project (github.com/sub-surface/hexgo) which frames
the hex grid as an Eisenstein integer ring and masks the kernel accordingly.

The module also provides:
    - :class:`HexResBlock`: residual block using ``HexConv2d``.
    - :class:`HexResNet`: full ResNet backbone with KataGo-style global
      pooling so that the network can reason about whole-board features
      (e.g. material count, global threat level) alongside local patterns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HexConv2d(nn.Conv2d):
    """Conv2d with a Z[omega] kernel mask for hex grids.

    On an axial-coordinate brick-wall layout, a 3x3 kernel covers 9 cells
    but only 6 are true hex neighbors (plus center).  Positions [0,0] and
    [2,2] correspond to the two non-adjacent diagonal corners.  This layer
    registers a persistent mask that zeros those kernel weights before every
    forward pass, ensuring the convolution respects hex geometry exactly.

    The mask pattern (1 = active, 0 = masked)::

        0 1 1
        1 1 1
        1 1 0
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        # Build the hex mask: all 1s except corners [0,0] and [2,2]
        mask = torch.ones(1, 1, 3, 3)
        mask[0, 0, 0, 0] = 0.0
        mask[0, 0, 2, 2] = 0.0
        self.register_buffer("hex_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Zero non-hex-neighbor kernel weights before convolution
        masked_weight = self.weight * self.hex_mask
        return F.conv2d(x, masked_weight, self.bias, self.stride, self.padding)


class HexResBlock(nn.Module):
    """Residual block using hex-masked convolutions.

    Architecture::

        x ─┬─ HexConv3x3 ─ BN ─ ReLU ─ HexConv3x3 ─ BN ─┬─ ReLU ─ out
           └──────────────── skip ────────────────────────┘

    Padding is ``same`` so spatial dimensions are preserved.  Both
    convolutions use the Z[omega] kernel mask to respect hex geometry.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = HexConv2d(channels, channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = HexConv2d(channels, channels, bias=False)
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
