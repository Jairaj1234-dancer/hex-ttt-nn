"""Compatibility wrapper for collaborator's neural network architecture.

The collaborator's model (e.g. net_gen0162.pt) uses a different architecture
from our :class:`~nn.model.HexTTTNet`.  Key differences:

    - 17 input feature planes (vs our 12)
    - 18x18 grid (vs our 19x19)
    - 6 residual blocks (vs our configurable count)
    - Single FC in global pooling (Linear(256, 128), no ReLU/second FC)
    - Policy head: conv(128->128,1x1)+BN+ReLU -> conv(128->1,1x1) (no FC)
    - Value head: conv(128->1,1x1)+BN+ReLU -> FC(324->256)+ReLU -> FC(256->1)
    - Ownership: conv(128->1,1x1) (single channel, not 3-class)
    - Threat: conv(128->1,1x1) -> global average pool -> scalar
    - Extra heads: value_var (variance estimate)

State dict key mapping (collaborator -> this class)::

    stem.0.weight           -> stem.0.weight (Conv2d, hex_mask in stem.0)
    stem.1.*                -> stem.1.* (BatchNorm2d)
    blocks.{i}.conv1.*      -> blocks.{i}.conv1.* (Conv2d + hex_mask buffer)
    blocks.{i}.bn1.*        -> blocks.{i}.bn1.*
    blocks.{i}.conv2.*      -> blocks.{i}.conv2.* (Conv2d + hex_mask buffer)
    blocks.{i}.bn2.*        -> blocks.{i}.bn2.*
    global_pool.fc.0.*      -> global_pool.fc.0.* (Linear(256, 128))
    p_conv.0.*              -> p_conv.0.* (Conv2d(128,128,1))
    p_conv.1.*              -> p_conv.1.* (BatchNorm2d(128))
    p_conv.3.*              -> p_conv.3.* (Conv2d(128,1,1))
    v_conv.0.*              -> v_conv.0.* (Conv2d(128,1,1))
    v_conv.1.*              -> v_conv.1.* (BatchNorm2d(1))
    v_fc.0.*                -> v_fc.0.* (Linear(324,256))
    v_fc.2.*                -> v_fc.2.* (Linear(256,1))
    aux_own.0.*             -> aux_own.0.* (Conv2d(128,1,1))
    aux_threat.*            -> aux_threat.* (Conv2d(128,1,1))
    value_var.2.*           -> value_var.2.* (Linear(128,1))

This module is designed so that ``model.load_state_dict(torch.load(...))``
works directly with the collaborator's checkpoint files.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hex-masked Conv2d (matches collaborator's buffer naming)
# ---------------------------------------------------------------------------

class _HexConv2d(nn.Conv2d):
    """Conv2d with hex_mask buffer, matching the collaborator's convention.

    The collaborator registers hex_mask as a buffer on each 3x3 conv layer.
    The mask zeros out corners [0,0] and [2,2] of the 3x3 kernel, restricting
    the convolution to the 6 true hex neighbors on a brick-wall layout.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        mask = torch.ones(1, 1, 3, 3)
        mask[0, 0, 0, 0] = 0.0
        mask[0, 0, 2, 2] = 0.0
        self.register_buffer("hex_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self.hex_mask
        return F.conv2d(x, masked_weight, self.bias, self.stride, self.padding)


# ---------------------------------------------------------------------------
# Residual block (matches collaborator's blocks.{i}.* naming)
# ---------------------------------------------------------------------------

class _CompatResBlock(nn.Module):
    """Residual block matching the collaborator's architecture.

    Structure: conv1(hex) -> bn1 -> relu -> conv2(hex) -> bn2 -> skip -> relu
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = _HexConv2d(channels, channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = _HexConv2d(channels, channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


# ---------------------------------------------------------------------------
# CompatNet — full model matching the collaborator's state_dict
# ---------------------------------------------------------------------------

class CompatNet(nn.Module):
    """Neural network matching the collaborator's architecture exactly.

    This class mirrors the collaborator's model structure so that their
    checkpoint files can be loaded directly via ``load_state_dict()``.
    The ``forward()`` method returns the same dict format as
    :class:`~nn.model.HexTTTNet` for seamless integration with MCTS
    and evaluation code.

    Args:
        grid_size: spatial dimension of the input (H = W = grid_size).
            Default 18 (collaborator's grid).
        in_channels: number of input feature planes. Default 17
            (collaborator's feature set).
        channels: internal channel width. Default 128.
        num_blocks: number of residual blocks. Default 6.
    """

    def __init__(
        self,
        grid_size: int = 18,
        in_channels: int = 17,
        channels: int = 128,
        num_blocks: int = 6,
        wdl_value: bool = False,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.channels = channels
        self.wdl_value = wdl_value
        hw = grid_size * grid_size  # 324 for 18x18

        # ---- Stem: HexConv2d + BatchNorm (stem.0, stem.1) ----
        self.stem = nn.Sequential(
            _HexConv2d(in_channels, channels, bias=False),  # stem.0
            nn.BatchNorm2d(channels),                        # stem.1
        )

        # ---- Residual tower (blocks.0 through blocks.5) ----
        self.blocks = nn.ModuleList(
            [_CompatResBlock(channels) for _ in range(num_blocks)]
        )

        # ---- Global pooling: GAP + GMP -> concat(256) -> FC(256->128) ----
        # The collaborator uses a single Linear inside a Sequential.
        # State dict key: global_pool.fc.0.weight, global_pool.fc.0.bias
        self.global_pool = nn.Module()
        self.global_pool.fc = nn.Sequential(
            nn.Linear(2 * channels, channels),  # fc.0
        )

        # ---- Policy head (p_conv.0: Conv+BN+ReLU, p_conv.3: Conv) ----
        # Sequential indices: 0=Conv2d, 1=BN, 2=ReLU, 3=Conv2d
        self.p_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),   # p_conv.0
            nn.BatchNorm2d(channels),                        # p_conv.1
            nn.ReLU(inplace=True),                           # p_conv.2
            nn.Conv2d(channels, 1, kernel_size=1),           # p_conv.3
        )

        # ---- Value head ----
        # v_conv: Conv2d(128,1,1) + BN (indices 0, 1)
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),  # v_conv.0
            nn.BatchNorm2d(1),                       # v_conv.1
        )
        # v_fc: Linear(324,256) + ReLU + Linear(256, value_dim)
        # gen162: value_dim=1 (scalar tanh), gen222+: value_dim=3 (WDL softmax)
        value_dim = 3 if wdl_value else 1
        self.v_fc = nn.Sequential(
            nn.Linear(hw, 256),          # v_fc.0
            nn.ReLU(inplace=True),       # v_fc.1
            nn.Linear(256, value_dim),   # v_fc.2
        )

        # ---- Auxiliary: ownership head (aux_own.0: Conv2d) ----
        self.aux_own = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),  # aux_own.0
        )

        # ---- Auxiliary: threat head (aux_threat: Conv2d) ----
        # Single conv producing 1-channel map, then global avg pool in forward()
        self.aux_threat = nn.Conv2d(channels, 1, kernel_size=1)

        # ---- Value variance head (value_var.2: Linear(128,1)) ----
        # The collaborator has a small head for estimating value variance.
        # State dict: value_var.2.weight, value_var.2.bias
        # Indices suggest 0=something, 1=something, 2=Linear.
        # We use a Sequential with placeholder layers at indices 0 and 1.
        self.value_var = nn.Sequential(
            nn.Identity(),              # value_var.0 (placeholder)
            nn.ReLU(inplace=True),      # value_var.1 (placeholder)
            nn.Linear(channels, 1),     # value_var.2
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_moves_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning the same dict format as HexTTTNet.

        Args:
            x: ``(B, in_channels, H, W)`` input features.
            valid_moves_mask: ``(B, H*W)`` binary mask where 1 = legal move.

        Returns:
            Dict with keys:
                - ``policy_logits``:    ``(B, H*W)``
                - ``policy``:           ``(B, H*W)``
                - ``value``:            ``(B, 1)``
                - ``ownership``:        ``(B, 1, H, W)`` raw logits
                - ``ownership_logits``: ``(B, 1, H, W)`` raw logits
                - ``threats``:          ``(B, 1)`` scalar threat signal
        """
        B = x.size(0)
        H = x.size(2)
        W = x.size(3)
        hw = H * W

        # ---- Stem ----
        out = F.relu(self.stem(x))  # (B, C, H, W)

        # ---- Residual tower ----
        for block in self.blocks:
            out = block(out)  # (B, C, H, W)

        # ---- Global pooling branch ----
        gap = out.mean(dim=(2, 3))   # (B, C)
        gmp = out.amax(dim=(2, 3))   # (B, C)
        pooled = torch.cat([gap, gmp], dim=1)  # (B, 2*C)
        global_feat = self.global_pool.fc(pooled)  # (B, C)

        # Broadcast and add to trunk
        trunk = out + global_feat.unsqueeze(-1).unsqueeze(-1)  # (B, C, H, W)

        # ---- Policy head ----
        p = self.p_conv(trunk)  # (B, 1, H, W)
        policy_logits = p.reshape(B, -1)  # (B, H*W)

        if valid_moves_mask is not None:
            policy_logits = policy_logits.masked_fill(
                valid_moves_mask == 0, float("-inf")
            )

        policy = F.softmax(policy_logits, dim=1)  # (B, H*W)

        # ---- Value head ----
        v = F.relu(self.v_conv(trunk))  # (B, 1, H, W)
        v = v.reshape(B, -1)            # (B, H*W)  i.e. (B, 324)
        v_out = self.v_fc(v)            # (B, 1) or (B, 3)
        if self.wdl_value:
            # WDL: convert (win, draw, loss) probabilities to scalar value in [-1, 1]
            wdl_probs = F.softmax(v_out, dim=1)  # (B, 3)
            value = (wdl_probs[:, 0] - wdl_probs[:, 2]).unsqueeze(1)  # win - loss → [-1, 1]
        else:
            value = torch.tanh(v_out)  # (B, 1)

        # ---- Ownership head ----
        ownership_logits = self.aux_own(trunk)  # (B, 1, H, W)
        # The collaborator outputs a single channel (not 3-class).
        # We return the raw logits; sigmoid can be applied externally.
        ownership = torch.sigmoid(ownership_logits)  # (B, 1, H, W)

        # ---- Threat head ----
        threat_map = self.aux_threat(trunk)  # (B, 1, H, W)
        threats = threat_map.mean(dim=(2, 3))  # (B, 1) global avg pool

        return {
            "policy_logits": policy_logits,
            "policy": policy,
            "value": value,
            "ownership": ownership,
            "ownership_logits": ownership_logits,
            "threats": threats,
        }


def load_compat_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    grid_size: int = 18,
    in_channels: int = 17,
    channels: int = 128,
    num_blocks: int = 6,
) -> CompatNet:
    """Load a collaborator's checkpoint into a CompatNet.

    Auto-detects WDL value head (gen222+) vs scalar value head (gen162).

    Args:
        checkpoint_path: path to the ``.pt`` file (e.g. ``net_gen0162.pt``).
        device: target device (default: CPU).
        grid_size: spatial dimension. Default 18.
        in_channels: input feature planes. Default 17.
        channels: internal channel width. Default 128.
        num_blocks: number of residual blocks. Default 6.

    Returns:
        :class:`CompatNet` with loaded weights, in eval mode.
    """
    if device is None:
        device = torch.device("cpu")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Some checkpoints wrap the state_dict inside a dict (e.g. {"model": ...}).
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    # Auto-detect WDL value head from checkpoint shape
    wdl_value = False
    if "v_fc.2.weight" in state_dict:
        v_fc_out_dim = state_dict["v_fc.2.weight"].shape[0]
        wdl_value = (v_fc_out_dim == 3)

    model = CompatNet(
        grid_size=grid_size,
        in_channels=in_channels,
        channels=channels,
        num_blocks=num_blocks,
        wdl_value=wdl_value,
    )

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model
