"""AlphaZero-style dual-headed network for Infinite Hex Tic-Tac-Toe.

Architecture overview::

    input (B, 12, H, W)
        |
    HexResNet backbone (shared representation)
        |
        +--- Policy head ---> (B, H*W) move logits / probabilities
        |
        +--- Value head ----> (B, 1)  position evaluation in [-1, 1]
        |
        +--- Ownership head -> (B, 3, H, W) per-cell ownership probs
        |
        +--- Threat head ---> (B, 2) predicted threat counts

The auxiliary ownership and threat heads provide extra training signal
that helps the shared backbone learn richer representations (a technique
popularised by KataGo).  At inference time they can be ignored, or used
for analysis/debugging.

Loss recipe:
    - **Policy**: cross-entropy against soft MCTS visit-count targets.
    - **Value**: mean squared error against game outcome.
    - **Ownership**: cross-entropy (per-cell, 3-class).
    - **Threat**: mean squared error against threat counts.
    - **L2 regularisation**: applied via ``weight_decay`` in the optimiser
      (e.g. ``torch.optim.AdamW(model.parameters(), weight_decay=1e-4)``)
      rather than computed manually in this loss function.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.hex_conv import HexResNet
from nn.features import NUM_INPUT_PLANES


class HexTTTNet(nn.Module):
    """Dual-headed network for Infinite Hex Tic-Tac-Toe with auxiliary heads.

    Args:
        grid_size: spatial dimension of the input (H = W = grid_size).
        num_blocks: number of residual blocks in the backbone.
        channels: internal channel width of the backbone.
        in_channels: number of input feature planes.
    """

    def __init__(
        self,
        grid_size: int = 19,
        num_blocks: int = 8,
        channels: int = 128,
        in_channels: int = NUM_INPUT_PLANES,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.channels = channels
        hw = grid_size * grid_size

        # ---- Shared backbone ----
        self.backbone = HexResNet(in_channels, num_blocks, channels)

        # ---- Policy head (fully convolutional) ----
        # Conv(ch→ch, 1x1) → BN → ReLU → Conv(ch→1, 1x1)
        # Spatially aware — no FC bottleneck.
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

        # ---- Value head ----
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(hw, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # ---- Auxiliary: ownership head ----
        # Predicts per-cell ownership: 3 classes (my / opponent / empty).
        self.ownership_conv = nn.Conv2d(channels, 3, kernel_size=1)

        # ---- Auxiliary: threat head ----
        # Predicts (my_threat_count, opp_threat_count) from global features.
        self.threat_fc1 = nn.Linear(channels, 64)
        self.threat_fc2 = nn.Linear(64, 2)

    def forward(
        self,
        x: torch.Tensor,
        valid_moves_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``(B, in_channels, H, W)`` input features.
            valid_moves_mask: ``(B, H*W)`` binary mask where 1 = legal move.
                When provided, illegal-move logits are set to ``-inf``
                before softmax.

        Returns:
            Dict with keys:
                - ``policy_logits``:    ``(B, H*W)`` raw logits (masked if mask given).
                - ``policy``:           ``(B, H*W)`` softmax probabilities.
                - ``value``:            ``(B, 1)``   value in [-1, 1].
                - ``ownership``:        ``(B, 3, H, W)`` ownership probs (softmax over dim=1).
                - ``ownership_logits``: ``(B, 3, H, W)`` raw ownership logits.
                - ``threats``:          ``(B, 2)`` predicted threat counts.
        """
        B = x.size(0)

        # ---- Backbone ----
        trunk = self.backbone(x)  # (B, C, H, W)

        # ---- Policy head (fully convolutional) ----
        p = self.policy_head(trunk)  # (B, 1, H, W)
        policy_logits = p.reshape(B, -1)  # (B, H*W)

        if valid_moves_mask is not None:
            # Set logits for illegal moves to -inf so they get zero probability.
            policy_logits = policy_logits.masked_fill(valid_moves_mask == 0, float("-inf"))

        policy = F.softmax(policy_logits, dim=1)  # (B, H*W)

        # ---- Value head ----
        v = F.relu(self.value_bn(self.value_conv(trunk)))  # (B, 1, H, W)
        v = v.reshape(B, -1)  # (B, H*W)
        v = F.relu(self.value_fc1(v))  # (B, 256)
        value = torch.tanh(self.value_fc2(v))  # (B, 1)

        # ---- Ownership head ----
        ownership_logits = self.ownership_conv(trunk)  # (B, 3, H, W)
        ownership = F.softmax(ownership_logits, dim=1)  # (B, 3, H, W)

        # ---- Threat head ----
        trunk_pooled = trunk.mean(dim=(2, 3))  # (B, C) -- global avg pool
        t = F.relu(self.threat_fc1(trunk_pooled))  # (B, 64)
        threats = self.threat_fc2(t)  # (B, 2)

        return {
            "policy_logits": policy_logits,
            "policy": policy,
            "value": value,
            "ownership": ownership,
            "ownership_logits": ownership_logits,
            "threats": threats,
        }

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined training loss.

        Args:
            predictions: output dict from :meth:`forward`.
            targets: dict with keys:
                - ``policy``: ``(B, H*W)`` MCTS visit-count distribution (soft target).
                - ``value``:  ``(B, 1)`` game outcome in {-1, +1}.
                - ``ownership``: ``(B, 3, H, W)`` ownership labels (optional).
                - ``threats``: ``(B, 2)`` threat count labels (optional).
            config: loss weighting hyperparameters:
                - ``value_weight``     (default 1.0)
                - ``policy_weight``    (default 1.0)
                - ``ownership_weight`` (default 0.15)
                - ``threat_weight``    (default 0.15)
                - ``l2_weight``        (default 1e-4) -- NOTE: L2 is applied
                  via ``weight_decay`` in the optimiser, not computed here.
                  This key is kept for documentation / config completeness.

        Returns:
            Dict with keys ``total``, ``value``, ``policy``, ``ownership``,
            ``threat`` (all scalar tensors).
        """
        if config is None:
            config = {}

        value_w: float = config.get("value_weight", 1.0)
        policy_w: float = config.get("policy_weight", 1.0)
        ownership_w: float = config.get("ownership_weight", 0.15)
        threat_w: float = config.get("threat_weight", 0.15)
        # l2_weight is intentionally unused here -- it should be passed
        # as weight_decay to the optimizer (e.g. AdamW).

        # ---- Policy loss: cross-entropy against soft targets ----
        # policy_logits: (B, H*W), target_policy: (B, H*W) probability distribution.
        # CE(p, q) = -sum(q * log(p)).  Using log_softmax for numerical stability.
        log_probs = F.log_softmax(predictions["policy_logits"], dim=1)  # (B, H*W)
        policy_loss = -torch.sum(targets["policy"] * log_probs, dim=1).mean()  # scalar

        # ---- Value loss: MSE ----
        value_loss = F.mse_loss(predictions["value"], targets["value"])  # scalar

        # ---- Ownership loss (optional): cross-entropy per cell ----
        if "ownership" in targets:
            # Use log_softmax on raw logits for numerical stability.
            ownership_logits = predictions["ownership_logits"]  # (B, 3, H, W)
            ownership_target = targets["ownership"]  # (B, 3, H, W)
            # Cross-entropy: -sum(target * log_softmax(logits)) over the 3-class dim.
            ownership_log_probs = F.log_softmax(ownership_logits, dim=1)  # (B, 3, H, W)
            ownership_loss = -torch.sum(
                ownership_target * ownership_log_probs, dim=1
            ).mean()  # mean over B, H, W
        else:
            ownership_loss = torch.tensor(0.0, device=predictions["value"].device)

        # ---- Threat loss (optional): MSE ----
        if "threats" in targets:
            threat_loss = F.mse_loss(predictions["threats"], targets["threats"])
        else:
            threat_loss = torch.tensor(0.0, device=predictions["value"].device)

        # ---- Total ----
        total = (
            policy_w * policy_loss
            + value_w * value_loss
            + ownership_w * ownership_loss
            + threat_w * threat_loss
        )

        return {
            "total": total,
            "policy": policy_loss,
            "value": value_loss,
            "ownership": ownership_loss,
            "threat": threat_loss,
        }
