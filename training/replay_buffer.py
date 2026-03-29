"""Replay buffer for storing and sampling self-play training data.

Each entry is a training sample: (state_features, policy_target, value_target,
ownership_target, threat_target).  Samples are stored as individual positions
rather than grouped by game, which simplifies sampling and capacity management.

Key features:
    - Fixed capacity with FIFO eviction via a deque
    - Recency-weighted sampling (75% from the recent half, 25% uniform)
    - On-the-fly D6 symmetry augmentation during sampling
    - Game-level metadata tracking for statistics
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Stores self-play game data for training.

    Each entry is a training sample dict with keys:
        ``features``  -- numpy array ``(C, H, W)``
        ``policy``    -- numpy array ``(H*W,)``
        ``value``     -- float
        ``ownership`` -- numpy array ``(3, H, W)`` or None
        ``threats``   -- numpy array ``(2,)`` or None
        ``game_id``   -- int identifying which game this position belongs to

    Supports:
        - Fixed capacity with FIFO eviction
        - Recency-weighted sampling (75% from recent half, 25% uniform)
        - D6 augmentation on-the-fly during sampling
    """

    def __init__(self, capacity: int = 500_000, augment: bool = True) -> None:
        """Initialize the replay buffer.

        Args:
            capacity: maximum number of individual positions stored.
            augment: whether to apply D6 symmetry augmentation when sampling.
        """
        self.capacity = capacity
        self.augment = augment
        self.buffer: deque = deque(maxlen=capacity)
        self._game_count: int = 0

    def add_game(self, game_data: List[dict]) -> None:
        """Add all positions from a single game.

        Each dict must have keys:
            ``features`` -- numpy ``(C, H, W)``
            ``policy``   -- numpy ``(H*W,)``
            ``value``    -- float
            ``ownership`` -- numpy ``(3, H, W)`` (optional, may be absent or None)
            ``threats``   -- numpy ``(2,)`` (optional, may be absent or None)

        A ``game_id`` field is injected automatically for tracking.
        """
        if not game_data:
            return

        self._game_count += 1
        gid = self._game_count

        for sample in game_data:
            entry = {
                "features": sample["features"],
                "policy": sample["policy"],
                "value": float(sample["value"]),
                "ownership": sample.get("ownership"),
                "threats": sample.get("threats"),
                "game_id": gid,
            }
            # Preserve any extra metadata (e.g. game_state for reanalyze)
            for key in sample:
                if key not in entry:
                    entry[key] = sample[key]
            self.buffer.append(entry)

        logger.debug(
            "Added game %d (%d positions) to replay buffer. Buffer size: %d",
            gid, len(game_data), len(self.buffer),
        )

    def sample(self, batch_size: int) -> dict:
        """Sample a batch with recency weighting.

        Sampling strategy: 75% of samples are drawn uniformly from the
        more-recent half of the buffer, and 25% are drawn uniformly from the
        entire buffer.  This biases training toward fresher data while
        retaining some diversity from older games.

        If D6 augmentation is enabled, a random symmetry transform is applied
        independently to each sample in the batch.

        Returns:
            Dict with torch tensors:
                ``features``  -- ``(B, C, H, W)``
                ``policy``    -- ``(B, H*W)``
                ``value``     -- ``(B, 1)``
                ``ownership`` -- ``(B, 3, H, W)`` or ``None``
                ``threats``   -- ``(B, 2)`` or ``None``
        """
        buf_len = len(self.buffer)
        if buf_len == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        actual_batch = min(batch_size, buf_len)

        # Recency-weighted index selection
        recent_count = max(1, int(actual_batch * 0.75))
        uniform_count = actual_batch - recent_count

        half_idx = buf_len // 2
        recent_start = max(0, buf_len - max(half_idx, 1))

        # Sample indices
        recent_indices = [
            random.randint(recent_start, buf_len - 1)
            for _ in range(recent_count)
        ]
        uniform_indices = [
            random.randint(0, buf_len - 1)
            for _ in range(uniform_count)
        ]
        indices = recent_indices + uniform_indices
        random.shuffle(indices)

        # Gather samples
        samples = [self.buffer[i] for i in indices]

        return self._collate(samples)

    def sample_indices(self, batch_size: int) -> List[int]:
        """Sample indices with recency weighting (for reanalyze).

        Returns a list of buffer indices.
        """
        buf_len = len(self.buffer)
        if buf_len == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        actual_batch = min(batch_size, buf_len)
        recent_count = max(1, int(actual_batch * 0.75))
        uniform_count = actual_batch - recent_count

        half_idx = buf_len // 2
        recent_start = max(0, buf_len - max(half_idx, 1))

        recent_indices = [
            random.randint(recent_start, buf_len - 1)
            for _ in range(recent_count)
        ]
        uniform_indices = [
            random.randint(0, buf_len - 1)
            for _ in range(uniform_count)
        ]
        return recent_indices + uniform_indices

    def get_entry(self, index: int) -> dict:
        """Retrieve a single entry by buffer index."""
        return self.buffer[index]

    def update_entry(self, index: int, updates: dict) -> None:
        """Update fields of a buffer entry in-place.

        Args:
            index: buffer index.
            updates: dict of key-value pairs to update.
        """
        entry = self.buffer[index]
        entry.update(updates)

    def _collate(self, samples: List[dict]) -> dict:
        """Collate a list of sample dicts into batched tensors.

        If augmentation is enabled, a random D6 symmetry transform is applied
        independently to each sample.
        """
        features_list = []
        policy_list = []
        value_list = []
        ownership_list = []
        threat_list = []

        has_ownership = samples[0].get("ownership") is not None
        has_threats = samples[0].get("threats") is not None

        for sample in samples:
            feat = torch.from_numpy(np.array(sample["features"], dtype=np.float32))
            pol = torch.from_numpy(np.array(sample["policy"], dtype=np.float32))

            # Pick a single random symmetry index to apply consistently
            # across features, policy, and ownership for this sample.
            sym_idx = random.randint(0, 11) if self.augment else None

            if self.augment and sym_idx is not None:
                feat, pol = self._apply_symmetry(feat, pol, sym_idx)

            features_list.append(feat)
            policy_list.append(pol)
            value_list.append(sample["value"])

            if has_ownership and sample.get("ownership") is not None:
                own = torch.from_numpy(
                    np.array(sample["ownership"], dtype=np.float32)
                )
                if self.augment and sym_idx is not None:
                    own = self._apply_symmetry_spatial(own, feat.shape[-1], sym_idx)
                ownership_list.append(own)
            elif has_ownership:
                # Fallback: uniform ownership if somehow missing for this sample
                gs = feat.shape[-1]
                ownership_list.append(
                    torch.full((3, gs, gs), 1.0 / 3, dtype=torch.float32)
                )

            if has_threats and sample.get("threats") is not None:
                threat_list.append(
                    torch.from_numpy(np.array(sample["threats"], dtype=np.float32))
                )
            elif has_threats:
                threat_list.append(torch.zeros(2, dtype=torch.float32))

        batch = {
            "features": torch.stack(features_list),     # (B, C, H, W)
            "policy": torch.stack(policy_list),          # (B, H*W)
            "value": torch.tensor(
                value_list, dtype=torch.float32
            ).unsqueeze(1),                              # (B, 1)
        }

        if has_ownership and ownership_list:
            batch["ownership"] = torch.stack(ownership_list)  # (B, 3, H, W)
        else:
            batch["ownership"] = None

        if has_threats and threat_list:
            batch["threats"] = torch.stack(threat_list)  # (B, 2)
        else:
            batch["threats"] = None

        return batch

    @staticmethod
    def _apply_symmetry(
        features: torch.Tensor,
        policy: torch.Tensor,
        sym_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a specific D6 symmetry to one (C, H, W) feature tensor
        and its corresponding (H*W,) policy vector.

        Args:
            features: ``(C, H, W)`` feature tensor.
            policy: ``(H*W,)`` policy vector.
            sym_idx: index into the 12 D6 symmetries (0-11).

        Uses the pre-computed remap indices from ``nn.symmetry``.
        """
        from nn.symmetry import _build_remap_indices, _apply_remap, _apply_remap_1d

        grid_size = features.shape[-1]
        all_idx = _build_remap_indices(grid_size)
        idx = all_idx[sym_idx]

        # _apply_remap expects (B, C, H, W); add and remove batch dim
        feat_4d = features.unsqueeze(0)
        pol_2d = policy.unsqueeze(0)

        feat_aug = _apply_remap(feat_4d, idx, grid_size).squeeze(0)
        pol_aug = _apply_remap_1d(pol_2d, idx, grid_size).squeeze(0)

        return feat_aug, pol_aug

    @staticmethod
    def _apply_symmetry_spatial(
        tensor: torch.Tensor,
        grid_size: int,
        sym_idx: int,
    ) -> torch.Tensor:
        """Apply a specific D6 symmetry to a (C, H, W) spatial tensor
        (e.g. ownership).

        Uses the same symmetry index as the corresponding features/policy
        so that all targets for a given sample are transformed consistently.

        Args:
            tensor: ``(C, H, W)`` spatial tensor.
            grid_size: spatial dimension.
            sym_idx: index into the 12 D6 symmetries (0-11).
        """
        from nn.symmetry import _build_remap_indices, _apply_remap

        all_idx = _build_remap_indices(grid_size)
        idx = all_idx[sym_idx]

        return _apply_remap(tensor.unsqueeze(0), idx, grid_size).squeeze(0)

    def __len__(self) -> int:
        """Number of individual positions in the buffer."""
        return len(self.buffer)

    @property
    def num_games(self) -> int:
        """Number of games that have been added (lifetime, not current)."""
        return self._game_count

    @property
    def num_games_in_buffer(self) -> int:
        """Approximate number of distinct games currently in the buffer."""
        if not self.buffer:
            return 0
        game_ids = {entry["game_id"] for entry in self.buffer}
        return len(game_ids)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity}, size={len(self)}, "
            f"games_added={self._game_count}, augment={self.augment})"
        )
