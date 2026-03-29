"""Reanalyze module: re-search stored positions with the latest network.

Implements the MuZero Reanalyze technique adapted for AlphaZero:
    - Sample positions from the replay buffer
    - Re-run MCTS with the latest network to generate fresher policy targets
    - Optionally blend the new value estimate with the original game outcome

This keeps the replay buffer's policy targets from going stale as the network
improves, providing higher-quality training signal from older games without
the cost of replaying them entirely.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

from game.rules import GameState
from nn.features import extract_features
from nn.model import HexTTTNet
from training.replay_buffer import ReplayBuffer
from training.self_play import policy_to_grid

logger = logging.getLogger(__name__)


class Reanalyzer:
    """Re-searches stored positions with the latest network to generate
    fresher policy targets.

    The reanalyzer samples positions from the replay buffer, reconstructs
    game states, runs MCTS with the current network, and updates the stored
    policy (and optionally value) targets in place.
    """

    def __init__(
        self,
        network: HexTTTNet,
        config: dict,
        device: str = "cpu",
    ) -> None:
        """Initialize the reanalyzer.

        Args:
            network: the latest neural network for MCTS evaluation.
            config: merged configuration dict.
            device: torch device string.
        """
        self.network = network
        self.config = config
        self.device = device

        self.grid_size: int = config.get("network", {}).get("grid_size", 19)

        reanalyze_cfg = config.get("reanalysis", {})
        self.default_batch_size: int = reanalyze_cfg.get("batch_size", 64)
        self.value_blend_weight: float = reanalyze_cfg.get(
            "value_blend_weight", 0.3
        )

        mcts_cfg = config.get("mcts", {})
        # Use fewer simulations for reanalysis than full self-play
        self.reanalyze_simulations: int = reanalyze_cfg.get(
            "num_simulations",
            mcts_cfg.get("num_simulations_reduced", mcts_cfg.get("num_simulations", 200) // 2),
        )

    def reanalyze_batch(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = None,
    ) -> int:
        """Sample positions from the replay buffer, re-search them with the
        current network, and update policy (and optionally value) targets.

        Steps:
            1. Sample ``batch_size`` position indices from the buffer.
            2. For each position, check if a ``game_state`` is stored.
               If not, skip (only feature-based positions without state cannot
               be reanalyzed).
            3. Run MCTS with the current network on each game state.
            4. Replace the stored policy target with the new MCTS policy.
            5. Blend the value target: ``0.7 * original_outcome + 0.3 * network_value``.

        Args:
            replay_buffer: the replay buffer to reanalyze from.
            batch_size: number of positions to reanalyze. Defaults to
                the configured value.

        Returns:
            The number of positions successfully reanalyzed.
        """
        from mcts.search import MCTS

        if batch_size is None:
            batch_size = self.default_batch_size

        if len(replay_buffer) == 0:
            logger.debug("Replay buffer is empty; skipping reanalysis.")
            return 0

        # Sample indices
        indices = replay_buffer.sample_indices(batch_size)

        # Build MCTS config for reanalysis (reduced simulations, no noise)
        mcts_config = dict(self.config.get("mcts", {}))
        mcts_config["num_simulations"] = self.reanalyze_simulations
        mcts_config["dirichlet_epsilon"] = 0.0  # no exploration noise

        mcts = MCTS(self.network, mcts_config)
        self.network.eval()

        reanalyzed_count = 0

        for buf_idx in indices:
            entry = replay_buffer.get_entry(buf_idx)

            # We need the game state to re-run MCTS
            game_state: Optional[GameState] = entry.get("game_state")
            if game_state is None:
                # Cannot reanalyze without the game state
                continue

            if game_state.is_terminal:
                # Terminal states have no policy to reanalyze
                continue

            try:
                # Re-run MCTS on this position
                with torch.no_grad():
                    _, new_policy_dict = mcts.get_move(
                        game_state, temperature=1.0
                    )

                # Determine the window centre for this position
                center = entry.get("center")
                if center is not None:
                    center_q, center_r = center
                else:
                    # Recompute the centre from the game state
                    _, (center_q, center_r) = extract_features(
                        game_state, grid_size=self.grid_size
                    )

                # Convert new policy dict to flat grid array
                new_policy = policy_to_grid(
                    new_policy_dict, center_q, center_r, self.grid_size
                )

                # Update the policy target in the buffer
                updates = {"policy": new_policy}

                # Optionally blend the value target
                if self.value_blend_weight > 0:
                    original_value = entry["value"]

                    # Get the network's value estimate for this position
                    with torch.no_grad():
                        features, _ = extract_features(
                            game_state, grid_size=self.grid_size
                        )
                        features_batch = features.unsqueeze(0).to(self.device)
                        net_output = self.network(features_batch)
                        network_value = net_output["value"].item()

                    # Blend: keep most of the original game outcome,
                    # mix in some of the network's current estimate
                    outcome_weight = 1.0 - self.value_blend_weight
                    blended_value = (
                        outcome_weight * original_value
                        + self.value_blend_weight * network_value
                    )
                    updates["value"] = float(blended_value)

                replay_buffer.update_entry(buf_idx, updates)
                reanalyzed_count += 1

            except Exception:
                logger.warning(
                    "Failed to reanalyze position at buffer index %d",
                    buf_idx,
                    exc_info=True,
                )
                continue

        if reanalyzed_count > 0:
            logger.info(
                "Reanalyzed %d / %d sampled positions (simulations=%d)",
                reanalyzed_count, len(indices), self.reanalyze_simulations,
            )

        return reanalyzed_count

    def update_network(self, network: HexTTTNet) -> None:
        """Replace the network used for reanalysis with a newer version.

        This should be called after each training iteration so that
        reanalysis uses the latest weights.

        Args:
            network: the updated network.
        """
        self.network = network
        logger.debug("Reanalyzer network updated.")
