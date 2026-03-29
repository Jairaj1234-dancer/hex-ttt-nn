"""Self-play game generation for AlphaZero-style training.

Each self-play game uses MCTS driven by the current neural network to
select moves.  The resulting trajectory of (state, MCTS policy, outcome)
triples forms the training data for the next network iteration.

The module properly handles the 2-move-per-turn mechanic and the
first-turn exception (Player 1 places only 1 stone on turn 1).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from game.hex_grid import HexCoord, axial_to_brick, brick_to_axial
from game.rules import GameState
from nn.features import extract_features
from nn.model import HexTTTNet

logger = logging.getLogger(__name__)


def policy_to_grid(
    policy_dict: Dict[HexCoord, float],
    center_q: int,
    center_r: int,
    grid_size: int,
) -> np.ndarray:
    """Convert a ``{HexCoord: probability}`` dict to a flat ``(grid_size*grid_size,)``
    array on the brick-wall grid.

    Each hex coordinate is mapped to a row/col position using
    :func:`~game.hex_grid.axial_to_brick` with the given window centre.
    Coordinates that fall outside the grid are dropped and probabilities are
    renormalised.

    Args:
        policy_dict: mapping from hex coordinates to visit-count probabilities.
        center_q: axial q of the feature window centre.
        center_r: axial r of the feature window centre.
        grid_size: spatial dimension of the grid (H = W).

    Returns:
        Flat ``(grid_size * grid_size,)`` float32 numpy array summing to ~1.0.
    """
    half = grid_size // 2
    hw = grid_size * grid_size
    policy_arr = np.zeros(hw, dtype=np.float32)

    for coord, prob in policy_dict.items():
        row, col = axial_to_brick(coord.q, coord.r, center_q, center_r, grid_size)
        row_idx = row + half
        col_idx = col + half
        if 0 <= row_idx < grid_size and 0 <= col_idx < grid_size:
            flat_idx = row_idx * grid_size + col_idx
            policy_arr[flat_idx] = prob

    # Renormalise in case some probability mass fell outside the window
    total = policy_arr.sum()
    if total > 0:
        policy_arr /= total
    else:
        # Fallback: if all mass was outside window, uniform over valid cells
        logger.warning(
            "All policy mass fell outside the grid window (center=(%d,%d), "
            "grid_size=%d). Using uniform fallback.",
            center_q, center_r, grid_size,
        )
        policy_arr[:] = 1.0 / hw

    return policy_arr


def grid_to_policy(
    flat_idx: int,
    center_q: int,
    center_r: int,
    grid_size: int,
) -> HexCoord:
    """Convert a flat grid index back to a HexCoord.

    Args:
        flat_idx: index into the ``(grid_size * grid_size,)`` array.
        center_q: axial q of the feature window centre.
        center_r: axial r of the feature window centre.
        grid_size: spatial dimension of the grid.

    Returns:
        The corresponding HexCoord.
    """
    half = grid_size // 2
    row_idx = flat_idx // grid_size
    col_idx = flat_idx % grid_size
    row = row_idx - half
    col = col_idx - half
    return brick_to_axial(row, col, center_q, center_r, grid_size)


class SelfPlayWorker:
    """Generates self-play games using MCTS.

    Each game produces a list of training samples with:
        - state features (from the current player's perspective)
        - MCTS policy (visit distribution)
        - game outcome (from the current player's perspective)
    """

    def __init__(self, network: HexTTTNet, config: dict) -> None:
        """Initialize the self-play worker.

        Args:
            network: the neural network to use for MCTS evaluation.
            config: merged game + mcts + training config dict.
        """
        self.network = network
        self.config = config
        self.grid_size: int = config.get("network", {}).get("grid_size", 19)

        # MCTS settings
        mcts_cfg = config.get("mcts", {})
        self.temperature_moves: int = mcts_cfg.get("temperature_moves", 10)
        self.temperature_final: float = mcts_cfg.get("temperature_final", 0.3)
        self.temperature_mid: Optional[float] = mcts_cfg.get("temperature_mid")
        self.temperature_mid_moves: Optional[int] = mcts_cfg.get("temperature_mid_moves")

        # Playout cap settings
        playout_cfg = config.get("playout_cap", {})
        self.playout_cap_enabled: bool = playout_cfg.get("enabled", False)
        self.playout_full_ratio: float = playout_cfg.get("full_ratio", 0.25)
        self.num_simulations: int = mcts_cfg.get("num_simulations", 200)
        self.num_simulations_reduced: int = mcts_cfg.get(
            "num_simulations_reduced", self.num_simulations // 4
        )

    def _get_temperature(self, half_move: int) -> float:
        """Determine the temperature for a given half-move number.

        Half-move numbering starts at 0.

        Uses cosine annealing for a smooth temperature decay (inspired by
        hexgo): ``tau = max(tau_final, cos(pi * half_move / horizon))``.
        This avoids the sharp cliff of a step schedule and produces a more
        natural exploration-to-exploitation transition.

        Falls back to the legacy step schedule if ``temperature_schedule``
        is set to ``"step"`` in the config.
        """
        schedule = self.config.get("mcts", {}).get("temperature_schedule", "cosine")

        if schedule == "step":
            # Legacy step schedule
            if half_move < self.temperature_moves:
                return 1.0
            if (
                self.temperature_mid is not None
                and self.temperature_mid_moves is not None
                and half_move < self.temperature_mid_moves
            ):
                return self.temperature_mid
            return self.temperature_final

        # Cosine annealing (default)
        import math
        horizon = self.temperature_moves * 3  # full cosine period
        tau = math.cos(math.pi * half_move / horizon)
        return max(self.temperature_final, tau)

    def _get_num_simulations(self) -> int:
        """Get the number of MCTS simulations for the current move.

        If playout cap randomization is enabled, randomly choose between
        full and reduced simulation counts.
        """
        if not self.playout_cap_enabled:
            return self.num_simulations

        import random
        if random.random() < self.playout_full_ratio:
            return self.num_simulations
        return self.num_simulations_reduced

    def play_game(self) -> List[dict]:
        """Play a complete self-play game and return training data.

        Steps:
            1. Initialize a fresh GameState.
            2. At each position:
               a. Extract features and determine the window centre.
               b. Run MCTS search to get a visit distribution.
               c. Select a move using the temperature schedule.
               d. Record (features, policy, current_player).
               e. Apply the move to advance the game state.
            3. When the game ends, assign value targets:
               +1 for the winner's positions, -1 for the loser's, 0 for draw.
            4. Return the list of sample dicts.

        Returns:
            List of sample dicts, each with keys:
                ``features``    -- numpy ``(C, H, W)``
                ``policy``      -- numpy ``(H*W,)``
                ``value``       -- float
                ``center``      -- ``(center_q, center_r)``
                ``game_state``  -- the GameState at that position (for reanalyze)
        """
        from mcts.search import MCTS

        game_state = GameState()
        trajectory: List[dict] = []
        half_move = 0

        # Create MCTS instance with current network
        mcts_config = dict(self.config.get("mcts", {}))
        mcts = MCTS(self.network, mcts_config)
        prev_root = None  # For tree reuse between moves

        self.network.eval()

        while not game_state.is_terminal:
            # Extract features for the current position
            with torch.no_grad():
                features, (center_q, center_r) = extract_features(
                    game_state, grid_size=self.grid_size
                )

            # Determine temperature for this half-move
            temperature = self._get_temperature(half_move)

            # Optionally adjust simulation count (playout cap)
            num_sims = self._get_num_simulations()
            mcts_config["num_simulations"] = num_sims

            # Run MCTS search with tree reuse
            with torch.no_grad():
                move, policy_dict, prev_root = mcts.get_move(
                    game_state, temperature=temperature, prev_root=prev_root
                )

            # Convert policy dict to flat grid array
            policy_grid = policy_to_grid(
                policy_dict, center_q, center_r, self.grid_size
            )

            # Record this position's data
            record = {
                "features": features.numpy(),         # (C, H, W)
                "policy": policy_grid,                 # (H*W,)
                "current_player": game_state.current_player,
                "center": (center_q, center_r),
                "game_state": game_state.copy(),
            }
            trajectory.append(record)

            # Apply move
            game_state = game_state.apply_move(move)
            half_move += 1

            # Safety: prevent infinite games (very unlikely but defensive)
            if half_move > 1000:
                logger.warning(
                    "Self-play game exceeded 1000 half-moves; terminating as draw."
                )
                break

        # Assign value targets based on game outcome.
        # Uses TD-lambda discounted targets (inspired by hexgo):
        #   z_t = gamma^(T-1-t) * z_final
        # Early positions receive a discounted signal, making them "less
        # certain" — this helps the value head converge faster by not
        # over-fitting early game positions to the final binary outcome.
        winner = game_state.winner
        game_data: List[dict] = []
        total_positions = len(trajectory)
        td_gamma: float = self.config.get("training", {}).get("td_gamma", 0.99)

        for t, record in enumerate(trajectory):
            if winner is None:
                value = 0.0
            else:
                raw_value = 1.0 if record["current_player"] == winner else -1.0
                discount = td_gamma ** (total_positions - 1 - t)
                value = discount * raw_value

            sample = {
                "features": record["features"],
                "policy": record["policy"],
                "value": value,
                "center": record["center"],
                "game_state": record["game_state"],
            }
            game_data.append(sample)

        logger.debug(
            "Self-play game finished: %d half-moves, winner=%s",
            half_move, winner,
        )

        return game_data

    def play_games(self, num_games: int) -> List[List[dict]]:
        """Play multiple self-play games.

        Args:
            num_games: number of games to play sequentially.

        Returns:
            List of game data lists, one per game.
        """
        all_games: List[List[dict]] = []

        for i in range(num_games):
            logger.info("Starting self-play game %d / %d", i + 1, num_games)
            game_data = self.play_game()
            all_games.append(game_data)
            logger.info(
                "Completed self-play game %d / %d: %d positions",
                i + 1, num_games, len(game_data),
            )

        total_positions = sum(len(g) for g in all_games)
        logger.info(
            "Self-play batch complete: %d games, %d total positions",
            num_games, total_positions,
        )

        return all_games
