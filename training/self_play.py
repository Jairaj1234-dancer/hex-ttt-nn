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
        self.win_length: int = config.get("game", {}).get("win_length", 6)

        # MCTS settings
        mcts_cfg = config.get("mcts", {})
        self.temperature_moves: int = mcts_cfg.get("temperature_moves", 10)
        self.temperature_final: float = mcts_cfg.get("temperature_final", 0.3)
        self.temperature_mid: Optional[float] = mcts_cfg.get("temperature_mid")
        self.temperature_mid_moves: Optional[int] = mcts_cfg.get("temperature_mid_moves")

        # Game length limit — forces the network to find wins fast
        self.max_moves: int = mcts_cfg.get("max_moves", 1000)
        self.zoi_margin: int = mcts_cfg.get("zoi_margin", 3)

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

        game_state = GameState(win_length=self.win_length)
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

            # Force short games — the NN must learn to win quickly
            if half_move >= self.max_moves:
                logger.debug(
                    "Self-play game hit %d move limit; terminating as draw.",
                    self.max_moves,
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

    def play_curriculum_game(self, opponent_fn, use_mcts: bool = True) -> Tuple[List[dict], bool]:
        """Play a game where the NN faces an external opponent.

        Can use either MCTS or raw policy for move selection. Raw policy
        is useful early in training when the value head is uncalibrated
        and MCTS actually degrades play quality.

        Args:
            opponent_fn: callable ``(GameState) -> HexCoord`` for the
                curriculum opponent.
            use_mcts: if True, use MCTS for move selection; if False,
                use raw policy network output (greedy argmax).

        Returns:
            Tuple of (sample dicts list, bool indicating if NN won).
        """
        from mcts.search import MCTS
        from game.hex_grid import axial_to_brick, brick_to_axial
        import random as _rng

        game_state = GameState(win_length=self.win_length)
        trajectory: List[dict] = []
        half_move = 0

        mcts_config = dict(self.config.get("mcts", {}))
        mcts = MCTS(self.network, mcts_config) if use_mcts else None
        prev_root = None

        self.network.eval()

        # Randomly assign NN to P1 or P2
        nn_player = _rng.choice([1, 2])

        while not game_state.is_terminal:
            is_nn_turn = (game_state.current_player == nn_player)

            if is_nn_turn:
                with torch.no_grad():
                    features, (center_q, center_r) = extract_features(
                        game_state, grid_size=self.grid_size
                    )

                if use_mcts:
                    # MCTS move selection
                    num_sims = self._get_num_simulations()
                    mcts_config["num_simulations"] = num_sims
                    with torch.no_grad():
                        move, policy_dict, prev_root = mcts.get_move(
                            game_state, temperature=0.3, prev_root=prev_root
                        )
                    policy_grid = policy_to_grid(
                        policy_dict, center_q, center_r, self.grid_size
                    )
                else:
                    # Raw policy with temperature sampling for exploration
                    legal = game_state.legal_moves(zoi_margin=self.zoi_margin)
                    valid_mask = torch.zeros(1, self.grid_size * self.grid_size)
                    legal_map = {}
                    for m in legal:
                        bx, by = axial_to_brick(m.q, m.r, center_q, center_r, self.grid_size)
                        if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                            idx = by * self.grid_size + bx
                            valid_mask[0, idx] = 1.0
                            legal_map[idx] = m

                    if not legal_map:
                        move = _rng.choice(legal)
                        policy_grid = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
                    else:
                        with torch.no_grad():
                            out = self.network(features.unsqueeze(0), valid_moves_mask=valid_mask)
                        policy = out["policy"][0]
                        policy_grid = policy.numpy()

                        # Temperature sampling with low temp for short games
                        temp = 0.3  # mild exploration, not the full cosine schedule
                        legal_indices = list(legal_map.keys())
                        if temp < 0.1:
                            # Greedy
                            chosen_idx = max(legal_indices, key=lambda i: policy_grid[i])
                        else:
                            # Sample from sharpened distribution
                            probs = np.array([policy_grid[i] for i in legal_indices])
                            probs = probs ** (1.0 / temp)
                            probs /= probs.sum() + 1e-8
                            chosen_idx = legal_indices[np.random.choice(len(legal_indices), p=probs)]
                        move = legal_map[chosen_idx]

                record = {
                    "features": features.numpy(),
                    "policy": policy_grid,
                    "current_player": game_state.current_player,
                    "center": (center_q, center_r),
                    "game_state": game_state.copy(),
                }
                trajectory.append(record)
            else:
                # Opponent move — also record as training data (teacher signal)
                move = opponent_fn(game_state)
                prev_root = None

                # Extract features from opponent's perspective and create
                # a one-hot policy target for the opponent's chosen move.
                # This teaches the NN the opponent's blocking/winning patterns.
                with torch.no_grad():
                    opp_features, (opp_cq, opp_cr) = extract_features(
                        game_state, grid_size=self.grid_size
                    )
                opp_bx, opp_by = axial_to_brick(
                    move.q, move.r, opp_cq, opp_cr, self.grid_size
                )
                if 0 <= opp_bx < self.grid_size and 0 <= opp_by < self.grid_size:
                    opp_policy = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
                    opp_idx = opp_by * self.grid_size + opp_bx
                    opp_policy[opp_idx] = 1.0
                    trajectory.append({
                        "features": opp_features.numpy(),
                        "policy": opp_policy,
                        "current_player": game_state.current_player,
                        "center": (opp_cq, opp_cr),
                        "game_state": game_state.copy(),
                    })

            game_state = game_state.apply_move(move)
            half_move += 1

            if half_move >= self.max_moves:
                logger.debug("Curriculum game hit %d move limit; draw.", self.max_moves)
                break

        # Assign value targets with TD-lambda
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

        result_str = f"winner=P{winner}" if winner else "draw"
        nn_won = (winner == nn_player) if winner is not None else False
        logger.debug(
            "Curriculum game: NN=P%d, %s, %d half-moves, %d NN positions",
            nn_player, result_str, half_move, len(game_data),
        )

        return game_data, nn_won

    def play_games(
        self,
        num_games: int,
        curriculum_fns: Optional[List[Tuple[str, object]]] = None,
        curriculum_ratio: float = 0.2,
        curriculum_use_mcts: bool = True,
        target_tier_idx: int = -1,
    ) -> Tuple[List[List[dict]], dict]:
        """Play multiple self-play games, optionally mixing in curriculum games.

        Args:
            num_games: total number of games to play.
            curriculum_fns: list of (name, callable) opponent tiers. If provided,
                games are split: 50% vs target tier, 50% vs lower tiers.
            curriculum_ratio: fraction of curriculum games (default 0.2).
            curriculum_use_mcts: whether to use MCTS in curriculum games.
            target_tier_idx: index into curriculum_fns for the current challenge tier.
                Only wins against this tier count toward promotion.

        Returns:
            Tuple of (game data list, stats dict with curriculum_wins/curriculum_total).
        """
        import random as _rng
        all_games: List[List[dict]] = []
        curriculum_wins = 0   # wins vs target tier only
        curriculum_total = 0  # games vs target tier only

        for i in range(num_games):
            use_curriculum = (
                curriculum_fns is not None
                and _rng.random() < curriculum_ratio
            )

            if use_curriculum:
                # Pick opponent: 50% target tier, 50% random lower tier (for value signal)
                if target_tier_idx > 0 and _rng.random() < 0.5:
                    opp_idx = _rng.randint(0, target_tier_idx - 1)
                    opp_name, opp_fn = curriculum_fns[opp_idx]
                    is_target = False
                else:
                    opp_name, opp_fn = curriculum_fns[target_tier_idx]
                    is_target = True

                logger.info(
                    "Starting curriculum game %d / %d (vs %s, mcts=%s)", i + 1, num_games, opp_name, curriculum_use_mcts
                )
                game_data, nn_won = self.play_curriculum_game(opp_fn, use_mcts=curriculum_use_mcts)
                game_type = f"curriculum-{opp_name}"
                if is_target:
                    curriculum_total += 1
                    if nn_won:
                        curriculum_wins += 1
            else:
                logger.info("Starting self-play game %d / %d", i + 1, num_games)
                game_data = self.play_game()
                game_type = "self-play"

            all_games.append(game_data)
            logger.info(
                "Completed %s game %d / %d: %d positions",
                game_type, i + 1, num_games, len(game_data),
            )

        total_positions = sum(len(g) for g in all_games)
        logger.info(
            "Self-play batch complete: %d games, %d total positions",
            num_games, total_positions,
        )

        stats = {
            "curriculum_wins": curriculum_wins,
            "curriculum_total": curriculum_total,
        }
        return all_games, stats
