"""Network evaluation via competitive match play.

Evaluates a candidate network against the current best by playing a series
of games where each side uses MCTS.  If the candidate achieves a win rate
above a configurable threshold, it becomes the new best network.

Both players use deterministic settings during evaluation:
    - Temperature = 0 (greedy move selection)
    - No Dirichlet noise at the root
    - Reduced MCTS simulations (optional, configurable)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from game.rules import GameState
from nn.model import HexTTTNet

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates a candidate network against the current best via match play.

    Plays a configurable number of games between two networks, alternating
    who plays first.  Returns the candidate's win rate and whether it passes
    the replacement threshold.
    """

    def __init__(self, config: dict, device: str = "cpu") -> None:
        """Initialize the evaluator.

        Args:
            config: merged configuration dict (should contain ``evaluation``
                and ``mcts`` keys).
            device: torch device string.
        """
        self.config = config
        self.device = device

        eval_cfg = config.get("evaluation", {})
        self.default_num_games: int = eval_cfg.get("games", 200)
        self.win_threshold: float = eval_cfg.get("win_threshold", 0.55)

        mcts_cfg = config.get("mcts", {})
        self.grid_size: int = config.get("network", {}).get("grid_size", 19)
        # Use fewer simulations for evaluation to save compute
        self.eval_simulations: int = mcts_cfg.get(
            "eval_simulations", mcts_cfg.get("num_simulations", 200)
        )

    def evaluate(
        self,
        candidate: HexTTTNet,
        current_best: HexTTTNet,
        num_games: Optional[int] = None,
    ) -> Tuple[float, bool]:
        """Play ``num_games`` between the candidate and current best networks.

        Games alternate which network plays first:
            - Even-indexed games: candidate is Player 1
            - Odd-indexed games: candidate is Player 2

        Both networks use MCTS with evaluation settings (temperature=0,
        no Dirichlet noise).

        Args:
            candidate: the challenger network.
            current_best: the reigning best network.
            num_games: number of games to play. Defaults to config value.

        Returns:
            ``(win_rate, accepted)`` where ``win_rate`` is the candidate's win
            rate (0.0 to 1.0) and ``accepted`` is ``True`` if the candidate
            exceeds the threshold.
        """
        if num_games is None:
            num_games = self.default_num_games

        candidate.eval()
        current_best.eval()
        candidate.to(self.device)
        current_best.to(self.device)

        candidate_wins = 0
        best_wins = 0
        draws = 0

        for game_idx in range(num_games):
            # Alternate first player
            if game_idx % 2 == 0:
                # Candidate is Player 1, current_best is Player 2
                result = self.play_eval_game(candidate, current_best)
                if result == 1:
                    candidate_wins += 1
                elif result == 2:
                    best_wins += 1
                else:
                    draws += 1
            else:
                # Current_best is Player 1, candidate is Player 2
                result = self.play_eval_game(current_best, candidate)
                if result == 1:
                    best_wins += 1
                elif result == 2:
                    candidate_wins += 1
                else:
                    draws += 1

            # Log progress periodically
            if (game_idx + 1) % 20 == 0:
                played = game_idx + 1
                current_wr = candidate_wins / played
                logger.info(
                    "Evaluation progress: %d/%d games | candidate wins=%d, "
                    "best wins=%d, draws=%d | current win rate=%.3f",
                    played, num_games, candidate_wins, best_wins, draws,
                    current_wr,
                )

                # Early termination: if candidate cannot possibly reach threshold
                remaining = num_games - played
                max_possible_wins = candidate_wins + remaining
                max_possible_wr = max_possible_wins / num_games
                if max_possible_wr < self.win_threshold:
                    logger.info(
                        "Early termination: candidate cannot reach threshold "
                        "%.3f (max possible=%.3f).",
                        self.win_threshold, max_possible_wr,
                    )
                    break

                # Early acceptance: if candidate has already secured threshold
                min_possible_wr = candidate_wins / num_games
                if min_possible_wr >= self.win_threshold and played >= num_games // 2:
                    logger.info(
                        "Early acceptance: candidate already secured threshold "
                        "%.3f (current=%.3f with %d/%d played).",
                        self.win_threshold, current_wr, played, num_games,
                    )
                    break

        total_played = candidate_wins + best_wins + draws
        win_rate = candidate_wins / total_played if total_played > 0 else 0.0
        accepted = win_rate >= self.win_threshold

        logger.info(
            "Evaluation complete: %d games played | candidate wins=%d (%.1f%%), "
            "best wins=%d (%.1f%%), draws=%d | accepted=%s",
            total_played,
            candidate_wins, 100.0 * candidate_wins / max(total_played, 1),
            best_wins, 100.0 * best_wins / max(total_played, 1),
            draws,
            accepted,
        )

        return win_rate, accepted

    def play_eval_game(
        self,
        player1_net: HexTTTNet,
        player2_net: HexTTTNet,
    ) -> int:
        """Play a single evaluation game between two networks.

        Both networks use MCTS with evaluation settings:
            - temperature = 0 (deterministic / greedy)
            - No Dirichlet noise at the root

        Args:
            player1_net: network controlling Player 1.
            player2_net: network controlling Player 2.

        Returns:
            ``1`` if Player 1 wins, ``2`` if Player 2 wins, ``0`` for draw.
        """
        from mcts.search import MCTS

        # Build eval-specific MCTS configs (no noise, deterministic)
        eval_mcts_config = dict(self.config.get("mcts", {}))
        eval_mcts_config["dirichlet_epsilon"] = 0.0  # disable noise
        eval_mcts_config["num_simulations"] = self.eval_simulations

        mcts_p1 = MCTS(player1_net, eval_mcts_config)
        mcts_p2 = MCTS(player2_net, eval_mcts_config)

        game_state = GameState()
        half_move = 0

        while not game_state.is_terminal:
            # Select the MCTS instance for the current player
            if game_state.current_player == 1:
                mcts = mcts_p1
            else:
                mcts = mcts_p2

            # Run MCTS with temperature=0 (greedy)
            with torch.no_grad():
                move, _ = mcts.get_move(game_state, temperature=0.0)

            game_state = game_state.apply_move(move)
            half_move += 1

            # Safety limit
            if half_move > 1000:
                logger.warning(
                    "Evaluation game exceeded 1000 half-moves; declaring draw."
                )
                return 0

        winner = game_state.winner
        if winner is None:
            return 0
        return winner
