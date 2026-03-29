#!/usr/bin/env python3
"""Self-play tournament for Infinite Hexagonal Tic-Tac-Toe.

Runs a round-robin or ladder tournament between multiple agents:
  - Neural network checkpoints at different training stages
  - Random player baseline
  - Greedy heuristic baseline
  - One-ply search baseline

Computes Elo ratings, win matrices, and detailed match logs.

Usage:
    # Tournament between checkpoints
    python tournament.py --checkpoints checkpoints/iter_0010.pt checkpoints/iter_0050.pt checkpoints/best.pt \
                         --games-per-pair 20 --simulations 200

    # Include baselines
    python tournament.py --checkpoints checkpoints/best.pt --include-baselines --games-per-pair 40

    # Quick smoke test (random vs random)
    python tournament.py --include-baselines --games-per-pair 10 --no-nn

    # Watch a specific matchup
    python tournament.py --checkpoints checkpoints/best.pt --include-baselines --games-per-pair 4 --verbose
"""

import argparse
import itertools
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import yaml

from game.hex_grid import HexCoord
from game.rules import GameState
from mcts.search import MCTS
from nn.model import HexTTTNet

logger = logging.getLogger(__name__)


# ======================================================================
# Agents
# ======================================================================

@dataclass
class Agent:
    """A tournament participant."""
    name: str
    get_move: Callable[[GameState], HexCoord]
    elo: float = 1000.0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played


class RandomAgent:
    """Places stones uniformly at random within the Zone of Interest."""

    def __init__(self, zoi_margin: int = 3):
        self.zoi_margin = zoi_margin

    def get_move(self, game_state: GameState) -> HexCoord:
        moves = game_state.legal_moves(zoi_margin=self.zoi_margin)
        if not moves:
            return HexCoord(0, 0)
        return random.choice(moves)


class GreedyAgent:
    """Picks the move that maximizes a simple heuristic score.

    Heuristic: for each candidate move, count how many of the current
    player's stones are adjacent (hex neighbors). Breaks ties randomly.
    Prefers moves that extend existing clusters.
    """

    def __init__(self, zoi_margin: int = 3):
        self.zoi_margin = zoi_margin

    def get_move(self, game_state: GameState) -> HexCoord:
        from game.hex_grid import hex_neighbors, HEX_AXES

        moves = game_state.legal_moves(zoi_margin=self.zoi_margin)
        if not moves:
            return HexCoord(0, 0)

        player = game_state.current_player
        board = game_state.board

        best_score = -1
        best_moves = []

        for move in moves:
            score = 0
            # Count adjacent friendly stones
            for nb in hex_neighbors(move):
                if board.stones.get(nb) == player:
                    score += 1

            # Bonus: count how long a line this move would create along each axis
            for axis in HEX_AXES:
                line_len = 1
                # Forward
                pos = HexCoord(move.q + axis.q, move.r + axis.r)
                while board.stones.get(pos) == player:
                    line_len += 1
                    pos = HexCoord(pos.q + axis.q, pos.r + axis.r)
                # Backward
                pos = HexCoord(move.q - axis.q, move.r - axis.r)
                while board.stones.get(pos) == player:
                    line_len += 1
                    pos = HexCoord(pos.q - axis.q, pos.r - axis.r)
                score += line_len * 2  # Weight line extension heavily

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)


class OnePlyAgent:
    """Looks one move ahead: checks if any move wins immediately or blocks
    an opponent's immediate win. Falls back to greedy otherwise."""

    def __init__(self, win_length: int = 6, zoi_margin: int = 3):
        self.win_length = win_length
        self.zoi_margin = zoi_margin
        self.greedy = GreedyAgent(zoi_margin)

    def get_move(self, game_state: GameState) -> HexCoord:
        moves = game_state.legal_moves(zoi_margin=self.zoi_margin)
        if not moves:
            return HexCoord(0, 0)

        player = game_state.current_player
        opponent = 3 - player
        board = game_state.board

        # Check for immediate winning move
        for move in moves:
            new_board = board.place(move, player)
            if new_board.check_win(move, self.win_length) == player:
                return move

        # Check for blocking move (opponent would win if they played here)
        for move in moves:
            new_board = board.place(move, opponent)
            if new_board.check_win(move, self.win_length) == opponent:
                return move

        # No immediate win/block — fall back to greedy
        return self.greedy.get_move(game_state)


class EisensteinGreedyAgent:
    """Zero-parameter geometric agent grounded in hex lattice structure.

    Inspired by hexgo's EisensteinGreedyAgent. Scores each candidate move
    by the maximum chain length it would create (or block) along any of the
    three Z[omega] axes (the Eisenstein integer unit directions).  With
    ``defensive=True``, also considers blocking the opponent's best chain.

    This provides a stronger-than-greedy baseline that forces NN agents to
    learn real tactical patterns rather than just beating random play.
    """

    def __init__(self, win_length: int = 6, zoi_margin: int = 3, defensive: bool = True):
        self.win_length = win_length
        self.zoi_margin = zoi_margin
        self.defensive = defensive

    def _chain_score(self, board_stones: dict, coord: HexCoord, player: int) -> int:
        """Score a move by the longest chain it creates along any hex axis."""
        from game.hex_grid import HEX_AXES
        best_chain = 0
        for axis in HEX_AXES:
            chain = 1
            # Forward
            pos = HexCoord(coord.q + axis.q, coord.r + axis.r)
            while board_stones.get(pos) == player:
                chain += 1
                pos = HexCoord(pos.q + axis.q, pos.r + axis.r)
            # Backward
            pos = HexCoord(coord.q - axis.q, coord.r - axis.r)
            while board_stones.get(pos) == player:
                chain += 1
                pos = HexCoord(pos.q - axis.q, pos.r - axis.r)
            best_chain = max(best_chain, chain)
        return best_chain

    def get_move(self, game_state: GameState) -> HexCoord:
        moves = game_state.legal_moves(zoi_margin=self.zoi_margin)
        if not moves:
            return HexCoord(0, 0)

        player = game_state.current_player
        opponent = 3 - player
        board = game_state.board

        best_score = -1
        best_moves = []

        for move in moves:
            # Offensive: chain we'd create
            attack = self._chain_score(board.stones, move, player)
            score = attack * 3  # weight offensive moves

            if self.defensive:
                # Defensive: chain opponent would create here
                defense = self._chain_score(board.stones, move, opponent)
                score += defense * 2  # slightly less weight on defense

            # Bonus for near-win chains
            if attack >= self.win_length:
                score += 1000  # winning move
            if self.defensive and self._chain_score(board.stones, move, opponent) >= self.win_length:
                score += 500  # must-block

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)


class MCTSAgent:
    """Neural network agent using MCTS."""

    def __init__(self, network: HexTTTNet, config: dict, name: str = "mcts"):
        self.network = network
        self.config = config
        self.name = name
        self.mcts = MCTS(network, config)

    def get_move(self, game_state: GameState) -> HexCoord:
        with torch.no_grad():
            move, _, _ = self.mcts.get_move(game_state, temperature=0.0)
        return move


# ======================================================================
# Elo computation
# ======================================================================

def expected_score(elo_a: float, elo_b: float) -> float:
    """Expected score for player A given Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_elo(
    elo_a: float, elo_b: float, score_a: float, k: float = 32.0
) -> Tuple[float, float]:
    """Update Elo ratings after a game.

    Args:
        elo_a: Player A's current Elo.
        elo_b: Player B's current Elo.
        score_a: Actual score for A (1.0 = win, 0.5 = draw, 0.0 = loss).
        k: K-factor (higher = more volatile ratings).

    Returns:
        (new_elo_a, new_elo_b)
    """
    ea = expected_score(elo_a, elo_b)
    eb = 1.0 - ea
    score_b = 1.0 - score_a
    new_a = elo_a + k * (score_a - ea)
    new_b = elo_b + k * (score_b - eb)
    return new_a, new_b


# ======================================================================
# Match play
# ======================================================================

@dataclass
class MatchResult:
    """Result of a single game."""
    player1_name: str
    player2_name: str
    winner: int  # 1, 2, or 0 for draw
    num_moves: int
    duration_s: float
    move_history: List[Tuple[int, int]] = field(default_factory=list)  # [(q, r), ...]


def play_match(
    agent1: Agent,
    agent2: Agent,
    win_length: int = 6,
    max_moves: int = 1000,
    verbose: bool = False,
) -> MatchResult:
    """Play a single game between two agents.

    agent1 plays as Player 1, agent2 as Player 2.

    Returns:
        MatchResult with outcome details.
    """
    game_state = GameState()
    half_move = 0
    start = time.time()

    while not game_state.is_terminal and half_move < max_moves:
        if game_state.current_player == 1:
            move = agent1.get_move(game_state)
        else:
            move = agent2.get_move(game_state)

        if verbose:
            player_name = agent1.name if game_state.current_player == 1 else agent2.name
            print(
                f"  Move {half_move + 1}: {player_name} (P{game_state.current_player}) "
                f"plays ({move.q}, {move.r}) "
                f"[sub-move {3 - game_state.moves_remaining}/{'1' if game_state.is_first_turn else '2'}]"
            )

        game_state = game_state.apply_move(move)
        half_move += 1

    duration = time.time() - start
    winner = game_state.winner if game_state.winner is not None else 0

    # Extract move history as plain tuples
    move_history = [(c.q, c.r) for c in game_state.move_history]

    return MatchResult(
        player1_name=agent1.name,
        player2_name=agent2.name,
        winner=winner,
        num_moves=half_move,
        duration_s=duration,
        move_history=move_history,
    )


# ======================================================================
# Tournament
# ======================================================================

@dataclass
class TournamentResult:
    """Aggregated tournament results."""
    agents: List[Agent]
    matches: List[MatchResult]
    win_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def compute_win_matrix(self):
        """Build win/loss/draw matrix."""
        names = [a.name for a in self.agents]
        self.win_matrix = {n: {m: {"w": 0, "l": 0, "d": 0} for m in names if m != n} for n in names}
        for match in self.matches:
            p1, p2 = match.player1_name, match.player2_name
            if match.winner == 1:
                self.win_matrix[p1][p2]["w"] += 1
                self.win_matrix[p2][p1]["l"] += 1
            elif match.winner == 2:
                self.win_matrix[p1][p2]["l"] += 1
                self.win_matrix[p2][p1]["w"] += 1
            else:
                self.win_matrix[p1][p2]["d"] += 1
                self.win_matrix[p2][p1]["d"] += 1


def run_tournament(
    agents: List[Agent],
    games_per_pair: int = 10,
    win_length: int = 6,
    verbose: bool = False,
) -> TournamentResult:
    """Run a round-robin tournament.

    Each pair plays `games_per_pair` games, alternating who goes first.

    Args:
        agents: list of Agent instances.
        games_per_pair: games per directional matchup (total games between
            a pair = games_per_pair, with sides alternated).
        win_length: win condition length.
        verbose: print individual moves.

    Returns:
        TournamentResult with all match details and computed Elo ratings.
    """
    all_matches: List[MatchResult] = []
    pairs = list(itertools.combinations(range(len(agents)), 2))
    total_games = len(pairs) * games_per_pair

    logger.info(
        "Tournament: %d agents, %d pairs, %d games/pair, %d total games",
        len(agents), len(pairs), games_per_pair, total_games,
    )

    game_num = 0
    for i, j in pairs:
        a1, a2 = agents[i], agents[j]
        pair_header = f"{a1.name} vs {a2.name}"
        logger.info("--- %s (%d games) ---", pair_header, games_per_pair)

        pair_results = {"p1_wins": 0, "p2_wins": 0, "draws": 0}

        for g in range(games_per_pair):
            game_num += 1

            # Alternate sides
            if g % 2 == 0:
                first, second = a1, a2
            else:
                first, second = a2, a1

            if verbose:
                print(f"\nGame {game_num}/{total_games}: {first.name} (P1) vs {second.name} (P2)")

            result = play_match(first, second, win_length=win_length, verbose=verbose)
            all_matches.append(result)

            # Map result back to a1/a2 regardless of side
            if result.winner == 0:
                pair_results["draws"] += 1
                a1.draws += 1
                a2.draws += 1
                score_first = 0.5
            elif result.winner == 1:
                # first player won
                if first is a1:
                    a1.wins += 1
                    a2.losses += 1
                    pair_results["p1_wins"] += 1
                else:
                    a2.wins += 1
                    a1.losses += 1
                    pair_results["p2_wins"] += 1
                score_first = 1.0
            else:
                # second player won
                if second is a1:
                    a1.wins += 1
                    a2.losses += 1
                    pair_results["p1_wins"] += 1
                else:
                    a2.wins += 1
                    a1.losses += 1
                    pair_results["p2_wins"] += 1
                score_first = 0.0

            # Update Elo
            first.elo, second.elo = update_elo(first.elo, second.elo, score_first)

            if (game_num % 5 == 0) or verbose:
                logger.info(
                    "Game %d/%d: %s (P1) vs %s (P2) -> %s in %d moves (%.1fs)",
                    game_num, total_games,
                    first.name, second.name,
                    "P1 win" if result.winner == 1 else ("P2 win" if result.winner == 2 else "draw"),
                    result.num_moves, result.duration_s,
                )

        logger.info(
            "  %s result: %s wins %d, %s wins %d, draws %d",
            pair_header, a1.name, pair_results["p1_wins"],
            a2.name, pair_results["p2_wins"], pair_results["draws"],
        )

    # Build tournament result
    tournament = TournamentResult(agents=agents, matches=all_matches)
    tournament.compute_win_matrix()

    return tournament


# ======================================================================
# Display
# ======================================================================

def print_standings(tournament: TournamentResult):
    """Print Elo standings table."""
    agents = sorted(tournament.agents, key=lambda a: a.elo, reverse=True)

    print("\n" + "=" * 70)
    print("TOURNAMENT STANDINGS")
    print("=" * 70)
    print(f"{'Rank':<6}{'Agent':<25}{'Elo':<10}{'W':<6}{'L':<6}{'D':<6}{'Win%':<8}{'Games':<6}")
    print("-" * 70)

    for rank, agent in enumerate(agents, 1):
        print(
            f"{rank:<6}{agent.name:<25}{agent.elo:<10.1f}"
            f"{agent.wins:<6}{agent.losses:<6}{agent.draws:<6}"
            f"{agent.win_rate * 100:<8.1f}{agent.games_played:<6}"
        )
    print("=" * 70)


def print_win_matrix(tournament: TournamentResult):
    """Print head-to-head win matrix."""
    agents = sorted(tournament.agents, key=lambda a: a.elo, reverse=True)
    names = [a.name for a in agents]

    # Truncate names for display
    short = {n: n[:12] for n in names}

    print("\nHEAD-TO-HEAD (W-L-D)")
    header = f"{'':>14}" + "".join(f"{short[n]:>14}" for n in names)
    print(header)
    print("-" * (14 + 14 * len(names)))

    for n1 in names:
        row = f"{short[n1]:>14}"
        for n2 in names:
            if n1 == n2:
                row += f"{'---':>14}"
            else:
                m = tournament.win_matrix[n1][n2]
                row += f"{m['w']}-{m['l']}-{m['d']}".rjust(14)
        print(row)
    print()


def save_replay(match: MatchResult, output_path: str, game_rules: dict = None):
    """Save a single game replay as JSON."""
    replay = {
        "player1": match.player1_name,
        "player2": match.player2_name,
        "winner": match.winner,
        "num_moves": match.num_moves,
        "move_history": match.move_history,
        "rules": game_rules or {"win_length": 6, "first_turn_stones": 1, "normal_turn_stones": 2},
    }
    with open(output_path, "w") as f:
        json.dump(replay, f, indent=2)
    logger.info("Replay saved to %s", output_path)


def save_results(tournament: TournamentResult, output_path: str):
    """Save tournament results as JSON."""
    data = {
        "standings": [
            {
                "name": a.name,
                "elo": round(a.elo, 1),
                "wins": a.wins,
                "losses": a.losses,
                "draws": a.draws,
                "win_rate": round(a.win_rate, 4),
            }
            for a in sorted(tournament.agents, key=lambda a: a.elo, reverse=True)
        ],
        "matches": [
            {
                "player1": m.player1_name,
                "player2": m.player2_name,
                "winner": m.winner,
                "num_moves": m.num_moves,
                "duration_s": round(m.duration_s, 2),
                "move_history": m.move_history,
            }
            for m in tournament.matches
        ],
        "win_matrix": tournament.win_matrix,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", output_path)


# ======================================================================
# Agent loading
# ======================================================================

def load_nn_agent(
    checkpoint_path: str,
    config: dict,
    device: str,
    simulations: int,
    name: Optional[str] = None,
) -> Agent:
    """Load a neural network agent from a checkpoint.

    Handles both full trainer checkpoints (with 'model_state_dict' key)
    and bare state_dict files.
    """
    net_cfg = config.get("network", {})
    network = HexTTTNet(
        grid_size=net_cfg.get("grid_size", 19),
        num_blocks=net_cfg.get("num_blocks", 8),
        channels=net_cfg.get("channels", 128),
        in_channels=net_cfg.get("in_channels", 12),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        network.load_state_dict(checkpoint["model_state_dict"])
        iteration = checkpoint.get("extra", {}).get("iteration", "?")
        if name is None:
            name = f"nn_iter{iteration}"
    else:
        network.load_state_dict(checkpoint)
        if name is None:
            name = Path(checkpoint_path).stem

    network.to(device)
    network.eval()

    mcts_config = dict(config.get("mcts", {}))
    mcts_config["num_simulations"] = simulations
    mcts_config["dirichlet_epsilon"] = 0.0  # No noise in tournament play
    mcts_config["device"] = device
    mcts_config["grid_size"] = net_cfg.get("grid_size", 19)

    mcts_agent = MCTSAgent(network, mcts_config, name=name)

    return Agent(name=name, get_move=mcts_agent.get_move)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Self-play tournament for Infinite Hex Tic-Tac-Toe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baselines only (no GPU needed)
  python tournament.py --include-baselines --no-nn --games-per-pair 20

  # One checkpoint vs baselines
  python tournament.py --checkpoints checkpoints/best.pt --include-baselines

  # Multiple checkpoints head-to-head
  python tournament.py --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt --games-per-pair 30
        """,
    )
    parser.add_argument(
        "--checkpoints", nargs="*", default=[],
        help="Paths to model checkpoint files",
    )
    parser.add_argument(
        "--config", type=str, default="configs/phase1.yaml",
        help="Config YAML (for network architecture)",
    )
    parser.add_argument(
        "--games-per-pair", type=int, default=10,
        help="Games per pair (alternating sides)",
    )
    parser.add_argument(
        "--simulations", type=int, default=200,
        help="MCTS simulations per move for NN agents",
    )
    parser.add_argument(
        "--include-baselines", action="store_true",
        help="Include Random, Greedy, and OnePly baseline agents",
    )
    parser.add_argument(
        "--no-nn", action="store_true",
        help="Skip loading neural network agents (baselines only)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (cpu/cuda/mps/auto)",
    )
    parser.add_argument(
        "--output", type=str, default="tournament_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print individual moves during games",
    )
    parser.add_argument(
        "--win-length", type=int, default=6,
        help="Win condition (consecutive stones)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info("Device: %s", device)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Config not found: %s — using defaults", config_path)
        config = {"network": {"grid_size": 19, "num_blocks": 8, "channels": 128, "in_channels": 12}}

    win_length = args.win_length
    zoi_margin = config.get("mcts", {}).get("zoi_margin", 3)

    # Build agent list
    agents: List[Agent] = []

    # Baseline agents
    if args.include_baselines or args.no_nn:
        random_agent = RandomAgent(zoi_margin=zoi_margin)
        agents.append(Agent(name="Random", get_move=random_agent.get_move))

        greedy_agent = GreedyAgent(zoi_margin=zoi_margin)
        agents.append(Agent(name="Greedy", get_move=greedy_agent.get_move))

        oneply_agent = OnePlyAgent(win_length=win_length, zoi_margin=zoi_margin)
        agents.append(Agent(name="OnePly", get_move=oneply_agent.get_move))

        eisenstein_agent = EisensteinGreedyAgent(
            win_length=win_length, zoi_margin=zoi_margin, defensive=True
        )
        agents.append(Agent(name="Eisenstein", get_move=eisenstein_agent.get_move))

        logger.info("Added baseline agents: Random, Greedy, OnePly, Eisenstein")

    # Neural network agents
    if not args.no_nn:
        for ckpt_path in args.checkpoints:
            if not Path(ckpt_path).exists():
                logger.warning("Checkpoint not found, skipping: %s", ckpt_path)
                continue
            agent = load_nn_agent(
                ckpt_path, config, device, args.simulations,
            )
            agents.append(agent)
            logger.info("Loaded NN agent: %s (from %s, %d sims)", agent.name, ckpt_path, args.simulations)

    if len(agents) < 2:
        logger.error(
            "Need at least 2 agents for a tournament. Got %d. "
            "Use --include-baselines and/or --checkpoints.",
            len(agents),
        )
        raise SystemExit(1)

    # Run tournament
    logger.info("Starting tournament with %d agents...", len(agents))
    start = time.time()

    tournament = run_tournament(
        agents=agents,
        games_per_pair=args.games_per_pair,
        win_length=win_length,
        verbose=args.verbose,
    )

    elapsed = time.time() - start

    # Display results
    print_standings(tournament)
    print_win_matrix(tournament)

    total_games = len(tournament.matches)
    print(f"\nTotal: {total_games} games in {elapsed:.1f}s ({elapsed / max(total_games, 1):.2f}s/game)")

    # Save results
    save_results(tournament, args.output)

    # Save first and last game replays
    if tournament.matches:
        game_rules = {"win_length": win_length, "first_turn_stones": 1, "normal_turn_stones": 2}
        save_replay(tournament.matches[0], "replay_first.json", game_rules)
        save_replay(tournament.matches[-1], "replay_last.json", game_rules)

    return tournament


if __name__ == "__main__":
    main()
