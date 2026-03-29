#!/usr/bin/env python3
"""Play against the trained Hex TTT AI, or watch AI vs AI.

Usage:
    python play.py --checkpoint checkpoints/best.pt --mode human-vs-ai
    python play.py --checkpoint checkpoints/best.pt --mode ai-vs-ai
    python play.py --mode human-vs-human
"""

import argparse
import sys
from typing import Dict, Optional, Tuple

import torch
import yaml

from game.hex_grid import HexCoord, hex_distance
from game.rules import GameState
from mcts.search import MCTS
from nn.features import extract_features
from nn.model import HexTTTNet


# ======================================================================
# Board rendering
# ======================================================================

def render_board(game_state: GameState, grid_size: int = 15) -> str:
    """Render the hex board as ASCII art with offset rows.

    Uses characters:
        *  for Player 1 stones
        o  for Player 2 stones
        .  for empty cells within the display window

    Odd rows are offset by one space to approximate the hex layout.
    Coordinate labels are shown on the edges.

    Args:
        game_state: current game state to render.
        grid_size: display window size (number of rows/columns shown).

    Returns:
        A multi-line string of the rendered board.
    """
    board = game_state.board
    stones = board.stones

    # Determine the centre of the display window
    if stones:
        cq_f, cr_f = board.stone_centroid()
        center_q = int(round(cq_f))
        center_r = int(round(cr_f))
    else:
        center_q, center_r = 0, 0

    half = grid_size // 2
    lines = []

    # Header: column indices (axial q relative to centre)
    header_indent = "     "  # space for row label + offset
    col_labels = []
    for col in range(grid_size):
        r_rel = 0
        q_rel = (col - half) - r_rel // 2
        col_labels.append(f"{q_rel:>2}")
    lines.append(header_indent + " " + " ".join(col_labels))
    lines.append("")

    # Mark the last move for highlighting
    last_move = game_state.move_history[-1] if game_state.move_history else None

    for row in range(grid_size):
        r_rel = row - half
        r_ax = r_rel + center_r

        # Row label (axial r)
        label = f"{r_ax:>3}  "

        # Offset odd rows to create hex staggering
        indent = " " if (r_rel % 2 != 0) else ""

        cells = []
        for col in range(grid_size):
            q_rel = (col - half) - r_rel // 2
            q_ax = q_rel + center_q
            coord = HexCoord(q_ax, r_ax)

            if coord in stones:
                player = stones[coord]
                if coord == last_move:
                    # Highlight last move with brackets
                    cells.append("[*]" if player == 1 else "[o]")
                else:
                    cells.append(" * " if player == 1 else " o ")
            else:
                cells.append(" . ")
        lines.append(label + indent + "".join(cells))

    # Footer: legend
    lines.append("")
    lines.append(f"  Player 1: *    Player 2: o    Last move: [*] or [o]")
    lines.append(
        f"  Turn {game_state.turn_number}, "
        f"Player {game_state.current_player}, "
        f"sub-moves remaining: {game_state.moves_remaining}"
    )
    if game_state.is_terminal:
        if game_state.winner is not None:
            lines.append(f"  *** Player {game_state.winner} wins! ***")
        else:
            lines.append("  *** Draw ***")

    return "\n".join(lines)


def parse_move(user_input: str) -> Optional[HexCoord]:
    """Parse user input into a HexCoord.

    Accepts formats: "q r" or "q,r".

    Returns:
        A HexCoord, or None if parsing fails.
    """
    user_input = user_input.strip()
    if not user_input:
        return None

    # Try comma-separated
    if "," in user_input:
        parts = user_input.split(",")
    else:
        parts = user_input.split()

    if len(parts) != 2:
        return None

    try:
        q = int(parts[0].strip())
        r = int(parts[1].strip())
        return HexCoord(q, r)
    except ValueError:
        return None


# ======================================================================
# AI move selection
# ======================================================================

def ai_select_move(
    game_state: GameState,
    network: HexTTTNet,
    config: dict,
    num_simulations: int,
    device: str,
) -> Tuple[HexCoord, Dict[HexCoord, float]]:
    """Use MCTS to select a move for the AI player.

    Args:
        game_state: current position.
        network: trained HexTTTNet.
        config: MCTS config dict.
        num_simulations: number of MCTS simulations.
        device: torch device string.

    Returns:
        (selected_move, policy_dict)
    """
    mcts_config = dict(config.get("mcts", {}))
    mcts_config["num_simulations"] = num_simulations
    mcts_config["device"] = device
    mcts_config["grid_size"] = config.get("network", {}).get("grid_size", 19)
    # No noise for play (deterministic evaluation)
    mcts_config["dirichlet_epsilon"] = 0.0

    mcts = MCTS(network, mcts_config)
    network.eval()

    with torch.no_grad():
        move, policy = mcts.get_move(game_state, temperature=0.0)

    return move, policy


def show_top_moves(policy: Dict[HexCoord, float], top_k: int = 5) -> None:
    """Print the top-k moves by MCTS visit probability."""
    sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)[:top_k]
    parts = [f"({m.q},{m.r}): {p:.1%}" for m, p in sorted_moves]
    print(f"  AI top moves: {', '.join(parts)}")


# ======================================================================
# Game modes
# ======================================================================

def play_human_vs_ai(
    network: HexTTTNet,
    config: dict,
    num_simulations: int,
    device: str,
) -> None:
    """Human vs AI interactive game."""
    # Let the human choose their color
    print("\nChoose your color:")
    print("  1 = Play as Player 1 (first move)")
    print("  2 = Play as Player 2")
    choice = input("Enter 1 or 2 [default: 1]: ").strip()
    human_player = 2 if choice == "2" else 1
    ai_player = 3 - human_player

    print(f"\nYou are Player {human_player} ({'*' if human_player == 1 else 'o'})")
    print(f"AI is Player {ai_player} ({'*' if ai_player == 1 else 'o'})")
    print("Enter moves as: q r   (axial coordinates)")
    print("Type 'quit' to exit, 'undo' to take back last move.\n")

    game_state = GameState()
    history_stack = [game_state]

    while not game_state.is_terminal:
        print(render_board(game_state))
        print()

        if game_state.current_player == human_player:
            # Human turn
            while True:
                prompt = f"Player {human_player}, your move (q r): "
                user_input = input(prompt).strip()

                if user_input.lower() == "quit":
                    print("Game abandoned.")
                    return

                if user_input.lower() == "undo":
                    # Undo back to the human's previous position
                    if len(history_stack) > 1:
                        history_stack.pop()
                        game_state = history_stack[-1]
                        print("Move undone.")
                        print(render_board(game_state))
                    else:
                        print("Nothing to undo.")
                    continue

                coord = parse_move(user_input)
                if coord is None:
                    print("Invalid format. Enter as: q r  (e.g., 0 0)")
                    continue

                # Check if the move is legal
                legal = game_state.legal_moves()
                if coord not in legal:
                    if game_state.board.is_occupied(coord):
                        print(f"Cell ({coord.q}, {coord.r}) is already occupied.")
                    else:
                        print(f"Cell ({coord.q}, {coord.r}) is outside the zone of interest.")
                    continue

                # Apply the move
                game_state = game_state.apply_move(coord)
                history_stack.append(game_state)
                break

        else:
            # AI turn
            print(f"AI (Player {ai_player}) is thinking ({num_simulations} simulations)...")
            move, policy = ai_select_move(game_state, network, config, num_simulations, device)
            show_top_moves(policy)
            print(f"  AI plays: ({move.q}, {move.r})")
            game_state = game_state.apply_move(move)
            history_stack.append(game_state)

    # Game over
    print(render_board(game_state))
    if game_state.winner == human_player:
        print("\nCongratulations, you win!")
    elif game_state.winner == ai_player:
        print("\nAI wins. Better luck next time!")
    else:
        print("\nIt's a draw!")


def play_ai_vs_ai(
    network: HexTTTNet,
    config: dict,
    num_simulations: int,
    device: str,
) -> None:
    """Watch two AI instances play against each other."""
    print("\nAI vs AI mode")
    print(f"Simulations per move: {num_simulations}")
    print("Press Enter to advance each move, or type 'quit' to stop.\n")

    game_state = GameState()
    half_move = 0

    while not game_state.is_terminal:
        print(render_board(game_state))
        print()

        print(f"Player {game_state.current_player} thinking ({num_simulations} sims)...")
        move, policy = ai_select_move(game_state, network, config, num_simulations, device)
        show_top_moves(policy)
        print(f"  Player {game_state.current_player} plays: ({move.q}, {move.r})")

        game_state = game_state.apply_move(move)
        half_move += 1

        # Prompt user to continue
        user_input = input("\n[Enter to continue, 'quit' to stop]: ").strip()
        if user_input.lower() == "quit":
            print("Game stopped.")
            return

        if half_move > 500:
            print("Game exceeded 500 half-moves. Stopping.")
            break

    print(render_board(game_state))
    if game_state.winner is not None:
        print(f"\nPlayer {game_state.winner} wins after {half_move} half-moves!")
    else:
        print(f"\nDraw after {half_move} half-moves.")


def play_human_vs_human() -> None:
    """Two humans playing on the same terminal."""
    print("\nHuman vs Human mode")
    print("Enter moves as: q r   (axial coordinates)")
    print("Type 'quit' to exit, 'undo' to take back.\n")

    game_state = GameState()
    history_stack = [game_state]

    while not game_state.is_terminal:
        print(render_board(game_state))
        print()

        while True:
            prompt = f"Player {game_state.current_player}, your move (q r): "
            user_input = input(prompt).strip()

            if user_input.lower() == "quit":
                print("Game abandoned.")
                return

            if user_input.lower() == "undo":
                if len(history_stack) > 1:
                    history_stack.pop()
                    game_state = history_stack[-1]
                    print("Move undone.")
                    print(render_board(game_state))
                else:
                    print("Nothing to undo.")
                continue

            coord = parse_move(user_input)
            if coord is None:
                print("Invalid format. Enter as: q r  (e.g., 0 0)")
                continue

            legal = game_state.legal_moves()
            if coord not in legal:
                if game_state.board.is_occupied(coord):
                    print(f"Cell ({coord.q}, {coord.r}) is already occupied.")
                else:
                    print(f"Cell ({coord.q}, {coord.r}) is outside the zone of interest.")
                continue

            game_state = game_state.apply_move(coord)
            history_stack.append(game_state)
            break

    print(render_board(game_state))
    if game_state.winner is not None:
        print(f"\nPlayer {game_state.winner} wins!")
    else:
        print("\nIt's a draw!")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Play Infinite Hex Tic-Tac-Toe")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Config YAML")
    parser.add_argument(
        "--mode",
        choices=["human-vs-ai", "ai-vs-ai", "human-vs-human"],
        default="human-vs-ai",
        help="Game mode",
    )
    parser.add_argument("--simulations", type=int, default=400, help="MCTS simulations per move")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/mps/auto)")
    args = parser.parse_args()

    # Detect device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Inject device and grid_size into mcts config
    if "mcts" not in config:
        config["mcts"] = {}
    config["mcts"]["device"] = device
    config["mcts"]["grid_size"] = config.get("network", {}).get("grid_size", 19)

    print("=" * 50)
    print("  Infinite Hexagonal Tic-Tac-Toe")
    print("=" * 50)
    print(f"  Mode: {args.mode}")
    print(f"  Device: {device}")

    # Load network if needed for AI modes
    network = None
    if args.mode in ("human-vs-ai", "ai-vs-ai"):
        if args.checkpoint is None:
            print("\nNo checkpoint provided. Using randomly initialized network.")
            print("(Train a model first with: python train.py --config configs/phase1.yaml)\n")

        net_cfg = config.get("network", {})
        network = HexTTTNet(
            grid_size=net_cfg.get("grid_size", 19),
            num_blocks=net_cfg.get("num_blocks", 8),
            channels=net_cfg.get("channels", 128),
            in_channels=net_cfg.get("in_channels", 12),
        )

        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            # Support both full checkpoint (with model_state_dict key) and bare state dict
            if "model_state_dict" in checkpoint:
                network.load_state_dict(checkpoint["model_state_dict"])
                print(f"  Loaded checkpoint: {args.checkpoint}")
                step = checkpoint.get("global_step", "?")
                print(f"  Training step: {step}")
            else:
                network.load_state_dict(checkpoint)
                print(f"  Loaded model weights: {args.checkpoint}")

        network.to(device)
        network.eval()
        print(f"  Network: {net_cfg.get('num_blocks', 8)} blocks, "
              f"{net_cfg.get('channels', 128)} channels, "
              f"{net_cfg.get('grid_size', 19)}x{net_cfg.get('grid_size', 19)} grid")
        print(f"  Simulations: {args.simulations}")

    print("=" * 50)

    # Dispatch to game mode
    if args.mode == "human-vs-ai":
        play_human_vs_ai(network, config, args.simulations, device)
    elif args.mode == "ai-vs-ai":
        play_ai_vs_ai(network, config, args.simulations, device)
    elif args.mode == "human-vs-human":
        play_human_vs_human()


if __name__ == "__main__":
    main()
