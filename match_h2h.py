#!/usr/bin/env python3
"""Head-to-head match: our model vs collaborator's model.

Handles the architecture mismatch:
  - Our model:  12 input planes, 13x13 grid, 6 blocks, 96 channels
  - Theirs:     17 input planes, 18x18 grid, 6 blocks, 128 channels

Each model uses its own feature extraction:
  - Ours: 12-plane extraction from nn.features
  - Theirs: 17-plane hexgo-compatible extraction from nn.compat_features

Usage:
    python match_h2h.py --ours checkpoints/bootstrap_w4_10k.pt \
                        --theirs ~/Downloads/net_gen0162.pt \
                        --games 20 --win-length 4
"""

import argparse
import random
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from game.hex_grid import HexCoord, axial_to_brick, brick_to_axial
from game.rules import GameState
from nn.features import extract_features
from nn.compat_features import extract_compat_features
from nn.model import HexTTTNet
from nn.compat_model import CompatNet, load_compat_model


def raw_policy_move(
    model: torch.nn.Module,
    game_state: GameState,
    grid_size: int,
    in_channels: int,
    device: torch.device,
    zoi_margin: int = 2,
    use_compat_features: bool = False,
) -> HexCoord:
    """Get a move using raw policy (no MCTS). Handles feature adaptation."""
    if use_compat_features:
        features, (cq, cr) = extract_compat_features(game_state, grid_size=grid_size)
    else:
        features, (cq, cr) = extract_features(game_state, grid_size=grid_size)

    # Build valid moves mask
    legal = game_state.legal_moves(zoi_margin=zoi_margin)
    if not legal:
        return HexCoord(0, 0)

    half = grid_size // 2
    mask = torch.zeros(grid_size * grid_size, dtype=torch.float32)
    legal_map = {}  # flat_idx -> HexCoord

    for move in legal:
        bx, by = axial_to_brick(move.q, move.r, cq, cr, grid_size)
        row = by + half
        col = bx + half
        if 0 <= row < grid_size and 0 <= col < grid_size:
            idx = row * grid_size + col
            mask[idx] = 1.0
            legal_map[idx] = move

    if not legal_map:
        # All legal moves are outside the window — pick randomly
        return random.choice(legal)

    # Forward pass
    x = features.unsqueeze(0).to(device)
    m = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x, valid_moves_mask=m)

    policy = out["policy"].squeeze(0).cpu()

    # Pick best legal move
    best_idx = max(legal_map.keys(), key=lambda i: policy[i].item())
    return legal_map[best_idx]


def play_game(
    agent1_fn,
    agent2_fn,
    win_length: int = 4,
    max_moves: int = 200,
    verbose: bool = False,
) -> Tuple[int, int, float]:
    """Play one game. Returns (winner, num_moves, duration_s)."""
    gs = GameState(win_length=win_length)
    move_count = 0
    start = time.time()

    while not gs.is_terminal and move_count < max_moves:
        if gs.current_player == 1:
            move = agent1_fn(gs)
        else:
            move = agent2_fn(gs)

        if verbose:
            p = gs.current_player
            print(f"  Move {move_count+1}: P{p} -> ({move.q}, {move.r})")

        gs = gs.apply_move(move)
        move_count += 1

    winner = gs.winner if gs.winner is not None else 0
    return winner, move_count, time.time() - start


def main():
    parser = argparse.ArgumentParser(description="Head-to-head: our model vs collaborator")
    parser.add_argument("--ours", default="checkpoints/bootstrap_scaled_w4_50k.pt")
    parser.add_argument("--theirs", default="")
    parser.add_argument("--blocks", type=int, default=6, help="Our model blocks")
    parser.add_argument("--channels", type=int, default=96, help="Our model channels")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--win-length", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load our model ---
    print(f"Loading our model: {args.ours}")
    our_net = HexTTTNet(grid_size=13, num_blocks=args.blocks, channels=args.channels, in_channels=12)
    ckpt = torch.load(args.ours, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        our_net.load_state_dict(ckpt["model_state_dict"])
    else:
        our_net.load_state_dict(ckpt)
    our_net.to(device)
    our_net.eval()
    print(f"  Loaded. Params: {sum(p.numel() for p in our_net.parameters()):,}")

    # --- Load their model ---
    print(f"Loading collaborator model: {args.theirs}")
    their_net = load_compat_model(args.theirs, device=device)
    print(f"  Loaded. Params: {sum(p.numel() for p in their_net.parameters()):,}")

    # --- Agent functions ---
    def our_move(gs: GameState) -> HexCoord:
        return raw_policy_move(our_net, gs, grid_size=13, in_channels=12, device=device)

    def their_move(gs: GameState) -> HexCoord:
        return raw_policy_move(their_net, gs, grid_size=18, in_channels=17, device=device, use_compat_features=True)

    # --- Run games ---
    our_wins = 0
    their_wins = 0
    draws = 0

    print(f"\nPlaying {args.games} games (win_length={args.win_length}, alternating sides)...\n")

    for g in range(args.games):
        # Alternate sides
        if g % 2 == 0:
            p1_fn, p2_fn = our_move, their_move
            p1_name, p2_name = "Ours", "Theirs"
        else:
            p1_fn, p2_fn = their_move, our_move
            p1_name, p2_name = "Theirs", "Ours"

        winner, n_moves, dur = play_game(
            p1_fn, p2_fn,
            win_length=args.win_length,
            verbose=args.verbose,
        )

        # Map winner to agent name
        if winner == 1:
            winner_name = p1_name
        elif winner == 2:
            winner_name = p2_name
        else:
            winner_name = "Draw"

        if winner_name == "Ours":
            our_wins += 1
        elif winner_name == "Theirs":
            their_wins += 1
        else:
            draws += 1

        side = "P1" if p1_name == "Ours" else "P2"
        print(f"Game {g+1:2d}/{args.games}: Ours as {side} | Winner: {winner_name:6s} | {n_moves:3d} moves | {dur:.1f}s")

    # --- Results ---
    total = args.games
    print(f"\n{'='*50}")
    print(f"RESULTS ({total} games, win_length={args.win_length})")
    print(f"{'='*50}")
    print(f"  Our model (bootstrap_w4_10k):  {our_wins:2d} wins ({our_wins/total*100:.0f}%)")
    print(f"  Collaborator (net_gen0162):     {their_wins:2d} wins ({their_wins/total*100:.0f}%)")
    print(f"  Draws:                          {draws:2d}")
    print(f"{'='*50}")

    if our_wins > their_wins:
        print(">>> Our model wins the series!")
    elif their_wins > our_wins:
        print(">>> Collaborator's model wins the series!")
    else:
        print(">>> Series tied!")


if __name__ == "__main__":
    main()
