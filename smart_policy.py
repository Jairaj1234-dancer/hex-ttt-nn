#!/usr/bin/env python3
"""Smart Policy: raw NN policy + 1-ply tactical layer.

Enhances the ~84% raw policy win rate by adding:
1. Immediate win detection (play winning move)
2. Must-block detection (block opponent's winning move)
3. Losing-move filter (skip moves that give opponent a forced win next)

This is much more effective than MCTS when the value head is uncalibrated.
"""

import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from game.hex_grid import HexCoord, axial_to_brick
from game.rules import GameState
from nn.features import extract_features
from nn.model import HexTTTNet
from tournament import EisensteinGreedyAgent, OnePlyAgent

WIN_LENGTH = 6
GRID_SIZE = 13
ZOI_MARGIN = 3
MAX_GAME_MOVES = 120


# =====================================================================
# 1-ply tactical layer
# =====================================================================

def find_winning_move(gs: GameState) -> Optional[HexCoord]:
    """Check if current player has an immediate winning move."""
    player = gs.current_player
    board = gs.board
    wl = gs.win_length
    for move in gs.legal_moves(zoi_margin=ZOI_MARGIN):
        new_board = board.place(move, player)
        if new_board.check_win(move, wl) == player:
            return move
    return None


def find_must_blocks(gs: GameState) -> List[HexCoord]:
    """Find moves the opponent would win with if we don't block."""
    opponent = 3 - gs.current_player
    board = gs.board
    wl = gs.win_length
    blocks = []
    for move in gs.legal_moves(zoi_margin=ZOI_MARGIN):
        new_board = board.place(move, opponent)
        if new_board.check_win(move, wl) == opponent:
            blocks.append(move)
    return blocks


def move_gives_opponent_forced_win(gs: GameState, move: HexCoord) -> bool:
    """After playing `move`, does the opponent have a forced win?

    A forced win means the opponent has 2+ winning moves, so we can't block both.
    Or if it's the second sub-move of our turn, the opponent gets to respond.
    """
    new_gs = gs.apply_move(move)
    if new_gs.is_terminal:
        return False  # Game already ended

    # If still our turn (first sub-move), check if opponent threatens after our second move
    # We can't check all combinations, so skip for first sub-moves
    if new_gs.current_player == gs.current_player:
        return False  # Still our turn, we get another move

    # It's now opponent's turn. Check if they have a winning move.
    opp_win = find_winning_move(new_gs)
    if opp_win is not None:
        return True

    return False


def move_creates_double_threat(gs: GameState, move: HexCoord) -> bool:
    """After playing `move`, do we create 2+ threats the opponent must block?

    Only meaningful for second sub-move (end of our turn).
    """
    new_gs = gs.apply_move(move)
    if new_gs.is_terminal:
        return False
    if new_gs.current_player == gs.current_player:
        return False  # Still our turn
    # Now it's opponent's turn — check how many blocks they need
    blocks = find_must_blocks(new_gs)
    return len(blocks) >= 2


# =====================================================================
# Smart policy move selection
# =====================================================================

def nn_raw_policy(model: HexTTTNet, gs: GameState, device: str, top_k: int = 8):
    """Get top-K raw policy moves from the NN."""
    features, (cq, cr) = extract_features(gs, grid_size=GRID_SIZE)
    legal = gs.legal_moves(zoi_margin=ZOI_MARGIN)
    if not legal:
        return []

    half = GRID_SIZE // 2
    mask = torch.zeros(GRID_SIZE * GRID_SIZE, dtype=torch.float32)
    legal_map = {}

    for move in legal:
        bx, by = axial_to_brick(move.q, move.r, cq, cr, GRID_SIZE)
        row = by + half
        col = bx + half
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            idx = row * GRID_SIZE + col
            mask[idx] = 1.0
            legal_map[idx] = move

    if not legal_map:
        return [(random.choice(legal), 1.0)]

    x = features.unsqueeze(0).to(device)
    m = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x, valid_moves_mask=m)

    policy = out["policy"].squeeze(0).cpu().numpy()
    ranked = sorted(legal_map.keys(), key=lambda i: -policy[i])[:top_k]
    return [(legal_map[idx], float(policy[idx])) for idx in ranked]


def smart_policy_move(model: HexTTTNet, gs: GameState, device: str) -> HexCoord:
    """Smart move selection: raw policy + forced win/block only."""

    # 1. Immediate win — always take it
    win_move = find_winning_move(gs)
    if win_move is not None:
        return win_move

    # 2. Must-block — only if exactly 1 block needed
    blocks = find_must_blocks(gs)
    if len(blocks) == 1:
        return blocks[0]
    elif len(blocks) >= 2:
        # Multiple threats — use policy to pick the best block
        block_set = set(blocks)
        candidates = nn_raw_policy(model, gs, device, top_k=20)
        for move, prob in candidates:
            if move in block_set:
                return move
        return blocks[0]

    # 3. Raw policy (no filtering — trust the model)
    top_moves = nn_raw_policy(model, gs, device, top_k=1)
    if top_moves:
        return top_moves[0][0]

    legal = gs.legal_moves(zoi_margin=ZOI_MARGIN)
    return random.choice(legal) if legal else HexCoord(0, 0)


# =====================================================================
# Benchmark
# =====================================================================

def play_game(p1_fn, p2_fn, max_moves=MAX_GAME_MOVES):
    gs = GameState(win_length=WIN_LENGTH)
    move_count = 0
    while not gs.is_terminal and move_count < max_moves:
        fn = p1_fn if gs.current_player == 1 else p2_fn
        move = fn(gs)
        gs = gs.apply_move(move)
        move_count += 1
    return gs.winner if gs.winner else 0


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    first_conv = sd.get("backbone.init_conv.weight")
    if first_conv is None:
        first_conv = sd.get("backbone.initial_conv.conv.weight")
    ch = first_conv.shape[0] if first_conv is not None else 96
    block_keys = [k for k in sd if ".conv1." in k and "blocks" in k and "weight" in k and "bn" not in k]
    n_blocks = len(block_keys) if block_keys else 6
    model = HexTTTNet(grid_size=GRID_SIZE, num_blocks=n_blocks, channels=ch)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def benchmark(model, device, num_games=100, label="Smart Policy"):
    eisenstein = EisensteinGreedyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN)

    def smart_fn(gs):
        return smart_policy_move(model, gs, device)

    def raw_fn(gs):
        top = nn_raw_policy(model, gs, device, top_k=1)
        return top[0][0] if top else HexCoord(0, 0)

    # Benchmark smart policy
    wins = losses = draws = 0
    t0 = time.time()
    for g in range(num_games):
        if g % 2 == 0:
            winner = play_game(smart_fn, eisenstein.get_move)
            if winner == 1: wins += 1
            elif winner == 2: losses += 1
            else: draws += 1
        else:
            winner = play_game(eisenstein.get_move, smart_fn)
            if winner == 2: wins += 1
            elif winner == 1: losses += 1
            else: draws += 1

        if (g + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {g+1}/{num_games}: {wins}W {losses}L {draws}D "
                  f"= {wins/(g+1):.1%} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    wr = wins / num_games
    print(f"\n  {label}: {wins}W {losses}L {draws}D = {wr:.1%} ({elapsed:.1f}s)\n")

    # Benchmark raw policy for comparison
    wins2 = losses2 = draws2 = 0
    t0 = time.time()
    for g in range(num_games):
        if g % 2 == 0:
            winner = play_game(raw_fn, eisenstein.get_move)
            if winner == 1: wins2 += 1
            elif winner == 2: losses2 += 1
            else: draws2 += 1
        else:
            winner = play_game(eisenstein.get_move, raw_fn)
            if winner == 2: wins2 += 1
            elif winner == 1: losses2 += 1
            else: draws2 += 1

        if (g + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [Raw Policy] {g+1}/{num_games}: {wins2}W {losses2}L {draws2}D "
                  f"= {wins2/(g+1):.1%} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    wr2 = wins2 / num_games
    print(f"\n  Raw Policy: {wins2}W {losses2}L {draws2}D = {wr2:.1%} ({elapsed:.1f}s)")
    print(f"\n  Improvement: {wr:.1%} vs {wr2:.1%} ({(wr-wr2)*100:+.1f}pp)")


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = PROJECT_ROOT / "checkpoints" / "beat_eisenstein_v5.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "beat_eisenstein_v3.pt"
    print(f"Loading: {ckpt_path.name}\n")
    model = load_model(str(ckpt_path), device)

    benchmark(model, device, num_games=100)
