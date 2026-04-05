#!/usr/bin/env python3
"""Test MCTS with branch limiting — keep only top-K policy moves per node."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from game.hex_grid import HexCoord
from game.rules import GameState
from nn.model import HexTTTNet
from mcts.search import MCTS
from tournament import EisensteinGreedyAgent

WIN_LENGTH = 6
GRID_SIZE = 13
ZOI_MARGIN = 3
MAX_GAME_MOVES = 120


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


def play_game(mcts_engine, eisenstein, mcts_is_p1):
    gs = GameState(win_length=WIN_LENGTH)
    move_count = 0
    while not gs.is_terminal and move_count < MAX_GAME_MOVES:
        is_mcts = (gs.current_player == 1) == mcts_is_p1
        if is_mcts:
            move, _, _ = mcts_engine.get_move(gs, temperature=0.0)
        else:
            move = eisenstein.get_move(gs)
        gs = gs.apply_move(move)
        move_count += 1
    winner = gs.winner if gs.winner else 0
    if mcts_is_p1:
        return 1 if winner == 1 else (-1 if winner == 2 else 0)
    else:
        return 1 if winner == 2 else (-1 if winner == 1 else 0)


def benchmark(model, device, num_sims, max_branches, num_games=30):
    config = {
        "num_simulations": num_sims,
        "cpuct": 2.5,
        "dirichlet_alpha": 0.10,
        "dirichlet_epsilon": 0.0,  # No noise for evaluation
        "temperature": 0.0,
        "fpu_reduction": 0.0,
        "zoi_margin": ZOI_MARGIN,
        "grid_size": GRID_SIZE,
        "virtual_loss": 3,
        "device": device,
        "max_branches": max_branches,
    }
    mcts_engine = MCTS(model, config)
    eisenstein = EisensteinGreedyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN)

    wins = losses = draws = 0
    t0 = time.time()
    for g in range(num_games):
        result = play_game(mcts_engine, eisenstein, g % 2 == 0)
        if result == 1: wins += 1
        elif result == -1: losses += 1
        else: draws += 1

    elapsed = time.time() - t0
    wr = wins / num_games
    print(f"  {num_sims:3d} sims, top-{max_branches:2d} branches: "
          f"{wins}W {losses}L {draws}D = {wr:.1%} ({elapsed:.1f}s)")
    return wr


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = PROJECT_ROOT / "checkpoints" / "beat_eisenstein_v5.pt"
    print(f"Loading: {ckpt_path.name}\n")
    model = load_model(str(ckpt_path), device)

    # Test branch limits with fixed sims
    print("--- Branch limit sweep (50 sims) ---")
    for branches in [3, 5, 8, 12, 0]:
        label = f"top-{branches}" if branches > 0 else "unlimited"
        benchmark(model, device, num_sims=50, max_branches=branches, num_games=30)

    print("\n--- Best branch limit with more sims ---")
    # Will fill in best branch count after seeing results
    for sims in [25, 50, 100]:
        benchmark(model, device, num_sims=sims, max_branches=5, num_games=30)
