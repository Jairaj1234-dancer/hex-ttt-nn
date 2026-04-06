#!/usr/bin/env python3
"""Test MCTS with value-tuned model (v6) + branch limiting."""

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
    model = HexTTTNet(grid_size=GRID_SIZE, num_blocks=6, channels=96)
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


def benchmark(model, device, num_sims, max_branches, num_games=40, cpuct=2.5):
    config = {
        "num_simulations": num_sims,
        "cpuct": cpuct,
        "dirichlet_alpha": 0.10,
        "dirichlet_epsilon": 0.0,
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
    br_label = f"top-{max_branches}" if max_branches > 0 else "unlimited"
    print(f"  {num_sims:3d} sims, {br_label:>12s}, cpuct={cpuct}: "
          f"{wins}W {losses}L {draws}D = {wr:.1%} ({elapsed:.1f}s)")
    return wr


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Test both v5 (raw) and v6 (value-tuned) with best configs
    for ckpt_name in ["beat_eisenstein_v5.pt", "beat_eisenstein_v6_mcts.pt"]:
        ckpt_path = PROJECT_ROOT / "checkpoints" / ckpt_name
        if not ckpt_path.exists():
            continue
        print(f"\n{'='*60}")
        print(f"  Model: {ckpt_name}")
        print(f"{'='*60}")
        model = load_model(str(ckpt_path), device)

        # Sweep: branch limits x sim counts
        for branches in [3, 5, 8]:
            for sims in [25, 50]:
                benchmark(model, device, num_sims=sims, max_branches=branches, num_games=40)

        # Also test with higher cpuct (trust policy more)
        print("  --- Higher cpuct (trust policy more) ---")
        for cpuct in [3.5, 5.0]:
            benchmark(model, device, num_sims=50, max_branches=8, num_games=40, cpuct=cpuct)
