#!/usr/bin/env python3
"""Quick benchmark: MCTS-NN vs Eisenstein after coordinate transpose fix."""

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
    # Infer architecture from state dict
    # Try both possible key patterns for the initial conv
    first_conv = sd.get("backbone.init_conv.weight")
    if first_conv is None:
        first_conv = sd.get("backbone.initial_conv.conv.weight")
    in_ch = first_conv.shape[1] if first_conv is not None else 12
    ch = first_conv.shape[0] if first_conv is not None else 96
    # Count res blocks
    block_keys = [k for k in sd if ".conv1." in k and k.startswith("backbone.blocks.") and "weight" in k and "bn" not in k]
    n_blocks = len(block_keys) if block_keys else 6
    model = HexTTTNet(grid_size=GRID_SIZE, num_blocks=n_blocks, channels=ch, in_channels=in_ch)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def play_game_mcts_vs_eis(mcts_engine, eisenstein, mcts_is_p1=True, verbose=False):
    gs = GameState(win_length=WIN_LENGTH)
    move_count = 0

    while not gs.is_terminal and move_count < MAX_GAME_MOVES:
        is_mcts_turn = (gs.current_player == 1) == mcts_is_p1
        if is_mcts_turn:
            move, policy, _ = mcts_engine.get_move(gs, temperature=0.0)
            if verbose and move_count < 10:
                top_moves = sorted(policy.items(), key=lambda x: -x[1])[:3]
                print(f"  MCTS move {move_count}: {move} (top: {[(str(m), f'{p:.3f}') for m,p in top_moves]})")
        else:
            move = eisenstein.get_move(gs)
            if verbose and move_count < 10:
                print(f"  Eis  move {move_count}: {move}")

        gs = gs.apply_move(move)
        move_count += 1

    winner = gs.winner if gs.winner else 0
    if mcts_is_p1:
        return 1 if winner == 1 else (-1 if winner == 2 else 0)
    else:
        return 1 if winner == 2 else (-1 if winner == 1 else 0)


def benchmark(model, device, num_sims, num_games=20, cpuct=2.5, dirichlet_alpha=0.10, dirichlet_epsilon=0.25):
    config = {
        "num_simulations": num_sims,
        "cpuct": cpuct,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_epsilon": dirichlet_epsilon,
        "temperature": 0.0,
        "fpu_reduction": 0.0,
        "zoi_margin": ZOI_MARGIN,
        "grid_size": GRID_SIZE,
        "virtual_loss": 3,
        "device": device,
    }
    mcts_engine = MCTS(model, config)
    eisenstein = EisensteinGreedyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN)

    wins = 0
    losses = 0
    draws = 0

    t0 = time.time()
    for g in range(num_games):
        mcts_is_p1 = (g % 2 == 0)
        verbose = (g == 0)  # verbose first game only
        result = play_game_mcts_vs_eis(mcts_engine, eisenstein, mcts_is_p1, verbose=verbose)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
        elapsed = time.time() - t0
        print(f"  Game {g+1}/{num_games}: {'W' if result==1 else 'L' if result==-1 else 'D'}  "
              f"(running: {wins}W {losses}L {draws}D, {elapsed:.1f}s)")

    elapsed = time.time() - t0
    wr = wins / num_games
    print(f"\n  {num_sims} sims: {wins}W {losses}L {draws}D = {wr:.1%} win rate ({elapsed:.1f}s)")
    return wr


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = PROJECT_ROOT / "checkpoints" / "beat_eisenstein_v5.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "beat_eisenstein_v3.pt"
    print(f"Loading: {ckpt_path.name}")
    model = load_model(str(ckpt_path), device)

    # Test different MCTS configs
    configs_to_test = [
        ("Raw MCTS (baseline)", {"cpuct": 2.5, "dirichlet_epsilon": 0.25, "dirichlet_alpha": 0.10}),
        ("No noise", {"cpuct": 2.5, "dirichlet_epsilon": 0.0, "dirichlet_alpha": 0.10}),
        ("No noise + low cpuct", {"cpuct": 1.0, "dirichlet_epsilon": 0.0, "dirichlet_alpha": 0.10}),
        ("No noise + very low cpuct", {"cpuct": 0.5, "dirichlet_epsilon": 0.0, "dirichlet_alpha": 0.10}),
    ]

    for label, overrides in configs_to_test:
        print(f"\n=== {label} (50 sims) ===")
        benchmark(model, device, num_sims=50, num_games=20, **overrides)
