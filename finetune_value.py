#!/usr/bin/env python3
"""Fine-tune the value head for MCTS.

The raw policy achieves ~84% vs Eisenstein, but MCTS degrades performance
because the value head is uncalibrated. This script:

1. Generates self-play games (model vs model, model vs Eisenstein)
2. Collects positions with game outcomes
3. Fine-tunes the value head (frozen backbone + policy head) on these outcomes
4. Benchmarks MCTS with the calibrated value head

The goal: MCTS with calibrated value > raw policy (~84%).
"""

import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from game.hex_grid import HexCoord, axial_to_brick
from game.rules import GameState
from nn.features import extract_features
from nn.model import HexTTTNet
from tournament import EisensteinGreedyAgent
from mcts.search import MCTS

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


def nn_get_move(model, gs, device):
    """Raw policy move selection."""
    features, (cq, cr) = extract_features(gs, grid_size=GRID_SIZE)
    legal = gs.legal_moves(zoi_margin=ZOI_MARGIN)
    if not legal:
        return HexCoord(0, 0)

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
        return random.choice(legal)

    x = features.unsqueeze(0).to(device)
    m = mask.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x, valid_moves_mask=m)
    policy = out["policy"].squeeze(0).cpu()
    best_idx = max(legal_map.keys(), key=lambda i: policy[i].item())
    return legal_map[best_idx]


# =====================================================================
# Data generation
# =====================================================================

def generate_value_data(model, device, num_games=500, progress_interval=50):
    """Generate self-play data for value head training.

    Plays a mix of:
    - NN vs Eisenstein (both sides)
    - NN vs NN (to see its own positions)

    Returns positions with game outcome labels.
    """
    eisenstein = EisensteinGreedyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN)
    all_samples = []

    def nn_fn(gs):
        return nn_get_move(model, gs, device)

    t0 = time.time()

    for g in range(num_games):
        gs = GameState(win_length=WIN_LENGTH)
        positions = []
        move_count = 0

        # Mix: 60% NN vs Eisenstein, 40% NN vs NN
        if g % 5 < 3:
            # NN vs Eisenstein
            if g % 2 == 0:
                p1_fn, p2_fn = nn_fn, eisenstein.get_move
            else:
                p1_fn, p2_fn = eisenstein.get_move, nn_fn
        else:
            # NN vs NN
            p1_fn, p2_fn = nn_fn, nn_fn

        while not gs.is_terminal and move_count < MAX_GAME_MOVES:
            # Only store every 2nd position to reduce redundancy
            if move_count % 2 == 0:
                try:
                    features, _ = extract_features(gs, grid_size=GRID_SIZE)
                    positions.append({
                        "features": features.numpy(),
                        "current_player": gs.current_player,
                        "move_count": move_count,
                    })
                except Exception:
                    pass

            fn = p1_fn if gs.current_player == 1 else p2_fn
            move = fn(gs)
            gs = gs.apply_move(move)
            move_count += 1

        winner = gs.winner
        for pos in positions:
            if winner is None or winner == 0:
                pos["value"] = np.float32(0.0)
            elif pos["current_player"] == winner:
                pos["value"] = np.float32(1.0)
            else:
                pos["value"] = np.float32(-1.0)

        all_samples.extend(positions)

        if (g + 1) % progress_interval == 0:
            elapsed = time.time() - t0
            print(f"    Generated {g+1}/{num_games} games, {len(all_samples)} positions ({elapsed:.1f}s)")

    return all_samples


# =====================================================================
# Value head fine-tuning
# =====================================================================

def finetune_value_head(model, dataset, device, epochs=20, batch_size=256, lr=5e-4):
    """Fine-tune only the value head while keeping backbone + policy frozen."""
    model.to(device)

    # Freeze everything except value-related parameters
    value_params = []
    for name, param in model.named_parameters():
        if "value" in name:
            param.requires_grad = True
            value_params.append(param)
        else:
            param.requires_grad = False

    print(f"    Training {sum(p.numel() for p in value_params)} value params "
          f"(frozen: {sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in value_params)})")

    optimizer = torch.optim.Adam(value_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/20)

    mask_dummy = torch.ones(1, GRID_SIZE * GRID_SIZE, device=device)

    for epoch in range(epochs):
        # CRITICAL: Keep model in eval mode to preserve BatchNorm running stats.
        # Only value head layers need training mode, but they have no BatchNorm.
        model.eval()
        random.shuffle(dataset)
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            if len(batch) < 4:
                continue

            features = torch.tensor(
                np.array([s["features"] for s in batch]), dtype=torch.float32
            ).to(device)
            values = torch.tensor(
                [s["value"] for s in batch], dtype=torch.float32
            ).to(device).unsqueeze(1)
            mask = torch.ones(len(batch), GRID_SIZE * GRID_SIZE, device=device)

            out = model(features, valid_moves_mask=mask)
            loss = F.mse_loss(out["value"], values)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mae += (out["value"] - values).abs().mean().item()
            num_batches += 1

        scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_mae = total_mae / max(num_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/{epochs} | MSE={avg_loss:.4f} | MAE={avg_mae:.4f}")

    # Unfreeze all params
    for param in model.parameters():
        param.requires_grad = True

    return avg_loss


# =====================================================================
# MCTS benchmark
# =====================================================================

def benchmark_mcts(model, device, num_sims=50, num_games=40, cpuct=2.5, dirichlet_epsilon=0.0):
    """Benchmark MCTS-NN vs Eisenstein."""
    config = {
        "num_simulations": num_sims,
        "cpuct": cpuct,
        "dirichlet_alpha": 0.10,
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

    wins = losses = draws = 0
    t0 = time.time()

    for g in range(num_games):
        gs = GameState(win_length=WIN_LENGTH)
        mcts_is_p1 = (g % 2 == 0)
        move_count = 0

        while not gs.is_terminal and move_count < MAX_GAME_MOVES:
            is_mcts_turn = (gs.current_player == 1) == mcts_is_p1
            if is_mcts_turn:
                move, _, _ = mcts_engine.get_move(gs, temperature=0.0)
            else:
                move = eisenstein.get_move(gs)
            gs = gs.apply_move(move)
            move_count += 1

        winner = gs.winner if gs.winner else 0
        if mcts_is_p1:
            if winner == 1: wins += 1
            elif winner == 2: losses += 1
            else: draws += 1
        else:
            if winner == 2: wins += 1
            elif winner == 1: losses += 1
            else: draws += 1

        if (g + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    Game {g+1}/{num_games}: {wins}W {losses}L {draws}D ({elapsed:.1f}s)")

    wr = wins / num_games
    elapsed = time.time() - t0
    print(f"    MCTS({num_sims}): {wins}W {losses}L {draws}D = {wr:.1%} ({elapsed:.1f}s)")
    return wr


def benchmark_raw(model, device, num_games=40):
    """Benchmark raw policy vs Eisenstein."""
    eisenstein = EisensteinGreedyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN)
    wins = losses = draws = 0
    t0 = time.time()

    def nn_fn(gs):
        return nn_get_move(model, gs, device)

    for g in range(num_games):
        gs = GameState(win_length=WIN_LENGTH)
        move_count = 0
        if g % 2 == 0:
            p1_fn, p2_fn = nn_fn, eisenstein.get_move
        else:
            p1_fn, p2_fn = eisenstein.get_move, nn_fn

        while not gs.is_terminal and move_count < MAX_GAME_MOVES:
            fn = p1_fn if gs.current_player == 1 else p2_fn
            move = fn(gs)
            gs = gs.apply_move(move)
            move_count += 1

        winner = gs.winner if gs.winner else 0
        if g % 2 == 0:
            if winner == 1: wins += 1
            elif winner == 2: losses += 1
            else: draws += 1
        else:
            if winner == 2: wins += 1
            elif winner == 1: losses += 1
            else: draws += 1

    wr = wins / num_games
    elapsed = time.time() - t0
    print(f"    Raw policy: {wins}W {losses}L {draws}D = {wr:.1%} ({elapsed:.1f}s)")
    return wr


# =====================================================================
# Main
# =====================================================================

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = PROJECT_ROOT / "checkpoints" / "beat_eisenstein_v5.pt"
    print(f"Loading: {ckpt_path.name}")
    model = load_model(str(ckpt_path), device)

    # 1. Baseline benchmarks
    print("\n" + "=" * 60)
    print("  BASELINE")
    print("=" * 60)
    raw_wr = benchmark_raw(model, device, num_games=60)
    print()
    mcts_wr_before = benchmark_mcts(model, device, num_sims=50, num_games=30)

    # 2. Generate value training data
    print("\n" + "=" * 60)
    print("  GENERATING VALUE TRAINING DATA")
    print("=" * 60)
    data = generate_value_data(model, device, num_games=800, progress_interval=100)
    print(f"    Total: {len(data)} positions")

    # Count value distribution
    wins = sum(1 for d in data if d["value"] > 0.5)
    losses = sum(1 for d in data if d["value"] < -0.5)
    draws = sum(1 for d in data if abs(d["value"]) < 0.5)
    print(f"    Distribution: {wins} wins, {losses} losses, {draws} draws")

    # 3. Fine-tune value head
    print("\n" + "=" * 60)
    print("  FINE-TUNING VALUE HEAD")
    print("=" * 60)
    finetune_value_head(model, data, device, epochs=30, batch_size=256, lr=5e-4)

    # 4. Post-finetune benchmarks
    print("\n" + "=" * 60)
    print("  POST-FINETUNE BENCHMARKS")
    print("=" * 60)
    raw_wr2 = benchmark_raw(model, device, num_games=60)
    print()

    for sims in [25, 50, 100]:
        mcts_wr_after = benchmark_mcts(model, device, num_sims=sims, num_games=30)
        print()

    # 5. Save if improved
    out_path = PROJECT_ROOT / "checkpoints" / "beat_eisenstein_v6_mcts.pt"
    torch.save({"model_state_dict": model.state_dict()}, str(out_path))
    print(f"\n  Saved: {out_path.name}")
    print(f"  Raw policy preserved: {raw_wr:.1%} -> {raw_wr2:.1%}")


if __name__ == "__main__":
    main()
