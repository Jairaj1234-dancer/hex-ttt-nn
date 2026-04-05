#!/usr/bin/env python3
"""Autoresearch: FIXED evaluation harness. DO NOT MODIFY.

This file is the immutable evaluator in the Karpathy autoresearch pattern.
It provides:
  1. Model loading from checkpoint
  2. Game playing infrastructure
  3. Evaluation: play N games vs EisensteinGreedy baseline
  4. Metric: win rate [0.0, 1.0] (higher is better)

The AI agent modifies train.py; this file stays constant.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from game.hex_grid import HexCoord, axial_to_brick
from game.rules import GameState
from nn.features import extract_features
from nn.model import HexTTTNet
from tournament import EisensteinGreedyAgent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_GAMES = 40          # Games per evaluation (alternating sides)
WIN_LENGTH = 6           # Game win condition: 6-in-a-row
ZOI_MARGIN = 3           # Zone of interest margin
MAX_MOVES = 120          # Max moves per game
GRID_SIZE = 13           # Model grid size

# Model architecture (fixed — must match checkpoint)
NUM_BLOCKS = 6
CHANNELS = 96
IN_CHANNELS = 12

# Paths
AUTORESEARCH_DIR = Path(__file__).resolve().parent
LOG_FILE = AUTORESEARCH_DIR / "experiment_log.jsonl"
BEST_MODEL = AUTORESEARCH_DIR / "best_model.pt"
BASELINE_CKPT = PROJECT_ROOT / "checkpoints" / "beat_eisenstein_v5.pt"


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def create_model():
    """Create a fresh model with the fixed architecture."""
    return HexTTTNet(
        grid_size=GRID_SIZE,
        num_blocks=NUM_BLOCKS,
        channels=CHANNELS,
        in_channels=IN_CHANNELS,
    )


def load_checkpoint(path, device="cpu"):
    """Load a model from checkpoint."""
    model = create_model()
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def load_best_model(device="cpu"):
    """Load the current best model, falling back to baseline checkpoint."""
    if BEST_MODEL.exists():
        return load_checkpoint(BEST_MODEL, device)
    elif BASELINE_CKPT.exists():
        return load_checkpoint(BASELINE_CKPT, device)
    else:
        print("WARNING: No checkpoint found. Using untrained model.")
        model = create_model()
        model.to(device)
        return model


def save_best_model(model, metric, experiment_id):
    """Save the new best model."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "metric": metric,
        "experiment_id": experiment_id,
    }, str(BEST_MODEL))
    print(f"New best model saved: metric={metric:.3f}")


# ---------------------------------------------------------------------------
# Raw policy move selection
# ---------------------------------------------------------------------------

def nn_get_move(model, game_state, device):
    """Pick the highest-probability legal move using raw policy.

    Uses the transposed (column-major) indexing that matches training.
    """
    features, (cq, cr) = extract_features(game_state, grid_size=GRID_SIZE)
    legal = game_state.legal_moves(zoi_margin=ZOI_MARGIN)
    if not legal:
        return HexCoord(0, 0)

    half = GRID_SIZE // 2
    mask = torch.zeros(GRID_SIZE * GRID_SIZE, dtype=torch.float32)
    legal_map = {}

    for move in legal:
        bx, by = axial_to_brick(move.q, move.r, cq, cr, GRID_SIZE)
        row = by + half   # column-major: by is col from axial_to_brick
        col = bx + half   # column-major: bx is row from axial_to_brick
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


# ---------------------------------------------------------------------------
# Game playing
# ---------------------------------------------------------------------------

def play_game(p1_fn, p2_fn):
    """Play one game. Returns winner (1, 2, or 0 for draw)."""
    gs = GameState(win_length=WIN_LENGTH)
    move_count = 0
    while not gs.is_terminal and move_count < MAX_MOVES:
        fn = p1_fn if gs.current_player == 1 else p2_fn
        move = fn(gs)
        gs = gs.apply_move(move)
        move_count += 1
    return gs.winner if gs.winner else 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, device, num_games=EVAL_GAMES):
    """Evaluate model vs EisensteinGreedy. Returns win rate [0.0, 1.0]."""
    model.to(device)
    model.eval()

    eisenstein = EisensteinGreedyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN)
    wins = losses = draws = 0

    def nn_fn(gs):
        return nn_get_move(model, gs, device)

    for g in range(num_games):
        if g % 2 == 0:
            winner = play_game(nn_fn, eisenstein.get_move)
            if winner == 1: wins += 1
            elif winner == 2: losses += 1
            else: draws += 1
        else:
            winner = play_game(eisenstein.get_move, nn_fn)
            if winner == 2: wins += 1
            elif winner == 1: losses += 1
            else: draws += 1

    win_rate = wins / num_games
    print(f"Eval: {wins}W / {losses}L / {draws}D vs Eisenstein "
          f"({num_games} games) = {win_rate:.1%}")
    return win_rate


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

def log_experiment(experiment_id, description, metric, accepted):
    """Append experiment result to the log file."""
    entry = {
        "id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
        "metric": metric,
        "accepted": accepted,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_experiment_log():
    """Load all experiment results."""
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_best_metric():
    """Get the best metric achieved so far."""
    log = load_experiment_log()
    accepted = [e for e in log if e.get("accepted") and e.get("metric") is not None]
    if not accepted:
        return 0.0
    return max(e["metric"] for e in accepted)


# ---------------------------------------------------------------------------
# Main (standalone test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Autoresearch prepare.py — evaluation harness")
    print(f"Device: {device}")
    print(f"Baseline checkpoint: {BASELINE_CKPT}")
    print(f"Best model: {BEST_MODEL}")

    model = load_best_model(device)
    wr = evaluate_model(model, device)
    print(f"\n>>> METRIC: {wr:.4f}")
