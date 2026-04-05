# EXPERIMENT: Baseline — adversarial refinement from v5 checkpoint (~84% win rate)
#!/usr/bin/env python3
"""Autoresearch train.py — the ONLY file the AI agent modifies.

Trains the model to beat EisensteinGreedy at 6-in-a-row Hex TTT.
Must print `>>> METRIC: {value}` exactly once at the end.
"""

import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

AUTORESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AUTORESEARCH_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from game.hex_grid import HexCoord, HEX_AXES, axial_to_brick
from game.rules import GameState
from nn.features import extract_features
from nn.model import HexTTTNet
from tournament import EisensteinGreedyAgent, OnePlyAgent, GreedyAgent

# Import prepare.py utilities (evaluation, model loading)
from prepare import (
    evaluate_model, load_best_model, save_best_model,
    nn_get_move, play_game, get_best_metric,
    GRID_SIZE, WIN_LENGTH, ZOI_MARGIN, MAX_MOVES,
    NUM_BLOCKS, CHANNELS, IN_CHANNELS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TIME_BUDGET = 600  # 10 minutes
BATCH_SIZE = 128
LR = 2e-4
POLICY_WEIGHT = 2.0
VALUE_WEIGHT = 1.0
ADV_GAMES_PER_ITER = 150
EXPERT_GAMES = 200


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_adversarial_data(model, device, num_games=ADV_GAMES_PER_ITER):
    """Play NN vs Eisenstein, collect training positions.

    From wins: reinforce NN's moves.
    From losses: use Eisenstein's moves as corrections.
    """
    eisenstein = EisensteinGreedyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN)
    all_samples = []
    nn_wins = 0

    def nn_fn(gs):
        return nn_get_move(model, gs, device)

    for g in range(num_games):
        gs = GameState(win_length=WIN_LENGTH)
        records = []
        move_count = 0
        nn_player = 1 if g % 2 == 0 else 2

        if nn_player == 1:
            p1_fn, p2_fn = nn_fn, eisenstein.get_move
        else:
            p1_fn, p2_fn = eisenstein.get_move, nn_fn

        while not gs.is_terminal and move_count < MAX_MOVES:
            fn = p1_fn if gs.current_player == 1 else p2_fn
            move = fn(gs)
            records.append({
                "game_state": gs,
                "move_played": move,
                "current_player": gs.current_player,
            })
            gs = gs.apply_move(move)
            move_count += 1

        winner = gs.winner if gs.winner else 0
        if winner == nn_player:
            nn_wins += 1

        # Extract training samples
        half = GRID_SIZE // 2
        for rec in records:
            rgs = rec["game_state"]
            cp = rec["current_player"]
            move_played = rec["move_played"]

            try:
                features, (cq, cr) = extract_features(rgs, grid_size=GRID_SIZE)
            except Exception:
                continue

            # Target move decision
            if cp == nn_player and winner == nn_player:
                target_move = move_played  # Reinforce winning moves
            elif cp != nn_player:
                target_move = move_played  # Learn from opponent's moves
            else:
                target_move = eisenstein.get_move(rgs)  # Eisenstein correction

            bx, by = axial_to_brick(target_move.q, target_move.r, cq, cr, GRID_SIZE)
            row = by + half
            col = bx + half
            if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
                continue

            policy = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
            policy[row * GRID_SIZE + col] = 1.0

            if winner == 0:
                value = np.float32(0.0)
            elif cp == winner:
                value = np.float32(1.0)
            else:
                value = np.float32(-1.0)

            all_samples.append({
                "features": features.numpy(),
                "policy": policy,
                "value": value,
            })

    return all_samples, nn_wins / max(num_games, 1)


def generate_expert_data(num_games=EXPERT_GAMES):
    """Generate supervised data from strong heuristic agents."""
    agents = [
        EisensteinGreedyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN, defensive=True),
        OnePlyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN),
    ]
    all_samples = []
    half = GRID_SIZE // 2

    for g in range(num_games):
        a1, a2 = random.choice(agents), random.choice(agents)
        gs = GameState(win_length=WIN_LENGTH)
        positions = []
        move_count = 0

        while not gs.is_terminal and move_count < 60:
            features, (cq, cr) = extract_features(gs, grid_size=GRID_SIZE)
            agent = a1 if gs.current_player == 1 else a2
            move = agent.get_move(gs)

            bx, by = axial_to_brick(move.q, move.r, cq, cr, GRID_SIZE)
            row = by + half
            col = bx + half
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                policy = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
                policy[row * GRID_SIZE + col] = 1.0
                positions.append({
                    "features": features.numpy(),
                    "policy": policy,
                    "current_player": gs.current_player,
                })

            gs = gs.apply_move(move)
            move_count += 1

        winner = gs.winner
        for pos in positions:
            if winner is None:
                pos["value"] = np.float32(0.0)
            elif pos["current_player"] == winner:
                pos["value"] = np.float32(1.0)
            else:
                pos["value"] = np.float32(-1.0)
            del pos["current_player"]

        all_samples.extend(positions)

    return all_samples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_on_data(model, dataset, device, epochs=8, lr=LR):
    """Train model on dataset."""
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 20
    )

    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i:i + BATCH_SIZE]
            if len(batch) < 4:
                continue

            features = torch.tensor(
                np.array([s["features"] for s in batch]), dtype=torch.float32
            ).to(device)
            policies = torch.tensor(
                np.array([s["policy"] for s in batch]), dtype=torch.float32
            ).to(device)
            values = torch.tensor(
                [s["value"] for s in batch], dtype=torch.float32
            ).to(device).unsqueeze(1)
            mask = torch.ones(len(batch), GRID_SIZE * GRID_SIZE, device=device)

            out = model(features, valid_moves_mask=mask)

            log_probs = F.log_softmax(out["policy_logits"], dim=1)
            policy_loss = -(policies * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(out["value"], values)
            loss = POLICY_WEIGHT * policy_loss + VALUE_WEIGHT * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    print(f"Time budget: {TIME_BUDGET}s")

    # Load current best model
    model = load_best_model(DEVICE)
    model.train()

    # Phase 1: Generate expert data for warmup
    print("\n--- Phase 1: Expert data ---")
    expert_data = generate_expert_data(num_games=EXPERT_GAMES)
    print(f"  Generated {len(expert_data)} expert positions")

    # Phase 2: Adversarial refinement loop
    print("\n--- Phase 2: Adversarial refinement ---")
    iteration = 0
    while time.time() - t_start < TIME_BUDGET - 120:  # Leave 2 min for eval
        iteration += 1
        print(f"\n  Iteration {iteration}:")

        # Generate adversarial data
        model.eval()
        adv_data, game_wr = generate_adversarial_data(model, DEVICE, num_games=ADV_GAMES_PER_ITER)
        print(f"    Generated {len(adv_data)} positions (game WR: {game_wr:.1%})")

        # Mix: 70% adversarial + 30% expert
        n_expert = len(adv_data) * 3 // 7
        mixed = adv_data + random.sample(expert_data, min(n_expert, len(expert_data)))
        random.shuffle(mixed)

        # Train
        loss = train_on_data(model, mixed, DEVICE, epochs=6, lr=LR)
        print(f"    Loss: {loss:.4f}")

        elapsed = time.time() - t_start
        print(f"    Elapsed: {elapsed:.0f}s / {TIME_BUDGET}s")

    # Phase 3: Evaluate
    print("\n--- Evaluation ---")
    model.eval()
    metric = evaluate_model(model, DEVICE, num_games=40)

    # Save if improved
    best = get_best_metric()
    if metric > best:
        save_best_model(model, metric, experiment_id=-1)
        print(f"  NEW BEST: {metric:.4f} > {best:.4f}")

    print(f"\n>>> METRIC: {metric:.4f}")


if __name__ == "__main__":
    main()
