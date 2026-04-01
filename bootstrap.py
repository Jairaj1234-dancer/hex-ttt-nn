#!/usr/bin/env python3
"""Bootstrap pre-training: supervised learning from strong baseline games.

Generates games between Eisenstein agents (and OnePly), extracts features
+ move targets, and trains the NN via supervised learning. This gives the
network a reasonable starting point before expensive MCTS self-play.

Phase 1: Generate labeled dataset from baseline agent games
Phase 2: Supervised pre-training (policy head learns to imitate strong play)
Phase 3: Short self-play fine-tuning with curriculum

Usage:
    python bootstrap.py --config configs/bootstrap.yaml
    python bootstrap.py --config configs/bootstrap.yaml --device mps
"""

import argparse
import logging
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from game.hex_grid import HexCoord
from game.rules import GameState
from nn.features import extract_features
from nn.model import HexTTTNet
from tournament import EisensteinGreedyAgent, OnePlyAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Dataset generation
# ======================================================================

def generate_games(
    num_games: int,
    grid_size: int = 13,
    win_length: int = 6,
    zoi_margin: int = 2,
    max_moves: int = 40,
) -> List[dict]:
    """Generate labeled positions from strong baseline games.

    Uses Eisenstein vs Eisenstein, Eisenstein vs OnePly, and OnePly vs
    OnePly matchups to produce diverse positions with known good moves.
    """
    agents = [
        ("Eisenstein", EisensteinGreedyAgent(win_length=win_length, zoi_margin=zoi_margin, defensive=True)),
        ("OnePly", OnePlyAgent(win_length=win_length, zoi_margin=zoi_margin)),
    ]

    all_samples = []

    for game_idx in range(num_games):
        # Pick two agents (can be the same)
        a1_name, a1 = random.choice(agents)
        a2_name, a2 = random.choice(agents)

        game_state = GameState(win_length=win_length)
        positions = []
        half_move = 0

        while not game_state.is_terminal and half_move < max_moves:
            # Extract features BEFORE the move
            features, (cq, cr) = extract_features(game_state, grid_size=grid_size)

            # Get the agent's move
            agent = a1 if game_state.current_player == 1 else a2
            move = agent.get_move(game_state)

            # Convert move to grid index for policy target
            from nn.features import axial_to_brick
            bx, by = axial_to_brick(move.q, move.r, cq, cr, grid_size)

            if 0 <= bx < grid_size and 0 <= by < grid_size:
                # Create one-hot policy target
                policy = np.zeros(grid_size * grid_size, dtype=np.float32)
                policy[by * grid_size + bx] = 1.0

                positions.append({
                    "features": features.numpy(),
                    "policy": policy,
                    "current_player": game_state.current_player,
                    "center": (cq, cr),
                })

            game_state = game_state.apply_move(move)
            half_move += 1

        # Assign value targets
        winner = game_state.winner
        for pos in positions:
            if winner is None:
                pos["value"] = 0.0
            elif pos["current_player"] == winner:
                pos["value"] = 1.0
            else:
                pos["value"] = -1.0

        all_samples.extend(positions)

        if (game_idx + 1) % 100 == 0:
            logger.info(
                "Generated %d/%d games (%d positions)",
                game_idx + 1, num_games, len(all_samples),
            )

    logger.info("Dataset: %d games -> %d positions", num_games, len(all_samples))
    return all_samples


# ======================================================================
# Supervised training
# ======================================================================

def train_supervised(
    model: HexTTTNet,
    dataset: List[dict],
    config: dict,
    device: str = "cpu",
) -> HexTTTNet:
    """Train the model to imitate strong baseline agent moves."""
    train_cfg = config.get("bootstrap", {})
    epochs = train_cfg.get("epochs", 30)
    batch_size = train_cfg.get("batch_size", 128)
    lr = train_cfg.get("learning_rate", 0.005)
    grid_size = config["network"]["grid_size"]

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            if len(batch) < 4:
                continue

            features = torch.tensor(
                np.array([s["features"] for s in batch]), dtype=torch.float32
            ).to(device)
            target_policy = torch.tensor(
                np.array([s["policy"] for s in batch]), dtype=torch.float32
            ).to(device)
            target_value = torch.tensor(
                [s["value"] for s in batch], dtype=torch.float32
            ).to(device).unsqueeze(1)

            # Valid moves mask: anywhere the policy target could be
            # (all positions in grid for supervised learning)
            valid_mask = torch.ones(len(batch), grid_size * grid_size).to(device)

            output = model(features, valid_moves_mask=valid_mask)

            # Policy loss: cross-entropy with soft targets
            log_policy = torch.log(output["policy"] + 1e-8)
            p_loss = -(target_policy * log_policy).sum(dim=1).mean()

            # Value loss: MSE
            v_loss = F.mse_loss(output["value"], target_value)

            loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            policy_loss_sum += p_loss.item()
            value_loss_sum += v_loss.item()
            num_batches += 1

            # Track top-1 accuracy
            pred_moves = output["policy"].argmax(dim=1)
            target_moves = target_policy.argmax(dim=1)
            correct += (pred_moves == target_moves).sum().item()
            total += len(batch)

        scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_p = policy_loss_sum / max(num_batches, 1)
        avg_v = value_loss_sum / max(num_batches, 1)
        acc = 100.0 * correct / max(total, 1)

        logger.info(
            "Epoch %d/%d | loss=%.4f (policy=%.4f value=%.4f) | "
            "move accuracy=%.1f%% | lr=%.6f",
            epoch + 1, epochs, avg_loss, avg_p, avg_v, acc,
            scheduler.get_last_lr()[0],
        )

    return model


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_vs_baselines(model: HexTTTNet, config: dict, device: str, num_games: int = 20):
    """Quick evaluation of the model against baseline agents."""
    from mcts.search import MCTS

    grid_size = config["network"]["grid_size"]
    mcts_cfg = dict(config.get("mcts", {}))
    mcts_cfg["dirichlet_epsilon"] = 0.0
    mcts_cfg["grid_size"] = grid_size
    mcts_cfg["device"] = device
    win_length = config["game"]["win_length"]
    zoi_margin = mcts_cfg.get("zoi_margin", 2)

    mcts = MCTS(model, mcts_cfg)
    model.eval()

    opponents = {
        "Random": lambda gs: random.choice(gs.legal_moves(zoi_margin=zoi_margin)),
        "Greedy": lambda gs: _greedy_move(gs, win_length, zoi_margin),
        "OnePly": OnePlyAgent(win_length=win_length, zoi_margin=zoi_margin).get_move,
        "Eisenstein": EisensteinGreedyAgent(win_length=win_length, zoi_margin=zoi_margin).get_move,
    }

    for opp_name, opp_fn in opponents.items():
        wins = 0
        for g in range(num_games):
            nn_player = 1 if g % 2 == 0 else 2
            game_state = GameState(win_length=win_length)
            moves = 0
            while not game_state.is_terminal and moves < 80:
                if game_state.current_player == nn_player:
                    with torch.no_grad():
                        move, _, _ = mcts.get_move(game_state, temperature=0.0)
                else:
                    move = opp_fn(game_state)
                game_state = game_state.apply_move(move)
                moves += 1
            if game_state.winner == nn_player:
                wins += 1

        logger.info("vs %-12s: %d/%d wins (%.0f%%)", opp_name, wins, num_games, 100 * wins / num_games)


def _greedy_move(gs, win_length, zoi_margin):
    """Simple greedy helper for eval."""
    from tournament import GreedyAgent
    return GreedyAgent(zoi_margin=zoi_margin).get_move(gs)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Bootstrap pre-training from baseline agents")
    parser.add_argument("--config", default="configs/bootstrap.yaml")
    parser.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    parser.add_argument("--skip-generate", action="store_true", help="Load existing dataset")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    net_cfg = config["network"]
    grid_size = net_cfg["grid_size"]
    bootstrap_cfg = config.get("bootstrap", {})

    logger.info("Bootstrap training on %s (grid=%d)", device, grid_size)

    # Step 1: Generate dataset
    dataset_path = "bootstrap_dataset.npy"
    if args.skip_generate and os.path.exists(dataset_path):
        logger.info("Loading existing dataset from %s", dataset_path)
        dataset = list(np.load(dataset_path, allow_pickle=True))
    else:
        num_games = bootstrap_cfg.get("num_games", 2000)
        max_moves = config.get("mcts", {}).get("max_moves", 40)
        zoi_margin = config.get("mcts", {}).get("zoi_margin", 2)
        dataset = generate_games(
            num_games=num_games,
            grid_size=grid_size,
            win_length=config["game"]["win_length"],
            zoi_margin=zoi_margin,
            max_moves=max_moves,
        )
        np.save(dataset_path, dataset, allow_pickle=True)
        logger.info("Dataset saved to %s", dataset_path)

    # Step 2: Create and train model
    model = HexTTTNet(
        grid_size=grid_size,
        num_blocks=net_cfg["num_blocks"],
        channels=net_cfg["channels"],
        in_channels=net_cfg.get("in_channels", 12),
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d blocks, %d channels, %d params", net_cfg["num_blocks"], net_cfg["channels"], param_count)

    t0 = time.time()
    model = train_supervised(model, dataset, config, device=device)
    elapsed = time.time() - t0
    logger.info("Supervised training done in %.1fs", elapsed)

    # Step 3: Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/bootstrap.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "bootstrap_games": bootstrap_cfg.get("num_games", 2000),
        "grid_size": grid_size,
    }, ckpt_path)
    logger.info("Checkpoint saved to %s", ckpt_path)

    # Step 4: Evaluate
    logger.info("Evaluating bootstrapped model...")
    evaluate_vs_baselines(model, config, device, num_games=10)

    logger.info("Done! Resume with: python train.py --config configs/bootstrap.yaml --checkpoint %s", ckpt_path)


if __name__ == "__main__":
    main()
