#!/usr/bin/env python3
"""Generate 6-in-a-row bootstrap dataset + train model. Laptop-safe (CPU)."""
import gc
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from nn.model import HexTTTNet
from nn.features import extract_features
from game.hex_grid import HexCoord, axial_to_brick
from game.rules import GameState
from tournament import EisensteinGreedyAgent, OnePlyAgent, GreedyAgent

LOG_PATH = "bootstrap_w6_log.txt"
log_file = open(LOG_PATH, "w")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def generate_games(num_games, grid_size, win_length, zoi_margin=3, max_moves=60):
    """Generate labeled positions from baseline agent games."""
    agents = [
        EisensteinGreedyAgent(win_length=win_length, zoi_margin=zoi_margin, defensive=True),
        OnePlyAgent(win_length=win_length, zoi_margin=zoi_margin),
        GreedyAgent(zoi_margin=zoi_margin),
    ]

    samples = []
    for g in range(num_games):
        a1, a2 = random.choice(agents), random.choice(agents)
        gs = GameState(win_length=win_length)
        positions = []
        hm = 0

        while not gs.is_terminal and hm < max_moves:
            features, (cq, cr) = extract_features(gs, grid_size=grid_size)

            # Get move from current agent
            agent = a1 if gs.current_player == 1 else a2
            move = agent.get_move(gs)

            # Build policy target (one-hot at the move)
            half = grid_size // 2
            bx, by = axial_to_brick(move.q, move.r, cq, cr, grid_size)
            row = by + half
            col = bx + half
            policy = np.zeros(grid_size * grid_size, dtype=np.float32)
            if 0 <= row < grid_size and 0 <= col < grid_size:
                policy[row * grid_size + col] = 1.0

            positions.append({
                "features": features.numpy(),
                "policy": policy,
                "current_player": gs.current_player,
                "center": (cq, cr),
            })

            gs = gs.apply_move(move)
            hm += 1

        # Assign value targets based on game outcome
        winner = gs.winner
        for pos in positions:
            if winner is None:
                pos["value"] = 0.0
            elif pos["current_player"] == winner:
                pos["value"] = 1.0
            else:
                pos["value"] = -1.0

        samples.extend(positions)
        if (g + 1) % 2000 == 0:
            log(f"Generated {g+1}/{num_games} games ({len(samples)} positions)")

    log(f"Total: {len(samples)} positions from {num_games} games")
    return samples


def main():
    with open("configs/scaled_w6.yaml") as f:
        config = yaml.safe_load(f)
    net_cfg = config["network"]
    grid_size = net_cfg["grid_size"]
    win_length = config["game"]["win_length"]

    log(f"Bootstrap for win_length={win_length}")

    # Step 1: Generate dataset
    dataset_path = f"bootstrap_dataset_w{win_length}_20k.npy"
    if os.path.exists(dataset_path):
        log(f"Loading existing dataset: {dataset_path}")
        data = list(np.load(dataset_path, allow_pickle=True))
        log(f"Loaded {len(data)} positions")
    else:
        log("Generating 20000 games...")
        data = generate_games(
            num_games=20000,
            grid_size=grid_size,
            win_length=win_length,
            zoi_margin=3,
            max_moves=60,
        )
        np.save(dataset_path, data, allow_pickle=True)
        log(f"Saved dataset to {dataset_path}")

    del_count = len(data)
    # Subsample for laptop-safe training
    if len(data) > 80000:
        random.shuffle(data)
        data = data[:80000]
        log(f"Subsampled to {len(data)} from {del_count}")

    # Step 2: Build model
    model = HexTTTNet(
        grid_size=grid_size,
        num_blocks=net_cfg["num_blocks"],
        channels=net_cfg["channels"],
        in_channels=net_cfg.get("in_channels", 12),
    )
    params = sum(p.numel() for p in model.parameters())
    log(f"Model: {params:,} params")

    # Step 3: Train on CPU (laptop-safe)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0.0001)

    t0 = time.time()
    for epoch in range(15):
        random.shuffle(data)
        total_loss = 0.0
        pol_sum = 0.0
        val_sum = 0.0
        correct = 0
        total = 0
        nb = 0

        for i in range(0, len(data), 128):
            batch = data[i:i+128]
            if len(batch) < 4:
                continue

            feat = torch.tensor(np.array([s["features"] for s in batch]), dtype=torch.float32)
            pol_t = torch.tensor(np.array([s["policy"] for s in batch]), dtype=torch.float32)
            val_t = torch.tensor([s["value"] for s in batch], dtype=torch.float32).unsqueeze(1)
            mask = torch.ones(len(batch), grid_size * grid_size)

            out = model(feat, valid_moves_mask=mask)
            log_probs = F.log_softmax(out["policy_logits"], dim=1)
            p_loss = -(pol_t * log_probs).sum(dim=1).mean()
            v_loss = F.mse_loss(out["value"], val_t)
            loss = p_loss + 0.5 * v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pol_sum += p_loss.item()
            val_sum += v_loss.item()
            correct += (out["policy"].argmax(1) == pol_t.argmax(1)).sum().item()
            total += len(batch)
            nb += 1

        scheduler.step()
        el = time.time() - t0
        log(
            f"Epoch {epoch+1:2d}/15 | loss={total_loss/nb:.4f} "
            f"(pol={pol_sum/nb:.4f} val={val_sum/nb:.4f}) | "
            f"acc={100*correct/total:.1f}% | lr={scheduler.get_last_lr()[0]:.6f} | {el:.0f}s"
        )

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/bootstrap_scaled_w{win_length}_20k.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        ckpt_path,
    )
    log(f"Saved to {ckpt_path} ({time.time()-t0:.0f}s)")
    log_file.close()


if __name__ == "__main__":
    main()
