#!/usr/bin/env python3
"""Laptop-safe bootstrap training. CPU-only, low memory, logs to file."""
import gc
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from nn.model import HexTTTNet

# --- Config ---
DEVICE = "cpu"
EPOCHS = 15
BATCH_SIZE = 64
LR = 0.003
SUBSAMPLE = 50000  # use 50K of the 230K positions
LOG_PATH = "bootstrap_safe_log.txt"

log_file = open(LOG_PATH, "w")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def main():
    log("Loading config...")
    with open("configs/scaled_w4.yaml") as f:
        config = yaml.safe_load(f)
    net_cfg = config["network"]
    grid_size = net_cfg["grid_size"]

    log("Loading dataset...")
    raw = np.load("bootstrap_dataset_w4_50k.npy", allow_pickle=True)
    data = list(raw[:SUBSAMPLE])
    del raw
    gc.collect()
    log(f"Using {len(data)} positions (subsampled)")

    log("Building model...")
    model = HexTTTNet(
        grid_size=grid_size,
        num_blocks=net_cfg["num_blocks"],
        channels=net_cfg["channels"],
        in_channels=net_cfg.get("in_channels", 12),
    )
    params = sum(p.numel() for p in model.parameters())
    log(f"Model: {net_cfg['num_blocks']}b/{net_cfg['channels']}ch, {params:,} params")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 30
    )

    t0 = time.time()
    for epoch in range(EPOCHS):
        random.shuffle(data)
        total_loss = 0.0
        pol_sum = 0.0
        val_sum = 0.0
        correct = 0
        total = 0
        nb = 0

        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]
            if len(batch) < 4:
                continue

            feat = torch.tensor(
                np.array([s["features"] for s in batch]), dtype=torch.float32
            )
            pol_t = torch.tensor(
                np.array([s["policy"] for s in batch]), dtype=torch.float32
            )
            val_t = torch.tensor(
                [s["value"] for s in batch], dtype=torch.float32
            ).unsqueeze(1)
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
        elapsed = time.time() - t0
        log(
            f"Epoch {epoch+1:2d}/{EPOCHS} | "
            f"loss={total_loss/nb:.4f} (pol={pol_sum/nb:.4f} val={val_sum/nb:.4f}) | "
            f"acc={100*correct/total:.1f}% | "
            f"lr={scheduler.get_last_lr()[0]:.6f} | {elapsed:.0f}s"
        )

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/bootstrap_scaled_w4_50k.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        ckpt_path,
    )
    total_time = time.time() - t0
    log(f"Saved to {ckpt_path} ({total_time:.0f}s total)")
    log_file.close()


if __name__ == "__main__":
    main()
