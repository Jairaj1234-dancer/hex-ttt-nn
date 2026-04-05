#!/usr/bin/env python3
"""Quick 20-epoch bootstrap training on MPS."""
import os, sys, time, random
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Log to file directly
LOG = open("bootstrap_20_log.txt", "w")
def log(msg):
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()

from nn.model import HexTTTNet

# Load config
with open("configs/scaled_w4.yaml") as f:
    config = yaml.safe_load(f)
net_cfg = config["network"]

# Load dataset
data = list(np.load("bootstrap_dataset_w4_50k.npy", allow_pickle=True))
log(f"Dataset: {len(data)} positions")

# Build model
model = HexTTTNet(
    grid_size=net_cfg["grid_size"],
    num_blocks=net_cfg["num_blocks"],
    channels=net_cfg["channels"],
    in_channels=net_cfg.get("in_channels", 12),
)
model.to("mps")
model.train()
params = sum(p.numel() for p in model.parameters())
log(f"Model: {params:,} params")

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001)
grid_size = net_cfg["grid_size"]

t0 = time.time()
for epoch in range(20):
    random.shuffle(data)
    total_loss = 0.0
    pol_sum = 0.0
    val_sum = 0.0
    correct = 0
    total = 0
    nb = 0

    for i in range(0, len(data), 256):
        batch = data[i : i + 256]
        if len(batch) < 4:
            continue

        feat = torch.tensor(
            np.array([s["features"] for s in batch]), dtype=torch.float32
        ).to("mps")
        pol_t = torch.tensor(
            np.array([s["policy"] for s in batch]), dtype=torch.float32
        ).to("mps")
        val_t = (
            torch.tensor([s["value"] for s in batch], dtype=torch.float32)
            .unsqueeze(1)
            .to("mps")
        )
        mask = torch.ones(len(batch), grid_size * grid_size).to("mps")

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
    print(
        f"Epoch {epoch+1:2d}/20 | loss={total_loss/nb:.4f} "
        f"(pol={pol_sum/nb:.4f} val={val_sum/nb:.4f}) | "
        f"acc={100*correct/total:.1f}% | lr={scheduler.get_last_lr()[0]:.6f} | {el:.0f}s",
        flush=True,
    )

# Save
os.makedirs("checkpoints", exist_ok=True)
torch.save(
    {"model_state_dict": model.state_dict(), "config": config},
    "checkpoints/bootstrap_scaled_w4_50k.pt",
)
print(f"\nSaved to checkpoints/bootstrap_scaled_w4_50k.pt in {time.time()-t0:.0f}s")
