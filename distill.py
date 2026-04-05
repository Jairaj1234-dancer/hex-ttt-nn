#!/usr/bin/env python3
"""Knowledge distillation: learn from the collaborator's stronger model.

Uses net_gen0162.pt (1.9M params, 162 generations of training) as a teacher
to improve our model's policy and value heads. Generates positions from
baseline games, extracts teacher predictions, and trains our model to match.

This supplements bootstrap training by providing policy knowledge from a
model that has already learned complex tactical patterns.

Usage:
    python distill.py --teacher ~/Downloads/net_gen0162.pt \
                      --student checkpoints/bootstrap_scaled_w4_50k.pt \
                      --config configs/scaled_w4.yaml \
                      --device mps --epochs 30
"""

import argparse
import logging
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from game.hex_grid import HexCoord, axial_to_brick
from game.rules import GameState
from nn.features import extract_features
from nn.compat_features import extract_compat_features
from nn.model import HexTTTNet
from nn.compat_model import load_compat_model
from tournament import EisensteinGreedyAgent, OnePlyAgent, GreedyAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_positions(
    num_games: int,
    win_length: int = 4,
    grid_size_student: int = 13,
    grid_size_teacher: int = 18,
    zoi_margin: int = 2,
    max_moves: int = 40,
) -> List[dict]:
    """Generate diverse positions with features for both student and teacher."""
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
            # Extract features for both models
            student_feat, (scq, scr) = extract_features(gs, grid_size=grid_size_student)
            teacher_feat, (tcq, tcr) = extract_compat_features(gs, grid_size=grid_size_teacher)

            positions.append({
                "student_features": student_feat.numpy(),
                "teacher_features": teacher_feat.numpy(),
                "student_center": (scq, scr),
                "teacher_center": (tcq, tcr),
                "current_player": gs.current_player,
            })

            agent = a1 if gs.current_player == 1 else a2
            move = agent.get_move(gs)
            gs = gs.apply_move(move)
            hm += 1

        # Assign value targets
        winner = gs.winner
        for pos in positions:
            if winner is None:
                pos["value"] = 0.0
            elif pos["current_player"] == winner:
                pos["value"] = 1.0
            else:
                pos["value"] = -1.0

        samples.extend(positions)
        if (g + 1) % 500 == 0:
            logger.info("Generated %d/%d games (%d positions)", g + 1, num_games, len(samples))

    logger.info("Total: %d positions from %d games", len(samples), num_games)
    return samples


def distill(
    student: HexTTTNet,
    teacher: torch.nn.Module,
    dataset: List[dict],
    config: dict,
    device: str = "cpu",
    epochs: int = 30,
    temperature: float = 3.0,
    alpha: float = 0.7,
) -> HexTTTNet:
    """Train student to match teacher's soft policy targets.

    Loss = alpha * KD_loss + (1-alpha) * hard_value_loss

    KD_loss uses soft targets from teacher with temperature scaling.
    """
    grid_s = config["network"]["grid_size"]
    grid_t = 18
    batch_size = config.get("bootstrap", {}).get("batch_size", 128)
    lr = config.get("bootstrap", {}).get("learning_rate", 0.001)

    student.to(device)
    student.train()
    teacher.to(device)
    teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0.0
        kd_loss_sum = 0.0
        val_loss_sum = 0.0
        match_count = 0
        total = 0
        num_batches = 0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            if len(batch) < 4:
                continue

            # Student forward pass
            s_feat = torch.tensor(
                np.array([s["student_features"] for s in batch]), dtype=torch.float32
            ).to(device)
            s_mask = torch.ones(len(batch), grid_s * grid_s).to(device)
            s_out = student(s_feat, valid_moves_mask=s_mask)

            # Teacher forward pass (no grad)
            t_feat = torch.tensor(
                np.array([s["teacher_features"] for s in batch]), dtype=torch.float32
            ).to(device)
            t_mask = torch.ones(len(batch), grid_t * grid_t).to(device)
            with torch.no_grad():
                t_out = teacher(t_feat, valid_moves_mask=t_mask)

            # Teacher's soft policy: softmax with temperature
            t_logits = t_out["policy_logits"] / temperature
            t_soft = F.softmax(t_logits, dim=1)  # (B, 18*18)

            # Map teacher policy to student grid
            # For each position, find the overlapping region
            kd_targets = torch.zeros(len(batch), grid_s * grid_s, device=device)
            for b_idx in range(len(batch)):
                t_policy_2d = t_soft[b_idx].reshape(grid_t, grid_t)
                # Simple center crop: take the center grid_s x grid_s from grid_t x grid_t
                offset = (grid_t - grid_s) // 2
                cropped = t_policy_2d[offset:offset + grid_s, offset:offset + grid_s]
                kd_targets[b_idx] = cropped.reshape(-1)
                # Renormalize
                kd_sum = kd_targets[b_idx].sum()
                if kd_sum > 0:
                    kd_targets[b_idx] /= kd_sum

            # KD loss: KL divergence between student and teacher soft targets
            s_log_probs = F.log_softmax(s_out["policy_logits"] / temperature, dim=1)
            kd_loss = F.kl_div(s_log_probs, kd_targets, reduction='batchmean') * (temperature ** 2)

            # Value loss
            target_value = torch.tensor(
                [s["value"] for s in batch], dtype=torch.float32
            ).to(device).unsqueeze(1)
            val_loss = F.mse_loss(s_out["value"], target_value)

            # Combined loss
            loss = alpha * kd_loss + (1 - alpha) * val_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            kd_loss_sum += kd_loss.item()
            val_loss_sum += val_loss.item()
            num_batches += 1

            # Track policy agreement
            s_moves = s_out["policy"].argmax(dim=1)
            t_moves_mapped = kd_targets.argmax(dim=1)
            match_count += (s_moves == t_moves_mapped).sum().item()
            total += len(batch)

        scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_kd = kd_loss_sum / max(num_batches, 1)
        avg_val = val_loss_sum / max(num_batches, 1)
        agreement = 100.0 * match_count / max(total, 1)

        logger.info(
            "Epoch %d/%d | loss=%.4f (kd=%.4f val=%.4f) | "
            "teacher agreement=%.1f%% | lr=%.6f",
            epoch + 1, epochs, avg_loss, avg_kd, avg_val, agreement,
            scheduler.get_last_lr()[0],
        )

    return student


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation from collaborator's model")
    parser.add_argument("--teacher", default=os.path.expanduser("~/Downloads/net_gen0162.pt"))
    parser.add_argument("--student", default="checkpoints/bootstrap_scaled_w4_50k.pt")
    parser.add_argument("--config", default="configs/scaled_w4.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-games", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--output", default="checkpoints/distilled.pt")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    net_cfg = config["network"]

    # Load teacher
    logger.info("Loading teacher: %s", args.teacher)
    teacher = load_compat_model(args.teacher, device=torch.device(device))
    logger.info("Teacher params: %d", sum(p.numel() for p in teacher.parameters()))

    # Load student
    logger.info("Loading student: %s", args.student)
    student = HexTTTNet(
        grid_size=net_cfg["grid_size"],
        num_blocks=net_cfg["num_blocks"],
        channels=net_cfg["channels"],
        in_channels=net_cfg.get("in_channels", 12),
    )
    if os.path.exists(args.student):
        ckpt = torch.load(args.student, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            student.load_state_dict(ckpt["model_state_dict"])
        else:
            student.load_state_dict(ckpt)
        logger.info("Loaded student weights")
    else:
        logger.info("No student checkpoint, starting from scratch")
    logger.info("Student params: %d", sum(p.numel() for p in student.parameters()))

    # Generate positions
    logger.info("Generating %d games for distillation...", args.num_games)
    win_length = config["game"]["win_length"]
    zoi_margin = config.get("mcts", {}).get("zoi_margin", 3 if win_length >= 6 else 2)
    max_moves = config.get("mcts", {}).get("max_moves", 60 if win_length >= 6 else 40)
    dataset = generate_positions(
        num_games=args.num_games,
        win_length=win_length,
        grid_size_student=net_cfg["grid_size"],
        grid_size_teacher=18,
        zoi_margin=zoi_margin,
        max_moves=max_moves,
    )

    # Distill
    logger.info("Starting distillation (%d epochs, temp=%.1f, alpha=%.2f)...", args.epochs, args.temperature, args.alpha)
    t0 = time.time()
    student = distill(
        student, teacher, dataset, config,
        device=device,
        epochs=args.epochs,
        temperature=args.temperature,
        alpha=args.alpha,
    )
    elapsed = time.time() - t0
    logger.info("Distillation complete in %.1fs", elapsed)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": student.state_dict(),
        "config": config,
        "teacher": os.path.basename(args.teacher),
        "distill_epochs": args.epochs,
    }, args.output)
    logger.info("Saved distilled model to %s", args.output)


if __name__ == "__main__":
    main()
