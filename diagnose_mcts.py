#!/usr/bin/env python3
"""Diagnose MCTS: compare raw policy move selection vs MCTS policy extraction."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import random
import torch
import numpy as np
from game.hex_grid import HexCoord, axial_to_brick, brick_to_axial
from game.rules import GameState
from nn.features import extract_features
from nn.model import HexTTTNet
from mcts.search import MCTS
from mcts.zoi import compute_zoi_mask

WIN_LENGTH = 6
GRID_SIZE = 13
ZOI_MARGIN = 3


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    first_conv = sd.get("backbone.init_conv.weight")
    if first_conv is None:
        first_conv = sd.get("backbone.initial_conv.conv.weight")
    in_ch = first_conv.shape[1] if first_conv is not None else 12
    ch = first_conv.shape[0] if first_conv is not None else 96
    block_keys = [k for k in sd if ".conv1." in k and "blocks" in k and "weight" in k and "bn" not in k]
    n_blocks = len(block_keys) if block_keys else 6
    model = HexTTTNet(grid_size=GRID_SIZE, num_blocks=n_blocks, channels=ch, in_channels=in_ch)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def raw_policy_top_moves(model, gs, device, top_k=5):
    """Raw policy move selection from beat_eisenstein.py logic (known to work at ~84%)."""
    features, (cq, cr) = extract_features(gs, grid_size=GRID_SIZE)
    legal = gs.legal_moves(zoi_margin=ZOI_MARGIN)
    if not legal:
        return []

    half = GRID_SIZE // 2
    mask = torch.zeros(GRID_SIZE * GRID_SIZE, dtype=torch.float32)
    legal_map = {}

    for move in legal:
        bx, by = axial_to_brick(move.q, move.r, cq, cr, GRID_SIZE)
        row = by + half  # TRANSPOSED: col+half
        col = bx + half  # TRANSPOSED: row+half
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            idx = row * GRID_SIZE + col
            mask[idx] = 1.0
            legal_map[idx] = move

    x = features.unsqueeze(0).to(device)
    m = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x, valid_moves_mask=m)

    policy = out["policy"].squeeze(0).cpu().numpy()
    value = out["value"].squeeze().item()

    ranked = sorted(legal_map.items(), key=lambda kv: -policy[kv[0]])[:top_k]
    return [(legal_map[idx], float(policy[idx]), idx) for idx, move in ranked], value


def mcts_root_policy(model, gs, device, top_k=5):
    """Extract policy from MCTS root (no search — just NN evaluation)."""
    grid_size = GRID_SIZE
    half = grid_size // 2
    zoi_margin = ZOI_MARGIN

    features, (center_q, center_r) = extract_features(gs, grid_size=grid_size)
    zoi_mask_2d = compute_zoi_mask(gs, center_q, center_r, grid_size, margin=zoi_margin)

    # Apply same transpose as in search.py fix
    zoi_mask_2d_T = zoi_mask_2d.T.copy()

    features_batch = features.unsqueeze(0).to(device)
    mask_flat = torch.from_numpy(zoi_mask_2d_T.reshape(1, -1)).to(device)

    with torch.no_grad():
        output = model(features_batch, valid_moves_mask=mask_flat)

    policy_probs = output["policy"].squeeze(0).cpu().numpy()
    value = output["value"].squeeze().item()

    zoi_mask_flat = zoi_mask_2d_T.reshape(-1)
    move_priors = {}

    for idx in range(grid_size * grid_size):
        if zoi_mask_flat[idx] > 0.0:
            prob = float(policy_probs[idx])
            if prob > 0.0:
                col_plus_half = idx // grid_size
                row_plus_half = idx % grid_size
                coord = brick_to_axial(
                    row_plus_half - half, col_plus_half - half,
                    center_q, center_r, grid_size
                )
                move_priors[coord] = prob

    ranked = sorted(move_priors.items(), key=lambda kv: -kv[1])[:top_k]
    return ranked, value


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_path = PROJECT_ROOT / "checkpoints" / "beat_eisenstein_v5.pt"
    model = load_model(str(ckpt_path), device)

    # Test on several positions
    gs = GameState(win_length=WIN_LENGTH)

    # Play a few moves to get a non-trivial position
    moves_sequence = [
        HexCoord(0, 0),   # P1
        HexCoord(1, 0), HexCoord(1, -1),  # P2
        HexCoord(0, 1), HexCoord(0, -1),  # P1
        HexCoord(2, -1), HexCoord(1, 1),  # P2
    ]

    positions = [gs]
    for m in moves_sequence:
        gs = gs.apply_move(m)
        positions.append(gs)

    print(f"Testing {len(positions)} positions...\n")

    for i, pos in enumerate(positions):
        if pos.is_terminal:
            continue
        print(f"=== Position {i} (player {pos.current_player}, first_move={pos.is_first_move_of_turn}) ===")

        raw_result = raw_policy_top_moves(model, pos, device)
        if not raw_result:
            print("  No legal moves")
            continue
        raw_moves, raw_value = raw_result

        mcts_result = mcts_root_policy(model, pos, device)
        mcts_moves, mcts_value = mcts_result

        print(f"  Raw value: {raw_value:.4f},  MCTS value: {mcts_value:.4f}")
        print(f"  Raw top moves:")
        for move, prob, idx in raw_moves:
            print(f"    {move}  p={prob:.4f}  (flat_idx={idx})")

        print(f"  MCTS-extracted top moves:")
        for move, prob in mcts_moves:
            print(f"    {move}  p={prob:.4f}")

        # Check agreement
        raw_best = raw_moves[0][0]
        mcts_best = mcts_moves[0][0] if mcts_moves else None
        agree = raw_best == mcts_best
        print(f"  Top move agreement: {'YES' if agree else 'NO'} (raw={raw_best}, mcts={mcts_best})")
        print()


if __name__ == "__main__":
    main()
