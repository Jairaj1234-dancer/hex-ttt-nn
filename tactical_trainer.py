#!/usr/bin/env python3
"""Tactical training: teach the model to build and block 6-in-a-row chains.

Generates focused training positions where:
  1. One side has a chain of length N (3-5) — correct move extends it
  2. One side has a chain the opponent must block — correct move blocks
  3. Both sides have competing chains — correct move balances attack/defense

Trains the model specifically on these tactical patterns to fix the
core weakness: the model doesn't understand chain building/blocking.
"""

import math
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from game.hex_grid import HexCoord, axial_to_brick
from game.rules import GameState
from nn.features import extract_features
from nn.model import HexTTTNet

# =====================================================================
# HEX AXIS DEFINITIONS
# =====================================================================
# Three axes on hex grid (axial coordinates):
#   Axis 0: (dq=1, dr=0)   — along q
#   Axis 1: (dq=0, dr=1)   — along r
#   Axis 2: (dq=1, dr=-1)  — diagonal (constant s=q+r)
HEX_AXES = [(1, 0), (0, 1), (1, -1)]


def _make_chain(start_q: int, start_r: int, dq: int, dr: int, length: int) -> List[HexCoord]:
    """Create a chain of hex coordinates along an axis."""
    return [HexCoord(start_q + i * dq, start_r + i * dr) for i in range(length)]


def _chain_ends(chain: List[HexCoord], dq: int, dr: int) -> List[HexCoord]:
    """Return the two extension points at either end of a chain."""
    first = chain[0]
    last = chain[-1]
    return [
        HexCoord(first.q - dq, first.r - dr),  # extend before start
        HexCoord(last.q + dq, last.r + dr),     # extend after end
    ]


def generate_chain_position(
    chain_length: int = 5,
    player: int = 1,
    win_length: int = 6,
    noise_stones: int = 6,
) -> Tuple[GameState, List[HexCoord], float]:
    """Generate a position with a chain and the correct extension/block moves.

    Args:
        chain_length: length of the main chain (3-5)
        player: which player owns the chain (1 or 2)
        win_length: game's win condition
        noise_stones: random opponent stones to add for realism

    Returns:
        (game_state, correct_moves, value_for_current_player)
    """
    # Pick random axis and starting position
    dq, dr = random.choice(HEX_AXES)
    # Center the chain near origin
    start_q = random.randint(-3, 3)
    start_r = random.randint(-3, 3)

    chain = _make_chain(start_q, start_r, dq, dr, chain_length)
    ends = _chain_ends(chain, dq, dr)

    # Build game state manually by placing stones
    gs = GameState(win_length=win_length)
    placed_coords = set()

    # We need to carefully construct the game state
    # The simplest approach: place stones directly on the board
    # and create a game state that has the right turn structure

    # Place chain stones for `player`
    chain_set = set((c.q, c.r) for c in chain)
    occupied = set()

    # Place dummy stones to reach the right game state
    # Strategy: alternate placing chain stones and noise stones
    # until all chain stones are placed

    opponent = 3 - player

    # We'll build a sequence of moves that results in the desired position
    # To place N stones for player and M for opponent, we need specific move ordering
    # Since first turn = 1 stone, subsequent turns = 2 stones each

    # Simpler approach: use board manipulation directly
    from game.board import Board
    board = Board()

    # Place chain
    for coord in chain:
        board = board.place(coord, player)
        occupied.add((coord.q, coord.r))

    # Place noise stones for opponent (scattered, not forming long chains)
    noise_placed = 0
    attempts = 0
    while noise_placed < noise_stones and attempts < 200:
        nq = random.randint(start_q - 5, start_q + chain_length + 5)
        nr = random.randint(start_r - 5, start_r + chain_length + 5)
        if (nq, nr) not in occupied and not any((nq, nr) == (e.q, e.r) for e in ends):
            board = board.place(HexCoord(nq, nr), opponent)
            occupied.add((nq, nr))
            noise_placed += 1
        attempts += 1

    # Place some noise stones for the chain player too (so it's realistic)
    for _ in range(max(0, noise_stones - 2)):
        attempts = 0
        while attempts < 100:
            nq = random.randint(start_q - 5, start_q + chain_length + 5)
            nr = random.randint(start_r - 5, start_r + chain_length + 5)
            if (nq, nr) not in occupied and not any((nq, nr) == (e.q, e.r) for e in ends):
                board = board.place(HexCoord(nq, nr), player)
                occupied.add((nq, nr))
                break
            attempts += 1

    # Determine current player and correct moves
    is_attack = random.random() < 0.5
    if is_attack:
        current_player = player  # chain owner's turn → extend
        correct_moves = ends
        if chain_length >= win_length - 1:
            value = 1.0  # winning position
        else:
            value = 0.5 + 0.1 * chain_length
    else:
        current_player = opponent  # opponent's turn → must block
        correct_moves = ends
        if chain_length >= win_length - 1:
            value = -0.9  # nearly lost
        else:
            value = -0.1 * chain_length

    # Construct GameState with the right board and turn
    gs = GameState(
        board=board,
        current_player=current_player,
        moves_remaining=2,  # normal turn: 2 sub-moves
        win_length=win_length,
        turn_number=10,  # arbitrary non-first turn
    )

    return gs, correct_moves, value


def generate_tactical_dataset(
    num_positions: int = 10000,
    win_length: int = 6,
    grid_size: int = 13,
) -> List[dict]:
    """Generate a dataset of tactical positions.

    Uses direct board construction for speed and reliability.
    Returns list of dicts with keys: features, policy, value
    """
    from game.board import Board

    dataset = []
    chain_lengths = [3, 4, 5]
    weights = [0.15, 0.35, 0.50]  # Focus on 5-chains (most critical)
    max_attempts = num_positions * 3  # allow some failures
    attempts = 0

    while len(dataset) < num_positions and attempts < max_attempts:
        attempts += 1
        cl = random.choices(chain_lengths, weights=weights, k=1)[0]
        player = random.choice([1, 2])
        opponent = 3 - player
        dq, dr = random.choice(HEX_AXES)
        # Random start near origin
        sq = random.randint(-3, 3)
        sr = random.randint(-3, 3)

        chain = _make_chain(sq, sr, dq, dr, cl)
        ends = _chain_ends(chain, dq, dr)

        # Build board directly
        board = Board()
        occupied = set()
        for c in chain:
            board = board.place(c, player)
            occupied.add((c.q, c.r))

        # Keep ends clear
        end_set = {(e.q, e.r) for e in ends}

        # Add noise stones
        noise_count = random.randint(4, 12)
        for _ in range(noise_count):
            for _ in range(30):
                nq = random.randint(sq - 5, sq + cl + 5)
                nr = random.randint(sr - 5, sr + cl + 5)
                if (nq, nr) not in occupied and (nq, nr) not in end_set:
                    board = board.place(HexCoord(nq, nr), random.choice([player, opponent]))
                    occupied.add((nq, nr))
                    break

        # Determine scenario: attack (extend) or defense (block)
        is_attack = random.random() < 0.5
        current = player if is_attack else opponent

        if is_attack:
            if cl >= win_length - 1:
                value = 1.0
            else:
                value = 0.3 + 0.15 * cl
        else:
            if cl >= win_length - 1:
                value = -0.9
            else:
                value = -0.15 * cl

        gs = GameState(
            board=board, current_player=current,
            moves_remaining=2, win_length=win_length, turn_number=10,
        )

        # Extract features
        try:
            features, (cq, cr) = extract_features(gs, grid_size=grid_size)
        except Exception:
            continue

        # Create policy target on valid chain ends
        policy = np.zeros(grid_size * grid_size, dtype=np.float32)
        valid_count = 0
        for move in ends:
            if move in gs.board.stones:
                continue  # end is occupied
            bx, by = axial_to_brick(move.q, move.r, cq, cr, grid_size)
            if 0 <= bx < grid_size and 0 <= by < grid_size:
                idx = by * grid_size + bx
                policy[idx] = 1.0
                valid_count += 1

        if valid_count == 0:
            continue

        policy /= policy.sum()

        dataset.append({
            "features": features.numpy(),
            "policy": policy,
            "value": value,
        })

    return dataset


def train_tactical(
    model: HexTTTNet,
    dataset: List[dict],
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.0003,
    grid_size: int = 13,
) -> HexTTTNet:
    """Train the model on tactical positions.

    Uses high policy weight and focused value targets to teach
    chain building and blocking.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 20
    )

    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0.0
        pol_sum = 0.0
        val_sum = 0.0
        correct_count = 0
        total = 0
        num_batches = 0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
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
            mask = torch.ones(len(batch), grid_size * grid_size).to(device)

            out = model(features, valid_moves_mask=mask)

            # Policy loss: cross-entropy with focused targets
            log_probs = F.log_softmax(out["policy_logits"], dim=1)
            policy_loss = -(policies * log_probs).sum(dim=1).mean()

            # Value loss: MSE with tactical values
            value_loss = F.mse_loss(out["value"], values)

            # Emphasize policy (chain moves) over value
            loss = 2.0 * policy_loss + 1.0 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pol_sum += policy_loss.item()
            val_sum += value_loss.item()
            num_batches += 1

            # Track top-1 accuracy
            pred_moves = out["policy"].argmax(dim=1)
            target_moves = policies.argmax(dim=1)
            correct_count += (pred_moves == target_moves).sum().item()
            total += len(batch)

        scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_pol = pol_sum / max(num_batches, 1)
        avg_val = val_sum / max(num_batches, 1)
        accuracy = 100.0 * correct_count / max(total, 1)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | loss={avg_loss:.4f} "
            f"(pol={avg_pol:.4f} val={avg_val:.4f}) | "
            f"accuracy={accuracy:.1f}% | lr={scheduler.get_last_lr()[0]:.6f}"
        )

    return model


def evaluate_tactical(model: HexTTTNet, device: str = "cpu", grid_size: int = 13) -> dict:
    """Evaluate model's tactical ability on fresh test positions."""
    model.eval()
    model.to(device)

    results = {"extend_5": 0, "block_5": 0, "extend_4": 0, "block_4": 0}
    totals = {"extend_5": 0, "block_5": 0, "extend_4": 0, "block_4": 0}

    for chain_len in [4, 5]:
        for is_attack in [True, False]:
            mode = f"{'extend' if is_attack else 'block'}_{chain_len}"
            for _ in range(100):
                player = random.choice([1, 2])
                opponent = 3 - player
                dq, dr = random.choice(HEX_AXES)
                sq, sr = random.randint(-2, 2), random.randint(-2, 2)
                chain = _make_chain(sq, sr, dq, dr, chain_len)
                ends = _chain_ends(chain, dq, dr)

                from game.board import Board
                board = Board()
                occupied = set()
                for c in chain:
                    board = board.place(c, player)
                    occupied.add((c.q, c.r))
                # Add some noise
                for _ in range(5):
                    for _ in range(50):
                        nq = random.randint(sq - 4, sq + chain_len + 4)
                        nr = random.randint(sr - 4, sr + chain_len + 4)
                        if (nq, nr) not in occupied and not any((nq, nr) == (e.q, e.r) for e in ends):
                            board = board.place(HexCoord(nq, nr), opponent)
                            occupied.add((nq, nr))
                            break

                current = player if is_attack else opponent
                gs = GameState(
                    board=board, current_player=current,
                    moves_remaining=2, win_length=6, turn_number=10,
                )

                try:
                    features, (cq, cr) = extract_features(gs, grid_size=grid_size)
                except Exception:
                    continue

                mask = torch.zeros(1, grid_size * grid_size)
                legal_map = {}
                legal = gs.legal_moves(zoi_margin=3)
                for m in legal:
                    bx, by = axial_to_brick(m.q, m.r, cq, cr, grid_size)
                    if 0 <= bx < grid_size and 0 <= by < grid_size:
                        idx = by * grid_size + bx
                        mask[0, idx] = 1.0
                        legal_map[idx] = m

                with torch.no_grad():
                    out = model(
                        features.unsqueeze(0).to(device),
                        valid_moves_mask=mask.to(device),
                    )

                policy = out["policy"][0].cpu().numpy()
                chosen_idx = max(legal_map.keys(), key=lambda i: policy[i])
                chosen_move = legal_map[chosen_idx]

                # Check if chosen move is one of the correct chain ends
                correct = any(
                    chosen_move.q == e.q and chosen_move.r == e.r
                    for e in ends
                    if e not in board.stones
                )

                totals[mode] += 1
                if correct:
                    results[mode] += 1

    print("\n=== Tactical Evaluation ===")
    for mode in results:
        total = totals[mode]
        correct = results[mode]
        pct = 100.0 * correct / max(total, 1)
        print(f"  {mode}: {correct}/{total} = {pct:.1f}%")

    return {k: results[k] / max(totals[k], 1) for k in results}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tactical chain training")
    parser.add_argument("--checkpoint", default="checkpoints/distilled_w6.pt")
    parser.add_argument("--output", default="checkpoints/tactical_w6.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-positions", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    grid_size = 13
    device = args.device

    # Load model
    model = HexTTTNet(grid_size=grid_size, num_blocks=6, channels=96, in_channels=12)
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint, starting fresh")

    # Evaluate before training
    print("\n--- Before tactical training ---")
    eval_before = evaluate_tactical(model, device=device, grid_size=grid_size)

    if args.eval_only:
        return

    # Generate tactical dataset
    print(f"\nGenerating {args.num_positions} tactical positions...")
    t0 = time.time()
    dataset = generate_tactical_dataset(
        num_positions=args.num_positions, win_length=6, grid_size=grid_size
    )
    print(f"Generated {len(dataset)} positions in {time.time() - t0:.1f}s")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    model = train_tactical(
        model, dataset, device=device,
        epochs=args.epochs, lr=args.lr, grid_size=grid_size,
    )

    # Evaluate after training
    print("\n--- After tactical training ---")
    eval_after = evaluate_tactical(model, device=device, grid_size=grid_size)

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"network": {"grid_size": grid_size, "num_blocks": 6, "channels": 96, "in_channels": 12}},
        "tactical_eval": eval_after,
    }, args.output)
    print(f"\nSaved tactical model to {args.output}")

    # Print improvement
    print("\n=== Improvement ===")
    for mode in eval_before:
        before = eval_before[mode] * 100
        after = eval_after[mode] * 100
        print(f"  {mode}: {before:.1f}% → {after:.1f}% ({after - before:+.1f}%)")


if __name__ == "__main__":
    main()
