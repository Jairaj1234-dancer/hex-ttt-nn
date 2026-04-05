#!/usr/bin/env python3
"""Comprehensive tactical neural network for 6-in-a-row hex: attack AND defense.

Builds on the existing hex-ttt-nn infrastructure to train a model that can:
  1. Extend chains toward 6-in-a-row (offensive)
  2. Block opponent chains before they reach 6 (defensive)
  3. Create forks / double-threats (advanced offense)
  4. Prioritize blocking the most dangerous threat (advanced defense)
  5. Decide between attacking and defending in race positions

Position types generated:
  - EXTEND:     Player has chain of N, correct moves extend it
  - BLOCK:      Opponent has chain of N, correct moves block ends
  - FORK:       Player has two chains sharing a common extension point
  - FORK_DEF:   Opponent has fork, player must block the longer chain
  - RACE:       Both sides have threatening chains, must decide attack vs defend

Training approach:
  - 3-phase curriculum (easy → medium → hard)
  - Weighted loss emphasizing policy (move selection) over value
  - Starts from best available checkpoint
  - Thorough per-category evaluation before and after

Usage:
    python tactical_nn.py                          # Full training from best checkpoint
    python tactical_nn.py --checkpoint path.pt     # Start from specific checkpoint
    python tactical_nn.py --eval-only              # Evaluate only, no training
    python tactical_nn.py --fresh                  # Train from scratch (no checkpoint)
"""

import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from game.hex_grid import HexCoord, HEX_AXES, axial_to_brick, brick_to_axial
from game.board import Board
from game.rules import GameState
from nn.features import extract_features
from nn.model import HexTTTNet

# =====================================================================
# Constants
# =====================================================================
WIN_LENGTH = 6
GRID_SIZE = 13

# Position type labels
POS_EXTEND = "extend"
POS_BLOCK = "block"
POS_FORK = "fork"
POS_FORK_DEF = "fork_defense"
POS_RACE = "race"

ALL_POS_TYPES = [POS_EXTEND, POS_BLOCK, POS_FORK, POS_FORK_DEF, POS_RACE]


# =====================================================================
# Utility: chain and board construction
# =====================================================================

def make_chain(start_q: int, start_r: int, axis: HexCoord, length: int) -> List[HexCoord]:
    """Create a chain of hex coordinates along an axis."""
    return [HexCoord(start_q + i * axis.q, start_r + i * axis.r) for i in range(length)]


def chain_ends(chain: List[HexCoord], axis: HexCoord) -> List[HexCoord]:
    """Return the two extension points at either end of a chain."""
    first, last = chain[0], chain[-1]
    return [
        HexCoord(first.q - axis.q, first.r - axis.r),
        HexCoord(last.q + axis.q, last.r + axis.r),
    ]


def add_noise_stones(
    board: Board, occupied: set, end_set: set,
    player: int, opponent: int,
    center_q: int, center_r: int, spread: int,
    count: int,
) -> Board:
    """Add random noise stones to make positions realistic."""
    for _ in range(count):
        for _ in range(50):
            nq = random.randint(center_q - spread, center_q + spread)
            nr = random.randint(center_r - spread, center_r + spread)
            if (nq, nr) not in occupied and (nq, nr) not in end_set:
                who = random.choice([player, opponent])
                board = board.place(HexCoord(nq, nr), who)
                occupied.add((nq, nr))
                break
    return board


def position_to_sample(
    gs: GameState, correct_moves: List[HexCoord], value: float,
    pos_type: str, grid_size: int = GRID_SIZE,
) -> Optional[dict]:
    """Convert a game state + correct moves into a training sample dict."""
    try:
        features, (cq, cr) = extract_features(gs, grid_size=grid_size)
    except Exception:
        return None

    policy = np.zeros(grid_size * grid_size, dtype=np.float32)
    valid_count = 0
    for move in correct_moves:
        if move in gs.board.stones:
            continue
        bx, by = axial_to_brick(move.q, move.r, cq, cr, grid_size)
        row_idx = by + grid_size // 2
        col_idx = bx + grid_size // 2
        if 0 <= row_idx < grid_size and 0 <= col_idx < grid_size:
            idx = row_idx * grid_size + col_idx
            policy[idx] = 1.0
            valid_count += 1

    if valid_count == 0:
        return None

    policy /= policy.sum()

    return {
        "features": features.numpy(),
        "policy": policy,
        "value": np.float32(value),
        "pos_type": pos_type,
    }


# =====================================================================
# Position generators
# =====================================================================

def gen_extend(chain_length: int) -> Optional[dict]:
    """Generate: player has chain, must extend it toward 6."""
    player = random.choice([1, 2])
    opponent = 3 - player
    axis = random.choice(HEX_AXES)
    sq, sr = random.randint(-3, 3), random.randint(-3, 3)

    chain = make_chain(sq, sr, axis, chain_length)
    ends = chain_ends(chain, axis)

    board = Board()
    occupied = set()
    for c in chain:
        board = board.place(c, player)
        occupied.add((c.q, c.r))

    end_set = {(e.q, e.r) for e in ends}
    noise = random.randint(4, 10)
    board = add_noise_stones(board, occupied, end_set, player, opponent, sq, sr, 6, noise)

    # Value: longer chains are more valuable
    if chain_length >= WIN_LENGTH - 1:
        value = 0.95
    else:
        value = 0.3 + 0.15 * chain_length

    gs = GameState(
        board=board, current_player=player,
        moves_remaining=2, win_length=WIN_LENGTH, turn_number=10,
    )
    return position_to_sample(gs, ends, value, POS_EXTEND)


def gen_block(chain_length: int) -> Optional[dict]:
    """Generate: opponent has chain, current player must block it."""
    player = random.choice([1, 2])
    opponent = 3 - player
    axis = random.choice(HEX_AXES)
    sq, sr = random.randint(-3, 3), random.randint(-3, 3)

    # Opponent's chain
    chain = make_chain(sq, sr, axis, chain_length)
    ends = chain_ends(chain, axis)

    board = Board()
    occupied = set()
    for c in chain:
        board = board.place(c, opponent)
        occupied.add((c.q, c.r))

    end_set = {(e.q, e.r) for e in ends}
    noise = random.randint(4, 10)
    board = add_noise_stones(board, occupied, end_set, player, opponent, sq, sr, 6, noise)

    # Value: must-block is urgent; longer enemy chains are worse
    if chain_length >= WIN_LENGTH - 1:
        value = -0.9  # critical — must block or lose
    else:
        value = -0.15 * chain_length

    gs = GameState(
        board=board, current_player=player,
        moves_remaining=2, win_length=WIN_LENGTH, turn_number=10,
    )
    return position_to_sample(gs, ends, value, POS_BLOCK)


def gen_fork() -> Optional[dict]:
    """Generate: player has two chains that share a common extension point (fork).

    A fork creates two threats simultaneously — the opponent can only block one,
    so the attacker wins. The correct move is the shared extension point.
    """
    player = random.choice([1, 2])
    opponent = 3 - player

    # Pick two different axes
    axes = random.sample(HEX_AXES, 2)
    # Pick a shared junction point
    jq, jr = random.randint(-2, 2), random.randint(-2, 2)
    junction = HexCoord(jq, jr)

    # Build two chains that END at the junction (so playing junction extends both)
    chain_len_1 = random.choice([3, 4])
    chain_len_2 = random.choice([3, 4])

    # Chain 1: goes backward from junction along axis1
    chain1 = [HexCoord(jq - (i + 1) * axes[0].q, jr - (i + 1) * axes[0].r)
              for i in range(chain_len_1)]
    # Chain 2: goes backward from junction along axis2
    chain2 = [HexCoord(jq - (i + 1) * axes[1].q, jr - (i + 1) * axes[1].r)
              for i in range(chain_len_2)]

    board = Board()
    occupied = set()

    # Place both chains
    for c in chain1 + chain2:
        if (c.q, c.r) in occupied:
            continue
        board = board.place(c, player)
        occupied.add((c.q, c.r))

    # Junction must be empty
    if (jq, jr) in occupied:
        return None

    end_set = {(jq, jr)}
    noise = random.randint(3, 8)
    board = add_noise_stones(board, occupied, end_set, player, opponent, jq, jr, 5, noise)

    # The correct move is the junction — it extends both chains
    correct_moves = [junction]
    value = 0.85 + 0.05 * max(chain_len_1, chain_len_2)

    gs = GameState(
        board=board, current_player=player,
        moves_remaining=2, win_length=WIN_LENGTH, turn_number=10,
    )
    return position_to_sample(gs, correct_moves, min(value, 0.99), POS_FORK)


def gen_fork_defense() -> Optional[dict]:
    """Generate: opponent has fork (two chains), player must block the longer one."""
    player = random.choice([1, 2])
    opponent = 3 - player

    axes = random.sample(HEX_AXES, 2)
    jq, jr = random.randint(-2, 2), random.randint(-2, 2)

    # Opponent's fork: two chains meeting near junction
    chain_len_1 = random.choice([4, 5])
    chain_len_2 = random.choice([3, 4])

    # Make chain1 longer (more dangerous) — player should prioritize blocking it
    if chain_len_2 > chain_len_1:
        chain_len_1, chain_len_2 = chain_len_2, chain_len_1

    # Chain 1: along axis1 from a start point
    sq1, sr1 = jq + axes[0].q, jr + axes[0].r
    chain1 = make_chain(sq1, sr1, axes[0], chain_len_1)
    ends1 = chain_ends(chain1, axes[0])

    # Chain 2: along axis2 from a different start
    sq2, sr2 = jq + axes[1].q, jr + axes[1].r
    chain2 = make_chain(sq2, sr2, axes[1], chain_len_2)
    ends2 = chain_ends(chain2, axes[1])

    board = Board()
    occupied = set()
    all_end_set = set()

    for c in chain1 + chain2:
        if (c.q, c.r) in occupied:
            continue
        board = board.place(c, opponent)
        occupied.add((c.q, c.r))

    for e in ends1 + ends2:
        all_end_set.add((e.q, e.r))

    noise = random.randint(3, 8)
    board = add_noise_stones(board, occupied, all_end_set, player, opponent, jq, jr, 6, noise)

    # Correct moves: block the longer (more dangerous) chain's ends
    correct_moves = [e for e in ends1 if (e.q, e.r) not in occupied]
    if not correct_moves:
        return None

    value = -0.7 - 0.05 * chain_len_1  # bad position but still fightable

    gs = GameState(
        board=board, current_player=player,
        moves_remaining=2, win_length=WIN_LENGTH, turn_number=10,
    )
    return position_to_sample(gs, correct_moves, max(value, -0.99), POS_FORK_DEF)


def gen_race() -> Optional[dict]:
    """Generate: both sides have chains — player must choose attack or defense.

    If player's chain is longer or equal → attack (extend own chain).
    If opponent's chain is longer → defend (block opponent's chain).
    """
    player = random.choice([1, 2])
    opponent = 3 - player

    # Two different axes for the two chains (avoids overlap)
    axes = random.sample(HEX_AXES, 2)

    player_len = random.choice([3, 4, 5])
    opp_len = random.choice([3, 4, 5])

    # Player's chain
    sq1, sr1 = random.randint(-3, 0), random.randint(-3, 0)
    p_chain = make_chain(sq1, sr1, axes[0], player_len)
    p_ends = chain_ends(p_chain, axes[0])

    # Opponent's chain (offset to avoid overlap)
    sq2, sr2 = random.randint(1, 4), random.randint(1, 4)
    o_chain = make_chain(sq2, sr2, axes[1], opp_len)
    o_ends = chain_ends(o_chain, axes[1])

    board = Board()
    occupied = set()

    for c in p_chain:
        if (c.q, c.r) not in occupied:
            board = board.place(c, player)
            occupied.add((c.q, c.r))

    for c in o_chain:
        if (c.q, c.r) not in occupied:
            board = board.place(c, opponent)
            occupied.add((c.q, c.r))

    all_end_set = {(e.q, e.r) for e in p_ends + o_ends}
    noise = random.randint(2, 6)
    center_q = (sq1 + sq2) // 2
    center_r = (sr1 + sr2) // 2
    board = add_noise_stones(board, occupied, all_end_set, player, opponent,
                             center_q, center_r, 7, noise)

    # Decision logic:
    # - If opponent has 5 → MUST block (they win next move)
    # - If player has 5 → MUST extend (we win)
    # - If opponent chain >= player chain → block (defensive priority)
    # - If player chain > opponent chain → extend (offensive)
    if opp_len >= WIN_LENGTH - 1:
        correct_moves = o_ends  # MUST block
        value = -0.5
    elif player_len >= WIN_LENGTH - 1:
        correct_moves = p_ends  # WIN by extending
        value = 0.9
    elif opp_len >= player_len:
        correct_moves = o_ends  # defend — block the longer threat
        value = -0.1 * (opp_len - player_len) - 0.1
    else:
        correct_moves = p_ends  # attack — extend our advantage
        value = 0.1 * (player_len - opp_len) + 0.1

    gs = GameState(
        board=board, current_player=player,
        moves_remaining=2, win_length=WIN_LENGTH, turn_number=10,
    )
    return position_to_sample(gs, correct_moves, value, POS_RACE)


# =====================================================================
# Dataset generation
# =====================================================================

GENERATORS = {
    POS_EXTEND: gen_extend,
    POS_BLOCK: gen_block,
    POS_FORK: gen_fork,
    POS_FORK_DEF: gen_fork_defense,
    POS_RACE: gen_race,
}


def generate_dataset(
    num_positions: int,
    mix: Dict[str, float],
    chain_lengths: Optional[Dict[str, List[int]]] = None,
) -> List[dict]:
    """Generate a mixed tactical dataset.

    Args:
        num_positions: total positions to generate
        mix: dict mapping pos_type -> fraction (should sum to ~1.0)
        chain_lengths: optional chain length ranges per type
    """
    if chain_lengths is None:
        chain_lengths = {
            POS_EXTEND: [3, 4, 5],
            POS_BLOCK: [3, 4, 5],
        }

    dataset = []
    counts = {t: int(num_positions * frac) for t, frac in mix.items()}
    # Distribute remainder
    remainder = num_positions - sum(counts.values())
    if remainder > 0:
        top_type = max(mix, key=mix.get)
        counts[top_type] += remainder

    for pos_type, target_count in counts.items():
        gen_fn = GENERATORS[pos_type]
        generated = 0
        attempts = 0
        max_attempts = target_count * 5

        while generated < target_count and attempts < max_attempts:
            attempts += 1
            if pos_type in (POS_EXTEND, POS_BLOCK):
                cl_options = chain_lengths.get(pos_type, [3, 4, 5])
                # Weight toward longer (more critical) chains
                weights = [1.0 + 0.5 * i for i in range(len(cl_options))]
                cl = random.choices(cl_options, weights=weights, k=1)[0]
                sample = gen_fn(cl)
            else:
                sample = gen_fn()

            if sample is not None:
                dataset.append(sample)
                generated += 1

    random.shuffle(dataset)
    return dataset


# =====================================================================
# Training
# =====================================================================

def train_phase(
    model: HexTTTNet,
    dataset: List[dict],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    policy_weight: float = 2.0,
    value_weight: float = 1.0,
    phase_name: str = "",
) -> Dict[str, List[float]]:
    """Train on a dataset for some epochs. Returns loss history."""
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 20
    )

    history = {"loss": [], "pol_loss": [], "val_loss": [], "accuracy": []}

    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0.0
        pol_sum = 0.0
        val_sum = 0.0
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

            loss = policy_weight * policy_loss + value_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pol_sum += policy_loss.item()
            val_sum += value_loss.item()
            num_batches += 1

            pred_moves = out["policy"].argmax(dim=1)
            target_moves = policies.argmax(dim=1)
            correct += (pred_moves == target_moves).sum().item()
            total += len(batch)

        scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_pol = pol_sum / max(num_batches, 1)
        avg_val = val_sum / max(num_batches, 1)
        acc = 100.0 * correct / max(total, 1)

        history["loss"].append(avg_loss)
        history["pol_loss"].append(avg_pol)
        history["val_loss"].append(avg_val)
        history["accuracy"].append(acc)

        print(
            f"  [{phase_name}] Epoch {epoch + 1:2d}/{epochs} | "
            f"loss={avg_loss:.4f} (pol={avg_pol:.4f} val={avg_val:.4f}) | "
            f"acc={acc:.1f}% | lr={scheduler.get_last_lr()[0]:.6f}"
        )

    return history


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_model(
    model: HexTTTNet, device: str, num_per_category: int = 200
) -> Dict[str, Dict[str, float]]:
    """Evaluate model across all tactical categories.

    Returns nested dict: {pos_type: {chain_len: accuracy, ...}, ...}
    """
    model.eval()
    model.to(device)

    results = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    # ---- Chain extend/block by chain length ----
    for chain_len in [3, 4, 5]:
        for pos_type in [POS_EXTEND, POS_BLOCK]:
            gen_fn = GENERATORS[pos_type]
            for _ in range(num_per_category):
                sample = gen_fn(chain_len)
                if sample is None:
                    continue
                correct = _eval_single(model, sample, device)
                key = f"{pos_type}_{chain_len}"
                results[pos_type][key]["total"] += 1
                if correct:
                    results[pos_type][key]["correct"] += 1

    # ---- Fork / fork_defense / race ----
    for pos_type in [POS_FORK, POS_FORK_DEF, POS_RACE]:
        gen_fn = GENERATORS[pos_type]
        for _ in range(num_per_category):
            sample = gen_fn()
            if sample is None:
                continue
            correct = _eval_single(model, sample, device)
            results[pos_type][pos_type]["total"] += 1
            if correct:
                results[pos_type][pos_type]["correct"] += 1

    return results


def _eval_single(model: HexTTTNet, sample: dict, device: str) -> bool:
    """Check if model's top move matches any correct move in the sample."""
    features = torch.tensor(sample["features"], dtype=torch.float32).unsqueeze(0).to(device)
    policy_target = sample["policy"]
    mask = torch.ones(1, GRID_SIZE * GRID_SIZE, device=device)

    with torch.no_grad():
        out = model(features, valid_moves_mask=mask)

    pred_idx = out["policy"][0].argmax().item()
    # Check if predicted move is one of the correct moves (nonzero in target)
    return policy_target[pred_idx] > 0.0


def print_evaluation(results: Dict, title: str = "Evaluation"):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    attack_correct = 0
    attack_total = 0
    defense_correct = 0
    defense_total = 0

    for pos_type in ALL_POS_TYPES:
        if pos_type not in results:
            continue
        print(f"\n  [{pos_type.upper()}]")
        for key, counts in sorted(results[pos_type].items()):
            total = counts["total"]
            correct = counts["correct"]
            pct = 100.0 * correct / max(total, 1)
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            print(f"    {key:20s}: {correct:3d}/{total:3d} = {pct:5.1f}% [{bar}]")

            # Aggregate attack vs defense
            if pos_type in (POS_EXTEND, POS_FORK):
                attack_correct += correct
                attack_total += total
            elif pos_type in (POS_BLOCK, POS_FORK_DEF):
                defense_correct += correct
                defense_total += total
            else:  # race — mixed
                attack_correct += correct // 2
                attack_total += total // 2
                defense_correct += correct - correct // 2
                defense_total += total - total // 2

    print(f"\n  {'─' * 56}")
    overall_total = attack_total + defense_total
    overall_correct = attack_correct + defense_correct
    overall_pct = 100.0 * overall_correct / max(overall_total, 1)
    atk_pct = 100.0 * attack_correct / max(attack_total, 1)
    def_pct = 100.0 * defense_correct / max(defense_total, 1)

    print(f"  ATTACK accuracy:  {attack_correct}/{attack_total} = {atk_pct:.1f}%")
    print(f"  DEFENSE accuracy: {defense_correct}/{defense_total} = {def_pct:.1f}%")
    print(f"  OVERALL accuracy: {overall_correct}/{overall_total} = {overall_pct:.1f}%")
    print(f"{'=' * 60}")

    return {
        "attack_pct": atk_pct,
        "defense_pct": def_pct,
        "overall_pct": overall_pct,
    }


# =====================================================================
# Curriculum training pipeline
# =====================================================================

def run_curriculum_training(
    model: HexTTTNet,
    device: str,
    total_positions: int = 30000,
    batch_size: int = 128,
) -> HexTTTNet:
    """3-phase curriculum: easy → medium → hard tactical positions.

    Phase 1 (Foundations): Long chains (4-5), simple extend/block
    Phase 2 (Tactics):     All chain lengths + forks
    Phase 3 (Strategy):    Full mix including races and fork defense
    """

    # ── Phase 1: Foundations ──────────────────────────────────────
    print("\n" + "━" * 60)
    print("  PHASE 1: Foundations — Long chain extend & block")
    print("━" * 60)

    n1 = total_positions // 3
    ds1 = generate_dataset(
        n1,
        mix={POS_EXTEND: 0.50, POS_BLOCK: 0.50},
        chain_lengths={POS_EXTEND: [4, 5], POS_BLOCK: [4, 5]},
    )
    print(f"  Generated {len(ds1)} phase-1 positions")

    train_phase(
        model, ds1, device,
        epochs=15, batch_size=batch_size, lr=3e-4,
        policy_weight=2.0, value_weight=0.5,
        phase_name="Phase 1",
    )

    # ── Phase 2: Tactics ──────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  PHASE 2: Tactics — All chains + forks")
    print("━" * 60)

    n2 = total_positions // 3
    ds2 = generate_dataset(
        n2,
        mix={POS_EXTEND: 0.30, POS_BLOCK: 0.30, POS_FORK: 0.20, POS_FORK_DEF: 0.20},
        chain_lengths={POS_EXTEND: [3, 4, 5], POS_BLOCK: [3, 4, 5]},
    )
    print(f"  Generated {len(ds2)} phase-2 positions")

    train_phase(
        model, ds2, device,
        epochs=20, batch_size=batch_size, lr=2e-4,
        policy_weight=2.5, value_weight=0.8,
        phase_name="Phase 2",
    )

    # ── Phase 3: Strategy ─────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  PHASE 3: Strategy — Full mix with races")
    print("━" * 60)

    n3 = total_positions - n1 - n2
    ds3 = generate_dataset(
        n3,
        mix={
            POS_EXTEND: 0.20, POS_BLOCK: 0.20,
            POS_FORK: 0.15, POS_FORK_DEF: 0.15,
            POS_RACE: 0.30,
        },
        chain_lengths={POS_EXTEND: [3, 4, 5], POS_BLOCK: [3, 4, 5]},
    )
    print(f"  Generated {len(ds3)} phase-3 positions")

    train_phase(
        model, ds3, device,
        epochs=25, batch_size=batch_size, lr=1e-4,
        policy_weight=3.0, value_weight=1.0,
        phase_name="Phase 3",
    )

    return model


# =====================================================================
# Checkpoint management
# =====================================================================

def find_best_checkpoint() -> Optional[Path]:
    """Find the best available checkpoint to start from."""
    candidates = [
        "checkpoints/tactical_w6.pt",
        "checkpoints/distilled_w6_v2.pt",
        "checkpoints/distilled_w6_blend.pt",
        "checkpoints/distilled_w6.pt",
        "checkpoints/best.pt",
        "checkpoints/latest.pt",
    ]
    for name in candidates:
        path = PROJECT_ROOT / name
        if path.exists():
            return path
    return None


def load_model(checkpoint_path: Optional[Path] = None) -> HexTTTNet:
    """Create and optionally load a HexTTTNet."""
    model = HexTTTNet(
        grid_size=GRID_SIZE, num_blocks=6, channels=96, in_channels=12
    )
    if checkpoint_path and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"  Loaded checkpoint: {checkpoint_path.name}")
    else:
        print("  Starting from scratch (random weights)")
    return model


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Tactical NN trainer: 6-in-a-row attack & defense"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: auto-detect best)")
    parser.add_argument("--output", type=str, default="checkpoints/tactical_nn_v1.pt",
                        help="Output checkpoint path")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device: cpu, cuda, mps")
    parser.add_argument("--total-positions", type=int, default=30000,
                        help="Total training positions across all phases")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate, no training")
    parser.add_argument("--fresh", action="store_true",
                        help="Train from scratch (no checkpoint)")
    parser.add_argument("--eval-samples", type=int, default=200,
                        help="Samples per evaluation category")
    args = parser.parse_args()

    device = args.device
    output_path = PROJECT_ROOT / args.output

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Tactical NN: 6-in-a-Row Attack & Defense Trainer      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Device: {device}")
    print(f"  Grid size: {GRID_SIZE}")
    print(f"  Win length: {WIN_LENGTH}")

    # Load model
    if args.fresh:
        ckpt_path = None
    elif args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_best_checkpoint()

    model = load_model(ckpt_path)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Pre-training evaluation
    print("\n  Evaluating baseline performance...")
    t0 = time.time()
    results_before = evaluate_model(model, device, num_per_category=args.eval_samples)
    summary_before = print_evaluation(results_before, "BEFORE Training")
    print(f"  Evaluation took {time.time() - t0:.1f}s")

    if args.eval_only:
        return

    # Curriculum training
    print(f"\n  Starting curriculum training ({args.total_positions} positions)...")
    t_train = time.time()
    model = run_curriculum_training(
        model, device,
        total_positions=args.total_positions,
        batch_size=args.batch_size,
    )
    train_time = time.time() - t_train
    print(f"\n  Training completed in {train_time:.1f}s ({train_time / 60:.1f} min)")

    # Post-training evaluation
    print("\n  Evaluating trained model...")
    results_after = evaluate_model(model, device, num_per_category=args.eval_samples)
    summary_after = print_evaluation(results_after, "AFTER Training")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "network": {
                "grid_size": GRID_SIZE,
                "num_blocks": 6,
                "channels": 96,
                "in_channels": 12,
            },
            "win_length": WIN_LENGTH,
        },
        "eval_before": summary_before,
        "eval_after": summary_after,
        "training_time_seconds": train_time,
        "source_checkpoint": str(ckpt_path) if ckpt_path else "fresh",
    }, output_path)
    print(f"\n  Saved to: {output_path}")

    # Improvement summary
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                   IMPROVEMENT SUMMARY                    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for metric in ["attack_pct", "defense_pct", "overall_pct"]:
        before = summary_before[metric]
        after = summary_after[metric]
        delta = after - before
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "─"
        label = metric.replace("_pct", "").upper()
        print(f"║  {label:10s}: {before:5.1f}% → {after:5.1f}%  ({arrow} {abs(delta):+.1f}%)  ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
