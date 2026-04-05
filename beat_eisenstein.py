#!/usr/bin/env python3
"""Train the NN to beat the EisensteinGreedy agent.

Strategy:
  Phase 1 — IMITATE: Supervised learning from Eisenstein game positions.
      Gives the NN baseline competence (threat detection, chain building).
  Phase 2 — EXPLOIT: Mix in fork/double-threat positions that Eisenstein
      cannot handle (it only sees 1-ply). The NN learns to create setups
      that produce unstoppable double threats.
  Phase 3 — REFINE: Play actual games NN vs Eisenstein. From losses, use
      Eisenstein's moves as corrective targets. From wins, reinforce.
      Iterate until win rate exceeds target.

The core insight: Eisenstein is a 1-ply heuristic that scores moves by
chain length. It cannot see forks coming. A network that creates setups
leading to two simultaneous 5-chains will always win, because Eisenstein
can only block one at a time.

Usage:
    python beat_eisenstein.py                        # Full pipeline
    python beat_eisenstein.py --device mps           # Use Apple Silicon GPU
    python beat_eisenstein.py --skip-imitate          # Skip phase 1
    python beat_eisenstein.py --target-winrate 0.80   # Custom target
"""

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
from tournament import EisensteinGreedyAgent, OnePlyAgent, GreedyAgent, RandomAgent

# =====================================================================
# Constants
# =====================================================================
WIN_LENGTH = 6
GRID_SIZE = 13
ZOI_MARGIN = 3
MAX_GAME_MOVES = 120  # cap per game to avoid infinite draws


# =====================================================================
# Raw-policy NN move selection (no MCTS — fast)
# =====================================================================

def nn_get_move(model: HexTTTNet, game_state: GameState, device: str) -> HexCoord:
    """Pick the highest-probability legal move using raw policy (no MCTS)."""
    features, (cq, cr) = extract_features(game_state, grid_size=GRID_SIZE)
    legal = game_state.legal_moves(zoi_margin=ZOI_MARGIN)
    if not legal:
        return HexCoord(0, 0)

    half = GRID_SIZE // 2
    mask = torch.zeros(GRID_SIZE * GRID_SIZE, dtype=torch.float32)
    legal_map = {}  # flat_idx -> HexCoord

    for move in legal:
        bx, by = axial_to_brick(move.q, move.r, cq, cr, GRID_SIZE)
        row = by + half
        col = bx + half
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


# =====================================================================
# Game playing infrastructure
# =====================================================================

def play_game(
    p1_fn, p2_fn, win_length: int = WIN_LENGTH, max_moves: int = MAX_GAME_MOVES,
) -> Tuple[int, List[dict]]:
    """Play one game. Returns (winner, position_records).

    Each position_record: {game_state, move_played, current_player}
    """
    gs = GameState(win_length=win_length)
    records = []
    move_count = 0

    while not gs.is_terminal and move_count < max_moves:
        agent_fn = p1_fn if gs.current_player == 1 else p2_fn
        move = agent_fn(gs)
        records.append({
            "game_state": gs,
            "move_played": move,
            "current_player": gs.current_player,
        })
        gs = gs.apply_move(move)
        move_count += 1

    winner = gs.winner if gs.winner is not None else 0
    return winner, records


def benchmark_vs_eisenstein(
    model: HexTTTNet, device: str, num_games: int = 40,
) -> Dict[str, float]:
    """Play NN (raw policy) vs Eisenstein, return stats."""
    model.to(device)
    model.eval()
    eisenstein = EisensteinGreedyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN)

    nn_wins = 0
    eis_wins = 0
    draws = 0

    def nn_fn(gs):
        return nn_get_move(model, gs, device)

    for g in range(num_games):
        # Alternate sides
        if g % 2 == 0:
            winner, _ = play_game(nn_fn, eisenstein.get_move)
            if winner == 1:
                nn_wins += 1
            elif winner == 2:
                eis_wins += 1
            else:
                draws += 1
        else:
            winner, _ = play_game(eisenstein.get_move, nn_fn)
            if winner == 2:
                nn_wins += 1
            elif winner == 1:
                eis_wins += 1
            else:
                draws += 1

    total = num_games
    win_rate = nn_wins / total
    return {
        "nn_wins": nn_wins,
        "eis_wins": eis_wins,
        "draws": draws,
        "win_rate": win_rate,
        "total": total,
    }


# =====================================================================
# Phase 1: Supervised imitation of Eisenstein
# =====================================================================

def generate_expert_data(
    num_games: int = 3000, max_moves: int = 60,
) -> List[dict]:
    """Generate supervised data from Eisenstein vs Eisenstein/OnePly games."""
    agents = [
        EisensteinGreedyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN, defensive=True),
        OnePlyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN),
        GreedyAgent(zoi_margin=ZOI_MARGIN),
    ]

    all_samples = []

    for game_idx in range(num_games):
        a1 = random.choice(agents)
        a2 = random.choice(agents)

        gs = GameState(win_length=WIN_LENGTH)
        positions = []
        half_move = 0

        while not gs.is_terminal and half_move < max_moves:
            features, (cq, cr) = extract_features(gs, grid_size=GRID_SIZE)
            agent = a1 if gs.current_player == 1 else a2
            move = agent.get_move(gs)

            bx, by = axial_to_brick(move.q, move.r, cq, cr, GRID_SIZE)
            half = GRID_SIZE // 2
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
            half_move += 1

        winner = gs.winner
        for pos in positions:
            if winner is None:
                pos["value"] = np.float32(0.0)
            elif pos["current_player"] == winner:
                pos["value"] = np.float32(1.0)
            else:
                pos["value"] = np.float32(-1.0)

        all_samples.extend(positions)

        if (game_idx + 1) % 500 == 0:
            print(f"    Expert games: {game_idx + 1}/{num_games} ({len(all_samples)} positions)")

    return all_samples


# =====================================================================
# Phase 2: Fork exploitation data
# =====================================================================

def generate_fork_data(num_positions: int = 5000) -> List[dict]:
    """Generate fork and double-threat positions that exploit Eisenstein's weakness.

    Eisenstein can't see forks because it evaluates moves independently.
    These positions teach the NN to create two-chain setups.
    """
    dataset = []
    attempts = 0
    max_attempts = num_positions * 5

    while len(dataset) < num_positions and attempts < max_attempts:
        attempts += 1
        sample = _gen_one_fork_position()
        if sample is not None:
            dataset.append(sample)

    return dataset


def _gen_one_fork_position() -> Optional[dict]:
    """Generate a single fork/double-threat position."""
    player = random.choice([1, 2])
    opponent = 3 - player

    # Two different axes
    axes = random.sample(HEX_AXES, 2)
    jq, jr = random.randint(-2, 2), random.randint(-2, 2)
    junction = HexCoord(jq, jr)

    # Two chains meeting at the junction
    cl1 = random.choice([3, 4])
    cl2 = random.choice([3, 4])

    # Chains go backward from junction along each axis
    chain1 = [HexCoord(jq - (i + 1) * axes[0].q, jr - (i + 1) * axes[0].r)
              for i in range(cl1)]
    chain2 = [HexCoord(jq - (i + 1) * axes[1].q, jr - (i + 1) * axes[1].r)
              for i in range(cl2)]

    board = Board()
    occupied = set()

    for c in chain1 + chain2:
        if (c.q, c.r) in occupied:
            continue
        board = board.place(c, player)
        occupied.add((c.q, c.r))

    if (jq, jr) in occupied:
        return None

    # Also add the extension beyond the junction on both axes
    # to make the fork devastating (playing junction creates two long chains)
    end_set = {(jq, jr)}

    # Add some opponent noise
    noise = random.randint(4, 10)
    for _ in range(noise):
        for _ in range(50):
            nq = random.randint(jq - 5, jq + 5)
            nr = random.randint(jr - 5, jr + 5)
            if (nq, nr) not in occupied and (nq, nr) not in end_set:
                board = board.place(HexCoord(nq, nr), opponent)
                occupied.add((nq, nr))
                break

    # Player's turn — the correct move is the junction (creates fork)
    correct_moves = [junction]
    value = np.float32(0.85 + 0.05 * max(cl1, cl2))

    gs = GameState(
        board=board, current_player=player,
        moves_remaining=2, win_length=WIN_LENGTH, turn_number=10,
    )

    try:
        features, (cq, cr) = extract_features(gs, grid_size=GRID_SIZE)
    except Exception:
        return None

    policy = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
    half = GRID_SIZE // 2
    valid_count = 0
    for move in correct_moves:
        if move in gs.board.stones:
            continue
        bx, by = axial_to_brick(move.q, move.r, cq, cr, GRID_SIZE)
        row = by + half
        col = bx + half
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            idx = row * GRID_SIZE + col
            policy[idx] = 1.0
            valid_count += 1

    if valid_count == 0:
        return None

    policy /= policy.sum()

    return {
        "features": features.numpy(),
        "policy": policy,
        "value": min(np.float32(value), np.float32(0.99)),
    }


# =====================================================================
# Phase 3: Adversarial refinement (play → learn from mistakes)
# =====================================================================

def generate_adversarial_data(
    model: HexTTTNet, device: str, num_games: int = 200,
) -> Tuple[List[dict], Dict[str, float]]:
    """Play NN vs Eisenstein, extract training data from all games.

    - From NN losses: use Eisenstein's move as the corrective target.
    - From NN wins: reinforce NN's own moves.
    - All positions get value labels from game outcome.

    Returns (dataset, stats).
    """
    model.to(device)
    model.eval()
    eisenstein = EisensteinGreedyAgent(win_length=WIN_LENGTH, zoi_margin=ZOI_MARGIN)

    all_samples = []
    nn_wins = 0
    eis_wins = 0
    draws = 0

    def nn_fn(gs):
        return nn_get_move(model, gs, device)

    for g in range(num_games):
        # Alternate sides
        if g % 2 == 0:
            nn_player = 1
            winner, records = play_game(nn_fn, eisenstein.get_move)
        else:
            nn_player = 2
            winner, records = play_game(eisenstein.get_move, nn_fn)

        if winner == nn_player:
            nn_wins += 1
        elif winner != 0:
            eis_wins += 1
        else:
            draws += 1

        # Extract training positions
        for rec in records:
            gs = rec["game_state"]
            move_played = rec["move_played"]
            cp = rec["current_player"]

            try:
                features, (cq, cr) = extract_features(gs, grid_size=GRID_SIZE)
            except Exception:
                continue

            half = GRID_SIZE // 2

            # Decide the target move:
            if cp == nn_player and winner == nn_player:
                # NN played and won — reinforce NN's move
                target_move = move_played
            elif cp != nn_player:
                # Eisenstein played — always learn from its moves
                target_move = move_played
            else:
                # NN played but lost — ask Eisenstein what it would have done
                target_move = eisenstein.get_move(gs)

            bx, by = axial_to_brick(target_move.q, target_move.r, cq, cr, GRID_SIZE)
            row = by + half
            col = bx + half

            if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
                continue

            policy = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
            policy[row * GRID_SIZE + col] = 1.0

            # Value from current player's perspective
            if winner is None or winner == 0:
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

    stats = {
        "nn_wins": nn_wins, "eis_wins": eis_wins, "draws": draws,
        "win_rate": nn_wins / max(num_games, 1),
    }
    return all_samples, stats


# =====================================================================
# Training function
# =====================================================================

def train_on_data(
    model: HexTTTNet, dataset: List[dict], device: str,
    epochs: int = 10, batch_size: int = 128, lr: float = 2e-4,
    policy_weight: float = 2.0, value_weight: float = 1.0,
    label: str = "",
) -> float:
    """Train model on dataset. Returns final accuracy."""
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 20
    )

    final_acc = 0.0

    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0.0
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
            num_batches += 1

            pred = out["policy"].argmax(dim=1)
            targ = policies.argmax(dim=1)
            correct += (pred == targ).sum().item()
            total += len(batch)

        scheduler.step()
        final_acc = 100.0 * correct / max(total, 1)
        avg_loss = total_loss / max(num_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"    [{label}] Epoch {epoch+1:2d}/{epochs} | "
                f"loss={avg_loss:.4f} | acc={final_acc:.1f}%"
            )

    return final_acc


# =====================================================================
# Main pipeline
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train NN to beat Eisenstein agent")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Starting checkpoint (default: auto-detect)")
    parser.add_argument("--output", type=str, default="checkpoints/beat_eisenstein.pt")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--target-winrate", type=float, default=0.70,
                        help="Stop when NN achieves this win rate vs Eisenstein")
    parser.add_argument("--max-iterations", type=int, default=8,
                        help="Max adversarial refinement iterations")
    parser.add_argument("--expert-games", type=int, default=3000,
                        help="Expert games for phase 1")
    parser.add_argument("--skip-imitate", action="store_true",
                        help="Skip phase 1 (imitation)")
    parser.add_argument("--benchmark-games", type=int, default=40,
                        help="Games per benchmark round")
    parser.add_argument("--adv-games", type=int, default=200,
                        help="Games per adversarial iteration")
    args = parser.parse_args()

    device = args.device
    output_path = PROJECT_ROOT / args.output

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          Beat Eisenstein — Adversarial Trainer           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Device: {device}")
    print(f"  Target win rate: {args.target_winrate:.0%}")

    # ── Load model ────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # Prefer tactical_nn_v1 > tactical_w6 > distilled > best
        for name in ["checkpoints/tactical_nn_v1.pt", "checkpoints/tactical_w6.pt",
                      "checkpoints/distilled_w6_v2.pt", "checkpoints/best.pt"]:
            p = PROJECT_ROOT / name
            if p.exists():
                ckpt_path = p
                break
        else:
            ckpt_path = None

    model = HexTTTNet(grid_size=GRID_SIZE, num_blocks=6, channels=96, in_channels=12)
    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"  Loaded: {ckpt_path.name}")
    else:
        print("  Starting from scratch")

    # ── Initial benchmark ─────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  INITIAL BENCHMARK: NN vs Eisenstein")
    print("━" * 60)
    t0 = time.time()
    stats = benchmark_vs_eisenstein(model, device, num_games=args.benchmark_games)
    print(f"  NN wins: {stats['nn_wins']}/{stats['total']}  "
          f"Eisenstein wins: {stats['eis_wins']}/{stats['total']}  "
          f"Draws: {stats['draws']}")
    print(f"  Win rate: {stats['win_rate']:.1%}  ({time.time()-t0:.1f}s)")
    best_winrate = stats["win_rate"]

    if best_winrate >= args.target_winrate:
        print(f"\n  Already at target! ({best_winrate:.1%} >= {args.target_winrate:.0%})")
        _save_model(model, output_path, best_winrate, "already_met")
        return

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Supervised imitation of Eisenstein
    # ══════════════════════════════════════════════════════════════
    if not args.skip_imitate:
        print("\n" + "━" * 60)
        print("  PHASE 1: Imitate Eisenstein (supervised learning)")
        print("━" * 60)

        t1 = time.time()
        expert_data = generate_expert_data(num_games=args.expert_games)
        print(f"  Generated {len(expert_data)} expert positions ({time.time()-t1:.1f}s)")

        train_on_data(
            model, expert_data, device,
            epochs=20, batch_size=128, lr=3e-4,
            policy_weight=1.5, value_weight=1.0,
            label="Imitate",
        )

        stats = benchmark_vs_eisenstein(model, device, num_games=args.benchmark_games)
        print(f"\n  After imitation: win rate = {stats['win_rate']:.1%} "
              f"(NN {stats['nn_wins']}, Eis {stats['eis_wins']}, Draw {stats['draws']})")
        best_winrate = max(best_winrate, stats["win_rate"])

        if best_winrate >= args.target_winrate:
            print(f"  Target reached after Phase 1!")
            _save_model(model, output_path, best_winrate, "phase1_imitate")
            return

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Fork exploitation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "━" * 60)
    print("  PHASE 2: Fork exploitation training")
    print("━" * 60)

    t2 = time.time()
    fork_data = generate_fork_data(num_positions=5000)
    print(f"  Generated {len(fork_data)} fork positions ({time.time()-t2:.1f}s)")

    train_on_data(
        model, fork_data, device,
        epochs=15, batch_size=128, lr=2e-4,
        policy_weight=2.5, value_weight=0.5,
        label="Forks",
    )

    stats = benchmark_vs_eisenstein(model, device, num_games=args.benchmark_games)
    print(f"\n  After fork training: win rate = {stats['win_rate']:.1%} "
          f"(NN {stats['nn_wins']}, Eis {stats['eis_wins']}, Draw {stats['draws']})")
    best_winrate = max(best_winrate, stats["win_rate"])

    if best_winrate >= args.target_winrate:
        print(f"  Target reached after Phase 2!")
        _save_model(model, output_path, best_winrate, "phase2_forks")
        return

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Adversarial refinement loop
    # ══════════════════════════════════════════════════════════════
    print("\n" + "━" * 60)
    print("  PHASE 3: Adversarial refinement (play → learn → repeat)")
    print("━" * 60)

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n  ── Iteration {iteration}/{args.max_iterations} ──")

        # Play games and collect adversarial data
        t3 = time.time()
        adv_data, play_stats = generate_adversarial_data(
            model, device, num_games=args.adv_games
        )
        print(f"    Played {args.adv_games} games: NN {play_stats['nn_wins']} - "
              f"Eis {play_stats['eis_wins']} - Draw {play_stats['draws']} "
              f"(win rate {play_stats['win_rate']:.1%}, {time.time()-t3:.1f}s)")
        print(f"    Collected {len(adv_data)} training positions")

        if not adv_data:
            print("    No data collected, skipping training")
            continue

        # Mix adversarial data with fresh fork positions for strategic edge
        fork_mix = generate_fork_data(num_positions=1000)
        combined = adv_data + fork_mix
        random.shuffle(combined)

        # Train — fewer epochs since this is iterative refinement
        # Use lower LR to avoid catastrophic forgetting
        lr = max(5e-5, 2e-4 * (0.7 ** (iteration - 1)))
        train_on_data(
            model, combined, device,
            epochs=8, batch_size=128, lr=lr,
            policy_weight=2.0, value_weight=1.0,
            label=f"Adv-{iteration}",
        )

        # Benchmark
        stats = benchmark_vs_eisenstein(model, device, num_games=args.benchmark_games)
        wr = stats["win_rate"]
        print(f"    Benchmark: {wr:.1%} win rate "
              f"(NN {stats['nn_wins']}, Eis {stats['eis_wins']}, Draw {stats['draws']})")

        if wr > best_winrate:
            best_winrate = wr
            _save_model(model, output_path, best_winrate, f"adv_iter{iteration}")
            print(f"    ★ New best: {best_winrate:.1%} — saved!")

        if best_winrate >= args.target_winrate:
            print(f"\n  ✓ TARGET REACHED: {best_winrate:.1%} >= {args.target_winrate:.0%}")
            break
    else:
        print(f"\n  Max iterations reached. Best win rate: {best_winrate:.1%}")

    # ── Final save ────────────────────────────────────────────────
    _save_model(model, output_path, best_winrate, "final")

    # ── Final comprehensive benchmark ─────────────────────────────
    print("\n" + "━" * 60)
    print("  FINAL BENCHMARK (60 games)")
    print("━" * 60)
    final_stats = benchmark_vs_eisenstein(model, device, num_games=60)
    print(f"  NN wins: {final_stats['nn_wins']}/60  "
          f"Eisenstein wins: {final_stats['eis_wins']}/60  "
          f"Draws: {final_stats['draws']}")
    print(f"  FINAL WIN RATE: {final_stats['win_rate']:.1%}")

    print("\n╔══════════════════════════════════════════════════════════╗")
    if final_stats["win_rate"] >= args.target_winrate:
        print("║         NN BEATS EISENSTEIN!                             ║")
    else:
        print("║         Training complete — more iterations may help     ║")
    print(f"║  Win rate: {final_stats['win_rate']:.1%}  (target: {args.target_winrate:.0%})             ║")
    print(f"║  Saved to: {output_path.name:<43s}  ║")
    print("╚══════════════════════════════════════════════════════════╝")


def _save_model(model: HexTTTNet, path: Path, win_rate: float, phase: str):
    """Save checkpoint with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "vs_eisenstein_win_rate": win_rate,
        "training_phase": phase,
    }, path)


if __name__ == "__main__":
    main()
