#!/usr/bin/env python3
"""AlphaZero-style training for Infinite Hexagonal Tic-Tac-Toe.

Usage:
    python train.py --config configs/phase1.yaml
    python train.py --config configs/phase2.yaml --checkpoint checkpoints/latest.pt
    python train.py --config configs/phase3.yaml --checkpoint checkpoints/latest.pt
"""

import argparse
import copy
import json
import logging
import os
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from nn.model import HexTTTNet
from tournament import EisensteinGreedyAgent, GreedyAgent, OnePlyAgent, RandomAgent
from training.evaluator import Evaluator
from training.reanalyze import Reanalyzer
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayWorker
from training.trainer import Trainer

logger = logging.getLogger(__name__)


def detect_device(requested: str) -> str:
    """Resolve device string, handling 'auto' detection."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_network(config: dict) -> HexTTTNet:
    """Instantiate a HexTTTNet from the config's network section."""
    net_cfg = config.get("network", {})
    return HexTTTNet(
        grid_size=net_cfg.get("grid_size", 19),
        num_blocks=net_cfg.get("num_blocks", 8),
        channels=net_cfg.get("channels", 128),
        in_channels=net_cfg.get("in_channels", 12),
    )


def append_log(log_path: str, entry: dict) -> None:
    """Append a JSON-lines entry to the training log file."""
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Hex TTT Neural Network")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/mps/auto)")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    # ---- Setup logging ----
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- Load config ----
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        raise SystemExit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from %s", config_path)

    # ---- Detect device ----
    device = detect_device(args.device)
    logger.info("Using device: %s", device)

    # Inject device into mcts config so MCTS uses the correct device
    if "mcts" not in config:
        config["mcts"] = {}
    config["mcts"]["device"] = device
    # Ensure grid_size is accessible in mcts config
    grid_size = config.get("network", {}).get("grid_size", 19)
    config["mcts"]["grid_size"] = grid_size

    # ---- Create output directories ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = str(log_dir / "training_log.jsonl")
    elo_log_path = str(log_dir / "elo_log.jsonl")

    # ---- Create network ----
    network = create_network(config)
    network.to(device)
    logger.info(
        "Created network: grid_size=%d, blocks=%d, channels=%d",
        config["network"].get("grid_size", 19),
        config["network"].get("num_blocks", 8),
        config["network"].get("channels", 128),
    )

    # ---- Create best network (copy for evaluation) ----
    best_network = create_network(config)
    best_network.load_state_dict(network.state_dict())
    best_network.to(device)

    # ---- Create trainer ----
    trainer = Trainer(network, config, device=device)

    # ---- Create replay buffer ----
    train_cfg = config.get("training", {})
    buffer_capacity = train_cfg.get("replay_buffer_size", 5000)
    # replay_buffer_size in config is in games; estimate positions per game
    # Average hex game ~40-80 positions. Use capacity * estimated avg positions.
    position_capacity = buffer_capacity * 60
    augment = train_cfg.get("augment_d6", True)
    replay_buffer = ReplayBuffer(capacity=position_capacity, augment=augment)

    # ---- Pre-fill replay buffer with bootstrap data (if available) ----
    bootstrap_dataset_path = train_cfg.get("bootstrap_prefill")
    if bootstrap_dataset_path is None:
        # Auto-detect: look for the matching bootstrap dataset
        import os
        for candidate in ["bootstrap_dataset_w4_10k.npy", "bootstrap_dataset.npy"]:
            if os.path.exists(candidate):
                bootstrap_dataset_path = candidate
                break

    if bootstrap_dataset_path and os.path.exists(bootstrap_dataset_path):
        import numpy as _np
        logger.info("Pre-filling replay buffer from %s...", bootstrap_dataset_path)
        bs_data = list(_np.load(bootstrap_dataset_path, allow_pickle=True))
        # Add in chunks as "games" for the buffer's game tracking
        chunk_size = 20
        for i in range(0, len(bs_data), chunk_size):
            chunk = bs_data[i:i + chunk_size]
            replay_buffer.add_game(chunk)
        logger.info("Pre-filled buffer with %d bootstrap positions", len(bs_data))

    # ---- Create self-play worker ----
    self_play_worker = SelfPlayWorker(best_network, config)

    # ---- Create curriculum opponent (if configured) ----
    curriculum_fn = None
    curriculum_ratio = 0.0
    curriculum_ramp_down = None
    curriculum_ladder = []       # list of (name, fn) pairs, weakest first
    curriculum_ladder_idx = 0    # current opponent tier
    curriculum_promote_threshold = train_cfg.get("curriculum_promote_threshold", 0.6)
    curriculum_promote_window = train_cfg.get("curriculum_promote_window", 30)
    from collections import deque
    curriculum_recent = deque(maxlen=curriculum_promote_window)  # sliding window of bools

    curriculum_agent_type = train_cfg.get("curriculum_agent")
    if curriculum_agent_type in ("eisenstein", "ladder"):
        win_length = config.get("game", {}).get("win_length", 6)
        zoi_margin = config.get("mcts", {}).get("zoi_margin", 3)

        # Build ladder: weakest → strongest
        curriculum_ladder = [
            ("Random", RandomAgent(zoi_margin=zoi_margin).get_move),
            ("Greedy", GreedyAgent(zoi_margin=zoi_margin).get_move),
            ("OnePly", OnePlyAgent(win_length=win_length, zoi_margin=zoi_margin).get_move),
            ("Eisenstein", EisensteinGreedyAgent(
                win_length=win_length, zoi_margin=zoi_margin, defensive=True
            ).get_move),
        ]

        # If agent is "eisenstein" (legacy), skip straight to Eisenstein
        if curriculum_agent_type == "eisenstein":
            curriculum_ladder_idx = len(curriculum_ladder) - 1

        curriculum_fn = curriculum_ladder[curriculum_ladder_idx][1]
        curriculum_ratio = train_cfg.get("curriculum_ratio", 0.2)
        curriculum_ramp_down = train_cfg.get("curriculum_ramp_down")
        logger.info(
            "Curriculum ladder enabled: %s, ratio=%.0f%%, promote at %.0f%% over %d games",
            [name for name, _ in curriculum_ladder],
            curriculum_ratio * 100,
            curriculum_promote_threshold * 100,
            curriculum_promote_window,
        )
        logger.info("Starting at tier %d: %s", curriculum_ladder_idx, curriculum_ladder[curriculum_ladder_idx][0])

    # ---- Create evaluator ----
    evaluator = Evaluator(config, device=device)

    # ---- Create reanalyzer (if enabled) ----
    reanalyze_cfg = config.get("reanalysis", {})
    reanalyze_enabled = reanalyze_cfg.get("enabled", False)
    reanalyze_interval = reanalyze_cfg.get("interval", 10)
    reanalyzer = None
    if reanalyze_enabled:
        reanalyzer = Reanalyzer(network, config, device=device)
        logger.info(
            "Reanalysis enabled: interval=%d, batch_size=%d",
            reanalyze_interval,
            reanalyze_cfg.get("batch_size", 64),
        )

    # ---- Extract training parameters ----
    num_iterations = train_cfg.get("num_iterations", 100)
    games_per_iteration = train_cfg.get("games_per_iteration", 500)
    training_steps_per_iteration = train_cfg.get("training_steps_per_iteration", 500)
    batch_size = train_cfg.get("batch_size", 256)

    eval_cfg = config.get("evaluation", {})
    checkpoint_interval = eval_cfg.get("checkpoint_interval", 10)
    eval_games = eval_cfg.get("games", 200)

    # ---- Optionally load checkpoint ----
    start_iteration = 0
    elo_estimate = 0.0

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            extra = trainer.load_checkpoint(str(ckpt_path))
            start_iteration = extra.get("iteration", 0)
            elo_estimate = extra.get("elo_estimate", 0.0)
            # Sync best network with loaded weights
            best_network.load_state_dict(network.state_dict())
            logger.info(
                "Resumed from checkpoint %s at iteration %d (Elo ~%.0f)",
                ckpt_path, start_iteration, elo_estimate,
            )
        else:
            logger.warning("Checkpoint not found: %s -- starting from scratch", ckpt_path)

    # ---- Main training loop ----
    logger.info(
        "Starting training: %d iterations, %d games/iter, %d steps/iter",
        num_iterations, games_per_iteration, training_steps_per_iteration,
    )

    for iteration in range(start_iteration, num_iterations):
        iter_start = time.time()
        logger.info("=" * 60)
        logger.info("Iteration %d / %d", iteration + 1, num_iterations)
        logger.info("=" * 60)

        # ----------------------------------------------------------------
        # Step 1: Self-play -- generate games with current best network
        # ----------------------------------------------------------------
        # Compute effective curriculum ratio (ramp down over time)
        effective_curriculum_ratio = curriculum_ratio
        if curriculum_ramp_down is not None and iteration >= curriculum_ramp_down:
            # Linear ramp from full ratio to 0 over remaining iterations
            remaining_frac = max(0.0, 1.0 - (iteration - curriculum_ramp_down) /
                                 max(1, num_iterations - curriculum_ramp_down))
            effective_curriculum_ratio = curriculum_ratio * remaining_frac

        if curriculum_fn and effective_curriculum_ratio > 0:
            logger.info(
                "Self-play: generating %d games (%.0f%% curriculum)...",
                games_per_iteration, effective_curriculum_ratio * 100,
            )
        else:
            logger.info("Self-play: generating %d games...", games_per_iteration)

        self_play_worker.network = best_network
        # Always use raw policy for curriculum games until value head is calibrated.
        # MCTS with uncalibrated value head actually degrades play quality.
        # Switch to MCTS only after beating the current tier at >30% with raw policy,
        # indicating the value head has learned something useful.
        curriculum_use_mcts = False
        all_games, sp_stats = self_play_worker.play_games(
            games_per_iteration,
            curriculum_fns=curriculum_ladder if curriculum_ladder and effective_curriculum_ratio > 0 else None,
            curriculum_ratio=effective_curriculum_ratio,
            curriculum_use_mcts=curriculum_use_mcts,
            target_tier_idx=curriculum_ladder_idx,
        )

        # ----------------------------------------------------------------
        # Step 2: Add game data to replay buffer + track curriculum progress
        # ----------------------------------------------------------------
        total_positions = 0
        for game_data in all_games:
            replay_buffer.add_game(game_data)
            total_positions += len(game_data)

        logger.info(
            "Added %d positions from %d games. Buffer size: %d",
            total_positions, len(all_games), len(replay_buffer),
        )

        # Track curriculum ladder progress (sliding window)
        if curriculum_ladder and sp_stats["curriculum_total"] > 0:
            # Add individual game results to sliding window
            cw, ct = sp_stats["curriculum_wins"], sp_stats["curriculum_total"]
            for _ in range(cw):
                curriculum_recent.append(True)
            for _ in range(ct - cw):
                curriculum_recent.append(False)

            tier_name = curriculum_ladder[curriculum_ladder_idx][0]
            window_wins = sum(curriculum_recent)
            window_total = len(curriculum_recent)
            tier_wr = window_wins / max(window_total, 1)
            logger.info(
                "Curriculum vs %s: %d/%d wins in last %d games (%.0f%%)",
                tier_name, window_wins, window_total, window_total, tier_wr * 100,
            )

            # Promote to next tier if sliding window win rate exceeds threshold
            if (
                window_total >= curriculum_promote_window
                and tier_wr >= curriculum_promote_threshold
                and curriculum_ladder_idx < len(curriculum_ladder) - 1
            ):
                curriculum_ladder_idx += 1
                curriculum_fn = curriculum_ladder[curriculum_ladder_idx][1]
                new_tier_name = curriculum_ladder[curriculum_ladder_idx][0]
                logger.info(
                    "*** PROMOTED to tier %d: %s (was %.0f%% vs %s) ***",
                    curriculum_ladder_idx, new_tier_name, tier_wr * 100, tier_name,
                )
                curriculum_recent.clear()

        # ----------------------------------------------------------------
        # Step 3: Training -- run gradient steps on replay buffer
        # ----------------------------------------------------------------
        if len(replay_buffer) < batch_size:
            logger.warning(
                "Buffer size (%d) < batch size (%d); skipping training this iteration.",
                len(replay_buffer), batch_size,
            )
            continue

        logger.info("Training: %d steps with batch_size=%d", training_steps_per_iteration, batch_size)
        epoch_losses = {"total": 0.0, "policy": 0.0, "value": 0.0, "ownership": 0.0, "threat": 0.0}

        for step in tqdm(range(training_steps_per_iteration), desc=f"Iter {iteration + 1} training"):
            batch = replay_buffer.sample(batch_size)
            loss_dict = trainer.train_step(batch)

            for key in epoch_losses:
                epoch_losses[key] += loss_dict.get(key, 0.0)

            # Log per-step
            step_entry = {
                "step": trainer.global_step,
                "iteration": iteration + 1,
                "total_loss": loss_dict["total"],
                "value_loss": loss_dict["value"],
                "policy_loss": loss_dict["policy"],
                "ownership_loss": loss_dict.get("ownership", 0.0),
                "threat_loss": loss_dict.get("threat", 0.0),
                "lr": trainer.learning_rate,
            }
            append_log(train_log_path, step_entry)

            # Step 4: Optionally reanalyze
            if (
                reanalyze_enabled
                and reanalyzer is not None
                and trainer.global_step % reanalyze_interval == 0
            ):
                reanalyzer.update_network(network)
                reanalyzer.reanalyze_batch(replay_buffer)

        # Compute average losses for this iteration
        for key in epoch_losses:
            epoch_losses[key] /= training_steps_per_iteration

        logger.info(
            "Iteration %d avg loss: total=%.4f policy=%.4f value=%.4f "
            "ownership=%.4f threat=%.4f",
            iteration + 1,
            epoch_losses["total"],
            epoch_losses["policy"],
            epoch_losses["value"],
            epoch_losses["ownership"],
            epoch_losses["threat"],
        )

        # ----------------------------------------------------------------
        # Step 5: Periodically evaluate candidate vs best
        # ----------------------------------------------------------------
        should_evaluate = (iteration + 1) % checkpoint_interval == 0 or iteration == num_iterations - 1

        if should_evaluate:
            # With high curriculum ratio, self-play gating is unreliable
            # (first-player advantage causes 50/50 splits). Skip gating
            # and always update when curriculum_ratio >= 0.8.
            skip_gating = (effective_curriculum_ratio >= 0.8)

            if skip_gating:
                best_network.load_state_dict(network.state_dict())
                logger.info(
                    "Gating skipped (curriculum_ratio=%.0f%%). Best network updated.",
                    effective_curriculum_ratio * 100,
                )
                win_rate = 0.5
                accepted = True
            else:
                logger.info("Evaluating candidate network vs current best (%d games)...", eval_games)
                win_rate, accepted = evaluator.evaluate(
                    candidate=network,
                    current_best=best_network,
                    num_games=eval_games,
                )

                if accepted:
                    best_network.load_state_dict(network.state_dict())
                    elo_delta = 400.0 * (win_rate - 0.5)  # Approximate Elo gain
                    elo_estimate += elo_delta
                    logger.info(
                        "Candidate ACCEPTED (win rate=%.3f). New best network. Elo ~%.0f (+%.0f)",
                        win_rate, elo_estimate, elo_delta,
                    )
                else:
                    logger.info(
                        "Candidate REJECTED (win rate=%.3f). Keeping current best.",
                        win_rate,
                    )

            elo_entry = {
                "iteration": iteration + 1,
                "step": trainer.global_step,
                "win_rate": win_rate,
                "accepted": accepted,
                "elo_estimate": elo_estimate,
            }
            append_log(elo_log_path, elo_entry)

        # ----------------------------------------------------------------
        # Step 6: Save checkpoint
        # ----------------------------------------------------------------
        if should_evaluate or (iteration + 1) % max(1, checkpoint_interval // 2) == 0:
            ckpt_path = str(output_dir / f"checkpoint_iter{iteration + 1:04d}.pt")
            trainer.save_checkpoint(
                ckpt_path,
                extra={
                    "iteration": iteration + 1,
                    "elo_estimate": elo_estimate,
                    "buffer_size": len(replay_buffer),
                    "buffer_games": replay_buffer.num_games,
                },
            )
            # Also save as latest for easy resumption
            latest_path = str(output_dir / "latest.pt")
            trainer.save_checkpoint(
                latest_path,
                extra={
                    "iteration": iteration + 1,
                    "elo_estimate": elo_estimate,
                    "buffer_size": len(replay_buffer),
                    "buffer_games": replay_buffer.num_games,
                },
            )

            # Save best network separately
            if should_evaluate:
                best_path = str(output_dir / "best.pt")
                torch.save(best_network.state_dict(), best_path)
                logger.info("Saved best network to %s", best_path)

        # ----------------------------------------------------------------
        # Step 7: Log progress
        # ----------------------------------------------------------------
        iter_time = time.time() - iter_start
        logger.info(
            "Iteration %d complete in %.1fs | Step %d | Elo ~%.0f | "
            "Buffer: %d positions (%d games) | LR: %.6f",
            iteration + 1,
            iter_time,
            trainer.global_step,
            elo_estimate,
            len(replay_buffer),
            replay_buffer.num_games_in_buffer,
            trainer.learning_rate,
        )

    # ---- Training complete ----
    logger.info("=" * 60)
    logger.info("Training complete after %d iterations.", num_iterations)
    logger.info("Final Elo estimate: %.0f", elo_estimate)
    logger.info("Total training steps: %d", trainer.global_step)
    logger.info("Checkpoints saved to: %s", output_dir)
    logger.info("Training log: %s", train_log_path)
    logger.info("Elo log: %s", elo_log_path)


if __name__ == "__main__":
    main()
