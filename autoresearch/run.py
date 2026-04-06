#!/usr/bin/env python3
"""Autoresearch: Experiment loop controller.

Runs train.py, extracts the metric, accepts/rejects changes, logs results.
This implements the Karpathy autoresearch hill-climbing loop.

Usage:
    python autoresearch/run.py                    # Run one experiment
    python autoresearch/run.py --loop             # Run indefinitely
    python autoresearch/run.py --loop --max 50    # Run up to 50 experiments
    python autoresearch/run.py --status           # Show experiment history
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

AUTORESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AUTORESEARCH_DIR.parent
TRAIN_SCRIPT = AUTORESEARCH_DIR / "train.py"
LOG_FILE = AUTORESEARCH_DIR / "experiment_log.jsonl"
BEST_MODEL = AUTORESEARCH_DIR / "best_model.pt"

# Time budget: 10 min training + 5 min eval buffer
TIMEOUT_SECONDS = 900  # 15 minutes max per experiment


def get_next_experiment_id():
    """Get the next experiment ID."""
    if not LOG_FILE.exists():
        return 1
    with open(LOG_FILE) as f:
        lines = [l.strip() for l in f if l.strip()]
    return len(lines) + 1


def get_best_metric():
    """Get the current best metric from the log."""
    if not LOG_FILE.exists():
        return 0.0
    best = 0.0
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("accepted", False):
                best = max(best, entry["metric"])
    return best


def run_experiment():
    """Run a single experiment. Returns (metric, stdout)."""
    print(f"\n{'='*60}")
    print(f"Running experiment...")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(AUTORESEARCH_DIR),
        )
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Experiment exceeded time budget")
        return None, "TIMEOUT"

    stdout = result.stdout
    stderr = result.stderr

    # Print output
    print(stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{stderr}")
        print(f"FAILED: Exit code {result.returncode}")
        return None, stdout + "\n" + stderr

    # Extract metric
    match = re.search(r">>> METRIC: ([\d.]+)", stdout)
    if not match:
        print("ERROR: Could not find METRIC in output")
        return None, stdout

    metric = float(match.group(1))
    print(f"\nExtracted metric: {metric:.4f}")

    # Save per-experiment checkpoint (train.py saves to last_experiment.pt)
    last_ckpt = AUTORESEARCH_DIR / "last_experiment.pt"
    if last_ckpt.exists():
        import shutil
        exp_id = get_next_experiment_id()
        ckpt_dir = AUTORESEARCH_DIR.parent / "checkpoints" / "autoresearch"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"exp{exp_id:03d}_{metric:.3f}.pt"
        shutil.copy2(str(last_ckpt), str(ckpt_path))
        print(f"Checkpoint saved: {ckpt_path.name}")

    return metric, stdout


def extract_description():
    """Try to extract experiment description from train.py comments."""
    with open(TRAIN_SCRIPT) as f:
        content = f.read()

    # Look for a comment at the top describing the experiment
    lines = content.split("\n")
    for line in lines:
        if line.strip().startswith("# EXPERIMENT:"):
            return line.strip().replace("# EXPERIMENT:", "").strip()

    return "No description"


def log_result(experiment_id, description, metric, accepted, output_excerpt=""):
    """Log experiment result."""
    entry = {
        "id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
        "metric": metric,
        "accepted": accepted,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def backup_train_py():
    """Save a backup of train.py before experiment."""
    backup = AUTORESEARCH_DIR / "train.py.bak"
    with open(TRAIN_SCRIPT) as f:
        content = f.read()
    with open(backup, "w") as f:
        f.write(content)
    return backup


def restore_train_py():
    """Restore train.py from backup (reject experiment)."""
    backup = AUTORESEARCH_DIR / "train.py.bak"
    if backup.exists():
        with open(backup) as f:
            content = f.read()
        with open(TRAIN_SCRIPT, "w") as f:
            f.write(content)
        print("Reverted train.py to previous version")


def show_status():
    """Show experiment history and current best."""
    if not LOG_FILE.exists():
        print("No experiments run yet.")
        return

    print(f"\n{'='*60}")
    print("EXPERIMENT HISTORY")
    print(f"{'='*60}\n")

    with open(LOG_FILE) as f:
        entries = [json.loads(l.strip()) for l in f if l.strip()]

    best_metric = 0.0
    accepted_count = 0

    for e in entries:
        status = "ACCEPT" if e["accepted"] else "REJECT"
        marker = " *" if e["accepted"] else ""
        if e["accepted"] and e["metric"] is not None:
            if e["metric"] > best_metric:
                best_metric = e["metric"]
                marker = " ** NEW BEST"
            accepted_count += 1
        metric_str = f"{e['metric']:.4f}" if e["metric"] is not None else "FAILED"
        print(f"  #{e['id']:3d} [{status:6s}] metric={metric_str:>8s} — {e['description']}{marker}")

    print(f"\nTotal: {len(entries)} experiments, {accepted_count} accepted")
    print(f"Best metric: {best_metric:.4f}")
    print(f"Target: >0.50 vs OnePly, then >0.30 vs Eisenstein")


def run_loop(max_experiments=None):
    """Run the autoresearch loop."""
    count = 0

    while True:
        if max_experiments and count >= max_experiments:
            print(f"\nReached max experiments ({max_experiments}). Stopping.")
            break

        exp_id = get_next_experiment_id()
        best = get_best_metric()
        print(f"\n{'#'*60}")
        print(f"EXPERIMENT #{exp_id} — Current best: {best:.4f}")
        print(f"{'#'*60}")

        # Backup current train.py
        backup_train_py()

        # Run experiment
        metric, output = run_experiment()
        description = extract_description()

        if metric is None:
            # Failed experiment
            log_result(exp_id, description, None, False)
            restore_train_py()
            print("Experiment FAILED — reverting")
        elif metric > best:
            # Improvement! Keep it.
            log_result(exp_id, description, metric, True)
            print(f"ACCEPTED: {metric:.4f} > {best:.4f} (improvement: +{metric-best:.4f})")
        else:
            # No improvement — revert
            log_result(exp_id, description, metric, False)
            restore_train_py()
            print(f"REJECTED: {metric:.4f} <= {best:.4f} — reverting")

        count += 1

        # Brief cooldown between experiments
        print("\nCooldown (10s)...")
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Autoresearch experiment loop")
    parser.add_argument("--loop", action="store_true", help="Run experiments in a loop")
    parser.add_argument("--max", type=int, default=None, help="Max experiments (with --loop)")
    parser.add_argument("--status", action="store_true", help="Show experiment history")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.loop:
        run_loop(max_experiments=args.max)
    else:
        # Single experiment
        exp_id = get_next_experiment_id()
        best = get_best_metric()

        backup_train_py()
        metric, output = run_experiment()
        description = extract_description()

        if metric is None:
            log_result(exp_id, description, None, False)
            restore_train_py()
            print("Experiment FAILED")
        elif metric > best:
            log_result(exp_id, description, metric, True)
            print(f"ACCEPTED: {metric:.4f}")
        else:
            log_result(exp_id, description, metric, False)
            restore_train_py()
            print(f"REJECTED: {metric:.4f} <= best ({best:.4f})")


if __name__ == "__main__":
    main()
