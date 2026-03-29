"""Visualization tools for Hex TTT training analysis.

Functions for:
- Rendering board states with policy/value heatmaps using proper hexagonal cells
- Training loss curves
- Elo progression
- Ownership heatmaps
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from game.hex_grid import HexCoord


# ======================================================================
# Hex geometry helpers
# ======================================================================

# Flat-top hex: x = sqrt(3) * q + sqrt(3)/2 * r, y = 3/2 * r
_SQRT3 = math.sqrt(3.0)


def _axial_to_pixel(q: int, r: int, size: float = 1.0) -> Tuple[float, float]:
    """Convert axial hex coordinates to pixel (x, y) for flat-top hexagons.

    Args:
        q: axial q coordinate.
        r: axial r coordinate.
        size: outer radius of each hexagon.

    Returns:
        (x, y) pixel position of the hex centre.
    """
    x = size * (_SQRT3 * q + _SQRT3 / 2.0 * r)
    y = size * (3.0 / 2.0 * r)
    return x, y


def _hex_polygon(cx: float, cy: float, size: float = 1.0) -> patches.RegularPolygon:
    """Create a matplotlib RegularPolygon for a flat-top hexagon.

    Args:
        cx: x-coordinate of hex centre.
        cy: y-coordinate of hex centre.
        size: outer radius.

    Returns:
        A RegularPolygon patch with 6 sides, oriented flat-top (0 degree rotation).
    """
    return patches.RegularPolygon(
        (cx, cy),
        numVertices=6,
        radius=size,
        orientation=math.radians(30),  # flat-top orientation
    )


# ======================================================================
# Board visualization
# ======================================================================

def plot_hex_board(
    game_state: object,
    policy: Optional[Dict[HexCoord, float]] = None,
    value: Optional[float] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    display_radius: int = 8,
) -> plt.Figure:
    """Render a hex board state using matplotlib with hexagonal cells.

    Filled hexagons for placed stones (blue=P1, red=P2). If policy is
    provided, empty cells are colored by policy probability (green heatmap).
    If value is provided, it is shown in the title.

    Args:
        game_state: a GameState instance from game.rules.
        policy: optional mapping from HexCoord to visit probability.
        value: optional value estimate to display in the title.
        title: optional custom title string.
        save_path: if provided, save figure to this path instead of displaying.
        display_radius: hex radius around centroid to display.

    Returns:
        The matplotlib Figure.
    """
    board = game_state.board
    stones = board.stones

    # Determine centre
    if stones:
        cq_f, cr_f = board.stone_centroid()
        center_q = int(round(cq_f))
        center_r = int(round(cr_f))
    else:
        center_q, center_r = 0, 0

    hex_size = 0.55  # visual size of each hex cell
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_aspect("equal")

    # Determine max policy value for normalization
    max_policy = 0.0
    if policy:
        max_policy = max(policy.values()) if policy.values() else 0.0

    # Draw hexagons
    for dq in range(-display_radius, display_radius + 1):
        for dr in range(-display_radius, display_radius + 1):
            # Skip cells too far from centre (keep roughly circular shape)
            ds = -(dq + dr)
            if max(abs(dq), abs(dr), abs(ds)) > display_radius:
                continue

            q = center_q + dq
            r = center_r + dr
            coord = HexCoord(q, r)
            cx, cy = _axial_to_pixel(dq, dr, size=1.0)

            hex_patch = _hex_polygon(cx, cy, size=hex_size)

            if coord in stones:
                player = stones[coord]
                if player == 1:
                    hex_patch.set_facecolor("#3B82F6")  # blue
                    hex_patch.set_edgecolor("#1E40AF")
                else:
                    hex_patch.set_facecolor("#EF4444")  # red
                    hex_patch.set_edgecolor("#991B1B")
                hex_patch.set_linewidth(1.5)
            elif policy and coord in policy and policy[coord] > 1e-6:
                # Color by policy probability (green heatmap)
                intensity = policy[coord] / max_policy if max_policy > 0 else 0.0
                green = (0.2 + 0.8 * intensity, 0.8, 0.2 + 0.4 * (1.0 - intensity))
                hex_patch.set_facecolor((*green, 0.3 + 0.7 * intensity))
                hex_patch.set_edgecolor("#6B7280")
                hex_patch.set_linewidth(0.5)
                # Show probability as text for high-probability moves
                if policy[coord] > 0.05:
                    ax.text(
                        cx, cy, f"{policy[coord]:.0%}",
                        ha="center", va="center", fontsize=6,
                        color="#1F2937", fontweight="bold",
                    )
            else:
                hex_patch.set_facecolor("#F9FAFB")  # light gray
                hex_patch.set_edgecolor("#D1D5DB")
                hex_patch.set_linewidth(0.5)

            ax.add_patch(hex_patch)

            # Coordinate label (small, for reference)
            ax.text(
                cx, cy - hex_size * 0.7, f"{q},{r}",
                ha="center", va="center", fontsize=4,
                color="#9CA3AF", alpha=0.6,
            )

    # Highlight last move
    if game_state.move_history:
        last = game_state.move_history[-1]
        lq = last.q - center_q
        lr = last.r - center_r
        lx, ly = _axial_to_pixel(lq, lr, size=1.0)
        highlight = patches.Circle((lx, ly), radius=hex_size * 0.3, fill=False,
                                   edgecolor="#FBBF24", linewidth=3)
        ax.add_patch(highlight)

    # Build title
    parts = []
    if title:
        parts.append(title)
    else:
        parts.append("Hex TTT Board")
    status = f"Turn {game_state.turn_number}, Player {game_state.current_player}"
    if game_state.is_terminal and game_state.winner is not None:
        status = f"Player {game_state.winner} wins!"
    parts.append(status)
    if value is not None:
        parts.append(f"Value: {value:+.3f}")
    ax.set_title(" | ".join(parts), fontsize=12, fontweight="bold")

    # Legend
    legend_elements = [
        patches.Patch(facecolor="#3B82F6", edgecolor="#1E40AF", label="Player 1"),
        patches.Patch(facecolor="#EF4444", edgecolor="#991B1B", label="Player 2"),
    ]
    if policy:
        legend_elements.append(
            patches.Patch(facecolor="#86EFAC", edgecolor="#6B7280", label="Policy (green)")
        )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.autoscale_view()
    ax.set_axis_off()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig


# ======================================================================
# Training curves
# ======================================================================

def plot_training_curves(log_path: str, save_path: Optional[str] = None) -> plt.Figure:
    """Plot training loss curves from a JSON-lines training log file.

    Expects each line to be a JSON object with keys:
        step, total_loss, value_loss, policy_loss
    Optional keys: ownership_loss, threat_loss

    Args:
        log_path: path to the training_log.jsonl file.
        save_path: if provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    steps: List[int] = []
    total_losses: List[float] = []
    value_losses: List[float] = []
    policy_losses: List[float] = []
    ownership_losses: List[float] = []
    threat_losses: List[float] = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            steps.append(entry["step"])
            total_losses.append(entry["total_loss"])
            value_losses.append(entry["value_loss"])
            policy_losses.append(entry["policy_loss"])
            ownership_losses.append(entry.get("ownership_loss", 0.0))
            threat_losses.append(entry.get("threat_loss", 0.0))

    if not steps:
        raise ValueError(f"No training data found in {log_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")

    # Smoothing helper: exponential moving average
    def ema(values: List[float], alpha: float = 0.01) -> List[float]:
        smoothed = []
        current = values[0]
        for v in values:
            current = alpha * v + (1 - alpha) * current
            smoothed.append(current)
        return smoothed

    # Total loss
    ax = axes[0, 0]
    ax.plot(steps, total_losses, alpha=0.2, color="#6B7280", linewidth=0.5)
    ax.plot(steps, ema(total_losses), color="#3B82F6", linewidth=1.5, label="Total (smoothed)")
    ax.set_title("Total Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Policy loss
    ax = axes[0, 1]
    ax.plot(steps, policy_losses, alpha=0.2, color="#6B7280", linewidth=0.5)
    ax.plot(steps, ema(policy_losses), color="#10B981", linewidth=1.5, label="Policy (smoothed)")
    ax.set_title("Policy Loss (Cross-Entropy)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Value loss
    ax = axes[1, 0]
    ax.plot(steps, value_losses, alpha=0.2, color="#6B7280", linewidth=0.5)
    ax.plot(steps, ema(value_losses), color="#EF4444", linewidth=1.5, label="Value (smoothed)")
    ax.set_title("Value Loss (MSE)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Auxiliary losses
    ax = axes[1, 1]
    if any(v > 0 for v in ownership_losses):
        ax.plot(steps, ownership_losses, alpha=0.2, color="#6B7280", linewidth=0.5)
        ax.plot(steps, ema(ownership_losses), color="#8B5CF6", linewidth=1.5, label="Ownership")
    if any(v > 0 for v in threat_losses):
        ax.plot(steps, threat_losses, alpha=0.2, color="#9CA3AF", linewidth=0.5)
        ax.plot(steps, ema(threat_losses), color="#F59E0B", linewidth=1.5, label="Threat")
    ax.set_title("Auxiliary Losses")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig


# ======================================================================
# Elo progression
# ======================================================================

def plot_elo_progression(elo_log_path: str, save_path: Optional[str] = None) -> plt.Figure:
    """Plot Elo rating over training iterations.

    Expects each line to be a JSON object with keys:
        iteration, elo_estimate
    Optional keys: win_rate, accepted, step

    Args:
        elo_log_path: path to the elo_log.jsonl file.
        save_path: if provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    iterations: List[int] = []
    elos: List[float] = []
    win_rates: List[float] = []
    accepted_flags: List[bool] = []

    with open(elo_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            iterations.append(entry["iteration"])
            elos.append(entry["elo_estimate"])
            win_rates.append(entry.get("win_rate", 0.5))
            accepted_flags.append(entry.get("accepted", False))

    if not iterations:
        raise ValueError(f"No Elo data found in {elo_log_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Training Progress: Elo Rating", fontsize=14, fontweight="bold")

    # Elo progression
    ax1.plot(iterations, elos, color="#3B82F6", linewidth=2, marker="o", markersize=4)
    # Mark accepted vs rejected checkpoints
    for i, (it, elo, accepted) in enumerate(zip(iterations, elos, accepted_flags)):
        color = "#10B981" if accepted else "#EF4444"
        marker = "^" if accepted else "v"
        ax1.plot(it, elo, marker=marker, color=color, markersize=8, zorder=5)
    ax1.set_ylabel("Elo Estimate")
    ax1.set_title("Elo Rating Over Training")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="#9CA3AF", linestyle="--", linewidth=0.5)

    # Win rate per evaluation
    colors = ["#10B981" if a else "#EF4444" for a in accepted_flags]
    ax2.bar(iterations, win_rates, color=colors, alpha=0.7, width=max(1, (max(iterations) - min(iterations)) / len(iterations) * 0.8) if len(iterations) > 1 else 1)
    ax2.axhline(y=0.55, color="#F59E0B", linestyle="--", linewidth=1.5, label="Threshold (55%)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Win Rate")
    ax2.set_title("Candidate Win Rate (green=accepted, red=rejected)")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig


# ======================================================================
# Ownership map
# ======================================================================

def plot_ownership_map(
    game_state: object,
    ownership: np.ndarray,
    save_path: Optional[str] = None,
    display_radius: int = 8,
) -> plt.Figure:
    """Visualize the ownership prediction as a heatmap over the hex board.

    Args:
        game_state: a GameState instance.
        ownership: (3, H, W) array where channels are [my, opponent, empty].
            Values should be probabilities summing to 1 along dim 0.
        save_path: if provided, save figure to this path.
        display_radius: hex radius around centroid to display.

    Returns:
        The matplotlib Figure.
    """
    board = game_state.board
    stones = board.stones

    if stones:
        cq_f, cr_f = board.stone_centroid()
        center_q = int(round(cq_f))
        center_r = int(round(cr_f))
    else:
        center_q, center_r = 0, 0

    _, H, W = ownership.shape
    half = H // 2
    hex_size = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    channel_names = ["Current Player", "Opponent", "Empty"]
    cmaps = ["Blues", "Reds", "Greys"]

    for ch_idx, (ax, name, cmap) in enumerate(zip(axes, channel_names, cmaps)):
        ax.set_aspect("equal")
        ax.set_title(f"Ownership: {name}", fontsize=11, fontweight="bold")

        for dq in range(-display_radius, display_radius + 1):
            for dr in range(-display_radius, display_radius + 1):
                ds = -(dq + dr)
                if max(abs(dq), abs(dr), abs(ds)) > display_radius:
                    continue

                q = center_q + dq
                r = center_r + dr
                coord = HexCoord(q, r)
                cx, cy = _axial_to_pixel(dq, dr, size=1.0)

                # Map axial to grid position for reading ownership values
                r_rel = r - center_r
                q_rel = q - center_q
                row_idx = r_rel + half
                col_idx = q_rel + r_rel // 2 + half

                hex_patch = _hex_polygon(cx, cy, size=hex_size)

                if 0 <= row_idx < H and 0 <= col_idx < W:
                    prob = float(ownership[ch_idx, row_idx, col_idx])
                    colormap = plt.get_cmap(cmap)
                    hex_patch.set_facecolor(colormap(prob))
                    hex_patch.set_edgecolor("#9CA3AF")
                    hex_patch.set_linewidth(0.5)

                    if prob > 0.3:
                        ax.text(
                            cx, cy, f"{prob:.0%}",
                            ha="center", va="center", fontsize=5,
                            color="white" if prob > 0.6 else "black",
                        )
                else:
                    hex_patch.set_facecolor("#F3F4F6")
                    hex_patch.set_edgecolor("#E5E7EB")
                    hex_patch.set_linewidth(0.3)

                ax.add_patch(hex_patch)

                # Mark stones
                if coord in stones:
                    player = stones[coord]
                    marker_color = "#1E40AF" if player == 1 else "#991B1B"
                    stone_marker = plt.Circle((cx, cy), radius=hex_size * 0.25,
                                              facecolor=marker_color, edgecolor="white",
                                              linewidth=1.0, zorder=5)
                    ax.add_patch(stone_marker)

        ax.autoscale_view()
        ax.set_axis_off()

    fig.suptitle(
        f"Ownership Predictions | Turn {game_state.turn_number}, "
        f"Player {game_state.current_player}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig
