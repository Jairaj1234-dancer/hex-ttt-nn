"""Feature extraction from GameState to neural network input tensor.

Converts the sparse, infinite-grid game state into a fixed-size
``(NUM_INPUT_PLANES, grid_size, grid_size)`` float32 tensor suitable
for the :class:`~nn.model.HexTTTNet` convolutional backbone.

The window is centred on the stone centroid (rounded to nearest integer)
so that the network always sees a locally-centred view of the board
regardless of absolute coordinate drift.

Feature planes
--------------
===  ===================  =====================================================
 #   Name                 Description
===  ===================  =====================================================
 0   MY_STONES            Current player's stones (binary)
 1   OPP_STONES           Opponent's stones (binary)
 2   MOVES_THIS_TURN      First sub-move of current turn (binary, nonzero only
                          when ``is_first_move_of_turn`` is False)
 3   IS_FIRST_MOVE        Uniform 1.0 if first move of turn, else 0.0
 4   COLOR_TO_PLAY        Uniform 1.0 for player 1, 0.0 for player 2
 5   MOVE_RECENCY         Per-cell recency weight (most recent = 1.0, decay 0.9)
 6   Q_RELATIVE           Normalised q-coordinate in [-1, 1]
 7   R_RELATIVE           Normalised r-coordinate in [-1, 1]
 8   DISTANCE_TO_CENTROID Normalised hex distance to centroid in [0, 1]
 9   MY_THREAT_5          Cells giving current player an open 5-in-a-row
10   OPP_THREAT_5         Cells giving opponent an open 5-in-a-row
11   MY_THREAT_4          Cells extending current player's open-4
===  ===================  =====================================================
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from game.hex_grid import HexCoord, HEX_AXES, hex_distance, axial_to_brick

# ---------------------------------------------------------------------------
# Public constant
# ---------------------------------------------------------------------------

NUM_INPUT_PLANES: int = 12

# Cache for coordinate/distance planes (keyed by grid_size)
_coord_plane_cache: Dict[int, np.ndarray] = {}


def _get_coord_planes(grid_size: int) -> np.ndarray:
    """Return cached (3, grid_size, grid_size) array for Q_REL, R_REL, DIST planes."""
    if grid_size in _coord_plane_cache:
        return _coord_plane_cache[grid_size]

    half = grid_size // 2
    planes = np.zeros((3, grid_size, grid_size), dtype=np.float32)
    max_dist = float(half) if half > 0 else 1.0

    for row in range(grid_size):
        for col in range(grid_size):
            r_rel = row - half
            q_rel = (col - half) - r_rel // 2
            planes[0, row, col] = q_rel / half if half > 0 else 0.0  # Q_RELATIVE
            planes[1, row, col] = r_rel / half if half > 0 else 0.0  # R_RELATIVE
            # DISTANCE_TO_CENTROID (approx — uses grid center, not dynamic centroid)
            dist = abs(q_rel) + abs(r_rel) + abs(-q_rel - r_rel)
            planes[2, row, col] = min((dist / 2) / max_dist, 1.0)

    _coord_plane_cache[grid_size] = planes
    return planes


# ---------------------------------------------------------------------------
# Threat detection
# ---------------------------------------------------------------------------


def compute_threats(
    board_stones: Dict[HexCoord, int],
    player: int,
    center_q: int,
    center_r: int,
    grid_size: int,
    threat_length: int,
) -> np.ndarray:
    """Find empty cells that would complete a line of *threat_length* for *player*.

    Optimized: only checks empty cells adjacent to the player's stones
    (within threat_length-1 steps along each axis), avoiding full grid scan.

    Returns:
        ``(grid_size, grid_size)`` binary float32 array.
    """
    half = grid_size // 2
    result = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Collect candidate empty cells: neighbors of player's stones along axes
    # This is much smaller than grid_size^2 for sparse boards
    candidates = set()
    for coord, pid in board_stones.items():
        if pid != player:
            continue
        for axis in HEX_AXES:
            for direction in (1, -1):
                c = HexCoord(
                    coord.q + direction * axis.q,
                    coord.r + direction * axis.r,
                )
                # Walk up to threat_length-1 steps to find empty cells
                for _ in range(threat_length - 1):
                    if c not in board_stones:
                        candidates.add(c)
                        break
                    if board_stones[c] != player:
                        break
                    c = HexCoord(
                        c.q + direction * axis.q,
                        c.r + direction * axis.r,
                    )

    # Check each candidate
    for coord in candidates:
        # Map to grid position
        r_rel = coord.r - center_r
        q_rel = coord.q - center_q
        row = r_rel + half
        col = q_rel + r_rel // 2 + half
        if not (0 <= row < grid_size and 0 <= col < grid_size):
            continue

        for axis in HEX_AXES:
            count = 1

            c = HexCoord(coord.q + axis.q, coord.r + axis.r)
            while board_stones.get(c) == player:
                count += 1
                c = HexCoord(c.q + axis.q, c.r + axis.r)

            c = HexCoord(coord.q - axis.q, coord.r - axis.r)
            while board_stones.get(c) == player:
                count += 1
                c = HexCoord(c.q - axis.q, c.r - axis.r)

            if count >= threat_length:
                result[row, col] = 1.0
                break

    return result


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    game_state: object,
    grid_size: int = 19,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Convert a :class:`~game.rules.GameState` to a feature tensor.

    Args:
        game_state: a ``GameState`` instance from ``game.rules``.
        grid_size: side length of the square output grid (default 19).

    Returns:
        ``(features, (center_q, center_r))`` where ``features`` is a
        ``(NUM_INPUT_PLANES, grid_size, grid_size)`` float32 tensor and
        ``(center_q, center_r)`` are the axial coordinates of the window
        centre.
    """
    board = game_state.board
    stones = board.stones
    current_player: int = game_state.current_player
    opponent: int = 3 - current_player  # 1 <-> 2

    half = grid_size // 2

    # ------------------------------------------------------------------
    # Window centre = rounded stone centroid
    # ------------------------------------------------------------------
    cq_f, cr_f = board.stone_centroid()
    center_q = int(round(cq_f))
    center_r = int(round(cr_f))

    # ------------------------------------------------------------------
    # Allocate feature planes
    # ------------------------------------------------------------------
    planes = np.zeros((NUM_INPUT_PLANES, grid_size, grid_size), dtype=np.float32)

    # ------------------------------------------------------------------
    # Build a reverse lookup: for each stone, find its grid position
    # ------------------------------------------------------------------
    for coord, player_id in stones.items():
        row, col = axial_to_brick(coord.q, coord.r, center_q, center_r, grid_size)
        # Offset so centre maps to (half, half).
        row_idx = row + half
        col_idx = col + half
        if not (0 <= row_idx < grid_size and 0 <= col_idx < grid_size):
            continue  # outside the window

        if player_id == current_player:
            planes[0, row_idx, col_idx] = 1.0  # MY_STONES
        else:
            planes[1, row_idx, col_idx] = 1.0  # OPP_STONES

    # ------------------------------------------------------------------
    # Plane 2: MOVES_THIS_TURN (first sub-move of the current turn)
    # ------------------------------------------------------------------
    if not game_state.is_first_move_of_turn and len(game_state.move_history) >= 1:
        first_submove = game_state.move_history[-1]
        row, col = axial_to_brick(first_submove.q, first_submove.r, center_q, center_r, grid_size)
        row_idx = row + half
        col_idx = col + half
        if 0 <= row_idx < grid_size and 0 <= col_idx < grid_size:
            planes[2, row_idx, col_idx] = 1.0

    # ------------------------------------------------------------------
    # Plane 3: IS_FIRST_MOVE (uniform)
    # ------------------------------------------------------------------
    if game_state.is_first_move_of_turn:
        planes[3, :, :] = 1.0

    # ------------------------------------------------------------------
    # Plane 4: COLOR_TO_PLAY (uniform)
    # ------------------------------------------------------------------
    if current_player == 1:
        planes[4, :, :] = 1.0

    # ------------------------------------------------------------------
    # Plane 5: MOVE_RECENCY
    # ------------------------------------------------------------------
    if game_state.move_history:
        decay = 0.9
        history = game_state.move_history
        for i, coord in enumerate(reversed(history)):
            weight = decay ** i  # most recent = 1.0, then 0.9, 0.81, ...
            row, col = axial_to_brick(coord.q, coord.r, center_q, center_r, grid_size)
            row_idx = row + half
            col_idx = col + half
            if 0 <= row_idx < grid_size and 0 <= col_idx < grid_size:
                # Only write if not already written (first occurrence = most recent).
                if planes[5, row_idx, col_idx] == 0.0:
                    planes[5, row_idx, col_idx] = weight

    # ------------------------------------------------------------------
    # Planes 6-8: Q_RELATIVE, R_RELATIVE, DISTANCE_TO_CENTROID (cached)
    # ------------------------------------------------------------------
    coord_planes = _get_coord_planes(grid_size)
    planes[6] = coord_planes[0]  # Q_RELATIVE
    planes[7] = coord_planes[1]  # R_RELATIVE
    planes[8] = coord_planes[2]  # DISTANCE_TO_CENTROID

    # ------------------------------------------------------------------
    # Planes 9-11: Threat planes (adapt to game's win_length)
    # ------------------------------------------------------------------
    wl = getattr(game_state, 'win_length', 6)
    planes[9] = compute_threats(stones, current_player, center_q, center_r, grid_size, threat_length=wl)
    planes[10] = compute_threats(stones, opponent, center_q, center_r, grid_size, threat_length=wl)
    planes[11] = compute_threats(stones, current_player, center_q, center_r, grid_size, threat_length=wl - 1)

    return torch.from_numpy(planes), (center_q, center_r)
