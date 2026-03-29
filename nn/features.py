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

    For every empty cell inside the window, check along each of the 3 hex
    axes whether placing a stone there would create a consecutive run of
    at least ``threat_length`` friendly stones (including the hypothetical
    stone itself).

    Args:
        board_stones: mapping of ``HexCoord -> player_id`` (the board's stones).
        player: the player to compute threats for (1 or 2).
        center_q: axial q of the window centre.
        center_r: axial r of the window centre.
        grid_size: spatial dimension of the output array.
        threat_length: required consecutive count (e.g. 5 for open-5 threat).

    Returns:
        ``(grid_size, grid_size)`` binary float32 array.
    """
    half = grid_size // 2
    result = np.zeros((grid_size, grid_size), dtype=np.float32)

    for row in range(grid_size):
        for col in range(grid_size):
            # Brick-wall -> axial (inverse of axial_to_brick with the given centre).
            r_ax = row - half + center_r
            q_ax = (col - half) - (row - half) // 2 + center_q
            coord = HexCoord(q_ax, r_ax)

            # Only consider empty cells.
            if coord in board_stones:
                continue

            # Check each of the 3 axes.
            for axis in HEX_AXES:
                count = 1  # the hypothetical stone itself

                # Walk in the positive direction.
                c = HexCoord(coord.q + axis.q, coord.r + axis.r)
                while board_stones.get(c) == player:
                    count += 1
                    c = HexCoord(c.q + axis.q, c.r + axis.r)

                # Walk in the negative direction.
                c = HexCoord(coord.q - axis.q, coord.r - axis.r)
                while board_stones.get(c) == player:
                    count += 1
                    c = HexCoord(c.q - axis.q, c.r - axis.r)

                if count >= threat_length:
                    result[row, col] = 1.0
                    break  # no need to check other axes for this cell

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
    # Planes 6-7: Q_RELATIVE, R_RELATIVE (normalised coordinates)
    # ------------------------------------------------------------------
    for row in range(grid_size):
        for col in range(grid_size):
            # Brick-wall -> relative axial coords
            # axial_to_brick: row = r - center_r, col = (q - center_q) + (r - center_r) // 2
            # We've offset by +half, so row_offset = row - half, col_offset = col - half
            r_rel = row - half  # = r - center_r
            q_rel = (col - half) - r_rel // 2  # = q - center_q
            planes[6, row, col] = q_rel / half if half > 0 else 0.0
            planes[7, row, col] = r_rel / half if half > 0 else 0.0

    # ------------------------------------------------------------------
    # Plane 8: DISTANCE_TO_CENTROID
    # ------------------------------------------------------------------
    # Centroid in fractional axial coords; we compute hex distance from
    # each cell to the centroid (using rounded centroid as reference since
    # hex_distance works on integers -- the centroid IS the window centre).
    centroid_coord = HexCoord(center_q, center_r)
    max_dist = float(half) if half > 0 else 1.0
    for row in range(grid_size):
        for col in range(grid_size):
            r_rel = row - half
            q_rel = (col - half) - r_rel // 2
            cell_coord = HexCoord(center_q + q_rel, center_r + r_rel)
            dist = hex_distance(cell_coord, centroid_coord)
            planes[8, row, col] = min(dist / max_dist, 1.0)

    # ------------------------------------------------------------------
    # Planes 9-11: Threat planes
    # ------------------------------------------------------------------
    planes[9] = compute_threats(stones, current_player, center_q, center_r, grid_size, threat_length=5)
    planes[10] = compute_threats(stones, opponent, center_q, center_r, grid_size, threat_length=5)
    planes[11] = compute_threats(stones, current_player, center_q, center_r, grid_size, threat_length=4)

    return torch.from_numpy(planes), (center_q, center_r)
