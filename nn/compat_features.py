"""Feature extraction matching the hexgo/collaborator's 17-plane format.

Channel layout (from hexgo):
    0     Player 1 stones (binary)
    1     Player 2 stones (binary)
    2     To-move plane (0.0 for P1, 1.0 for P2)
    3-6   Player 1 move history (last 4, most recent = ch 3)
    7-10  Player 2 move history (last 4, most recent = ch 7)
    11-13 Current player chain potential along 3 Eisenstein axes
    14-16 Opponent chain potential along 3 Eisenstein axes

Grid: 18x18, centered on stone centroid.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from game.hex_grid import HexCoord, HEX_AXES, axial_to_brick
from game.rules import GameState

COMPAT_NUM_PLANES = 17
COMPAT_GRID_SIZE = 18


def _chain_potential_per_axis(
    board_stones: Dict[HexCoord, int],
    player: int,
    center_q: int,
    center_r: int,
    grid_size: int,
) -> np.ndarray:
    """Compute per-axis chain potential for a player.

    Returns (3, grid_size, grid_size) where each plane is the chain
    length along one Eisenstein axis if a stone were placed there.
    """
    half = grid_size // 2
    result = np.zeros((3, grid_size, grid_size), dtype=np.float32)

    for row in range(grid_size):
        for col in range(grid_size):
            r_ax = row - half + center_r
            q_ax = (col - half) - (row - half) // 2 + center_q
            coord = HexCoord(q_ax, r_ax)

            if coord in board_stones:
                continue

            for ax_idx, axis in enumerate(HEX_AXES):
                count = 1
                c = HexCoord(coord.q + axis.q, coord.r + axis.r)
                while board_stones.get(c) == player:
                    count += 1
                    c = HexCoord(c.q + axis.q, c.r + axis.r)
                c = HexCoord(coord.q - axis.q, coord.r - axis.r)
                while board_stones.get(c) == player:
                    count += 1
                    c = HexCoord(c.q - axis.q, c.r - axis.r)
                result[ax_idx, row, col] = count / 6.0  # normalize

    return result


def extract_compat_features(
    game_state: GameState,
    grid_size: int = COMPAT_GRID_SIZE,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Extract 17-plane features matching the hexgo/collaborator format.

    Returns:
        (features, (center_q, center_r)) where features is (17, H, W).
    """
    board = game_state.board
    stones = board.stones
    current_player = game_state.current_player
    opponent = 3 - current_player
    half = grid_size // 2

    cq_f, cr_f = board.stone_centroid()
    center_q = int(round(cq_f))
    center_r = int(round(cr_f))

    planes = np.zeros((COMPAT_NUM_PLANES, grid_size, grid_size), dtype=np.float32)

    # Planes 0-1: Stones (absolute, not relative to current player)
    for coord, player_id in stones.items():
        row, col = axial_to_brick(coord.q, coord.r, center_q, center_r, grid_size)
        row_idx = row + half
        col_idx = col + half
        if 0 <= row_idx < grid_size and 0 <= col_idx < grid_size:
            if player_id == 1:
                planes[0, row_idx, col_idx] = 1.0
            else:
                planes[1, row_idx, col_idx] = 1.0

    # Plane 2: To-move (0.0 for P1, 1.0 for P2)
    if current_player == 2:
        planes[2, :, :] = 1.0

    # Planes 3-10: Move history (last 4 per player)
    history = game_state.move_history
    p1_moves = []
    p2_moves = []
    # Reconstruct which player made each move by replaying
    temp_gs = GameState(win_length=game_state.win_length)
    for move in history:
        if temp_gs.current_player == 1:
            p1_moves.append(move)
        else:
            p2_moves.append(move)
        if not temp_gs.is_terminal:
            temp_gs = temp_gs.apply_move(move)

    # P1 history: channels 3-6 (most recent = ch 3)
    for i, move in enumerate(reversed(p1_moves[-4:])):
        row, col = axial_to_brick(move.q, move.r, center_q, center_r, grid_size)
        row_idx = row + half
        col_idx = col + half
        if 0 <= row_idx < grid_size and 0 <= col_idx < grid_size:
            planes[3 + i, row_idx, col_idx] = 1.0

    # P2 history: channels 7-10 (most recent = ch 7)
    for i, move in enumerate(reversed(p2_moves[-4:])):
        row, col = axial_to_brick(move.q, move.r, center_q, center_r, grid_size)
        row_idx = row + half
        col_idx = col + half
        if 0 <= row_idx < grid_size and 0 <= col_idx < grid_size:
            planes[7 + i, row_idx, col_idx] = 1.0

    # Planes 11-13: Current player chain potential (3 axes)
    planes[11:14] = _chain_potential_per_axis(
        stones, current_player, center_q, center_r, grid_size
    )

    # Planes 14-16: Opponent chain potential (3 axes)
    planes[14:17] = _chain_potential_per_axis(
        stones, opponent, center_q, center_r, grid_size
    )

    return torch.from_numpy(planes), (center_q, center_r)
