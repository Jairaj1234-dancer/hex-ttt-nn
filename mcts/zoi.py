"""Zone of Interest computation for MCTS action-space pruning.

The Zone of Interest (ZoI) defines which empty cells the MCTS should
consider as candidate moves.  Rather than treating the entire infinite
grid as the action space, we restrict attention to empty cells within a
configurable hex-distance margin of existing stones.  This keeps the
branching factor manageable while ensuring all tactically relevant moves
are included.
"""

from __future__ import annotations

from typing import Set

import numpy as np

from game.hex_grid import HexCoord, hex_distance, axial_to_brick


def compute_zoi(game_state: object, margin: int = 3) -> Set[HexCoord]:
    """Compute the Zone of Interest -- all empty hex cells within *margin*
    hex-distance of any existing stone.

    If the board is empty, returns cells within *margin* of the origin
    ``(0, 0)``, which is the canonical first-move location.

    Args:
        game_state: a ``GameState`` instance from ``game.rules``.
        margin: maximum hex-distance from any stone to be considered.

    Returns:
        Set of ``HexCoord`` representing valid candidate moves.
    """
    board = game_state.board
    stones = board.stones

    if not stones:
        # Empty board: return all cells within margin of origin.
        origin = HexCoord(0, 0)
        result: Set[HexCoord] = set()
        for q in range(-margin, margin + 1):
            for r in range(-margin, margin + 1):
                candidate = HexCoord(q, r)
                if hex_distance(candidate, origin) <= margin:
                    result.add(candidate)
        return result

    # Compute bounding box expanded by margin.
    min_q, min_r, max_q, max_r = board.get_bounding_box()
    min_q -= margin
    min_r -= margin
    max_q += margin
    max_r += margin

    occupied = stones.keys()
    # Pre-collect stone list for distance checks.
    stone_list = list(occupied)

    result = set()
    for q in range(min_q, max_q + 1):
        for r in range(min_r, max_r + 1):
            candidate = HexCoord(q, r)
            if candidate in stones:
                continue
            # Accept if within margin of any existing stone.
            for s in stone_list:
                if hex_distance(candidate, s) <= margin:
                    result.add(candidate)
                    break

    return result


def compute_zoi_mask(
    game_state: object,
    center_q: int,
    center_r: int,
    grid_size: int,
    margin: int = 3,
) -> np.ndarray:
    """Compute a ``(grid_size, grid_size)`` binary mask for the ZoI.

    The mask has 1 at positions corresponding to valid moves within the
    Zone of Interest, mapped to the brick-wall grid centred at
    ``(center_q, center_r)``.  Positions outside the grid window or that
    are already occupied are set to 0.

    This mask is intended to be reshaped to ``(1, grid_size * grid_size)``
    and passed as ``valid_moves_mask`` to :meth:`HexTTTNet.forward`.

    Args:
        game_state: a ``GameState`` instance from ``game.rules``.
        center_q: axial q of the window centre.
        center_r: axial r of the window centre.
        grid_size: spatial dimension of the output mask.
        margin: ZoI margin passed to :func:`compute_zoi`.

    Returns:
        ``(grid_size, grid_size)`` float32 numpy array with 1.0 for valid
        moves and 0.0 elsewhere.
    """
    half = grid_size // 2
    zoi_cells = compute_zoi(game_state, margin=margin)

    mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    for coord in zoi_cells:
        row, col = axial_to_brick(coord.q, coord.r, center_q, center_r, grid_size)
        row_idx = row + half
        col_idx = col + half
        if 0 <= row_idx < grid_size and 0 <= col_idx < grid_size:
            mask[row_idx, col_idx] = 1.0

    return mask
