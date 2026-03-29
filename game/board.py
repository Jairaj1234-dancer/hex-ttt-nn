"""Board state for infinite hexagonal tic-tac-toe.

The board is a sparse dict mapping :class:`HexCoord` to a player id
(``1`` or ``2``).  It is treated as *immutable*: :meth:`Board.place`
returns a **new** ``Board`` rather than mutating in place, which makes
MCTS tree-sharing safe and undo trivial.

A Zobrist hash is maintained incrementally so hash lookups and
transposition table checks are O(1).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from game.hex_grid import HexCoord, HEX_AXES


class Board:
    """Sparse, immutable-ish hex board with incremental Zobrist hashing."""

    __slots__ = ("stones", "_hash")

    def __init__(
        self,
        stones: Dict[HexCoord, int] | None = None,
        _hash: int = 0,
    ) -> None:
        self.stones: Dict[HexCoord, int] = stones if stones is not None else {}
        self._hash: int = _hash

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def place(self, coord: HexCoord, player: int) -> Board:
        """Return a **new** Board with *player*'s stone at *coord*.

        Raises :class:`ValueError` if the cell is already occupied.
        """
        if coord in self.stones:
            raise ValueError(f"Cell {coord} is already occupied")

        # Lazy import to break the circular module dependency at load time.
        from game.zobrist import HASHER  # noqa: F811

        new_stones = self.stones.copy()
        new_stones[coord] = player
        new_hash = HASHER.update_hash(self._hash, coord, player)
        return Board(new_stones, new_hash)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_occupied(self, coord: HexCoord) -> bool:
        return coord in self.stones

    def check_win(self, last_move: HexCoord, win_length: int = 6) -> Optional[int]:
        """Check whether the stone at *last_move* completes a line of
        *win_length* for its player along any of the 3 hex axes.

        Returns the winning player (``1`` or ``2``) or ``None``.
        """
        player = self.stones.get(last_move)
        if player is None:
            return None

        stones = self.stones  # local alias for speed

        for axis in HEX_AXES:
            count = 1  # the stone itself

            # Walk in the positive direction
            c = HexCoord(last_move.q + axis.q, last_move.r + axis.r)
            while stones.get(c) == player:
                count += 1
                if count >= win_length:
                    return player
                c = HexCoord(c.q + axis.q, c.r + axis.r)

            # Walk in the negative direction
            c = HexCoord(last_move.q - axis.q, last_move.r - axis.r)
            while stones.get(c) == player:
                count += 1
                if count >= win_length:
                    return player
                c = HexCoord(c.q - axis.q, c.r - axis.r)

        return None

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Return ``(min_q, min_r, max_q, max_r)``.

        Returns ``(0, 0, 0, 0)`` when the board is empty.
        """
        if not self.stones:
            return (0, 0, 0, 0)

        qs = [c.q for c in self.stones]
        rs = [c.r for c in self.stones]
        return (min(qs), min(rs), max(qs), max(rs))

    def stone_centroid(self) -> Tuple[float, float]:
        """Average ``(q, r)`` of all placed stones.

        Returns ``(0.0, 0.0)`` when the board is empty.
        """
        if not self.stones:
            return (0.0, 0.0)
        n = len(self.stones)
        sq = sum(c.q for c in self.stones)
        sr = sum(c.r for c in self.stones)
        return (sq / n, sr / n)

    # ------------------------------------------------------------------
    # Copy / equality / hashing
    # ------------------------------------------------------------------

    def copy(self) -> Board:
        """Return a deep copy."""
        return Board(self.stones.copy(), self._hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        # Fast path: compare Zobrist hashes first.
        if self._hash != other._hash:
            return False
        return self.stones == other.stones

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"Board({len(self.stones)} stones, hash={self._hash:#018x})"
