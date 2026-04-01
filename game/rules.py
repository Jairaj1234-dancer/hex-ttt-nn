"""Game-state and rules management for infinite hexagonal tic-tac-toe.

Rules summary
-------------
* Two players alternate turns on an infinite hex grid.
* Each turn a player places **2 stones**, except Player 1's very first
  turn which is only **1 stone** (the *pie rule* opening balance).
* The first player to form an unbroken line of **6** stones along any
  of the three hex axes wins.
* The board is conceptually infinite; :meth:`GameState.legal_moves`
  constrains search to a zone-of-interest around existing stones.

``GameState`` is treated as immutable: :meth:`apply_move` returns a new
object.
"""

from __future__ import annotations

from typing import List, Optional

from game.board import Board
from game.hex_grid import HexCoord, hex_distance


class GameState:
    """Complete snapshot of a game in progress (or finished)."""

    __slots__ = (
        "board",
        "current_player",
        "moves_remaining",
        "move_history",
        "winner",
        "turn_number",
        "win_length",
    )

    def __init__(
        self,
        board: Board | None = None,
        current_player: int = 1,
        moves_remaining: int = 1,
        move_history: List[HexCoord] | None = None,
        winner: Optional[int] = None,
        turn_number: int = 1,
        win_length: int = 6,
    ) -> None:
        self.board: Board = board if board is not None else Board()
        self.current_player: int = current_player
        self.moves_remaining: int = moves_remaining
        self.move_history: List[HexCoord] = move_history if move_history is not None else []
        self.winner: Optional[int] = winner
        self.turn_number: int = turn_number
        self.win_length: int = win_length

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_first_turn(self) -> bool:
        """``True`` when it is still Player 1's opening (1-stone) turn."""
        return self.turn_number == 1

    @property
    def is_first_move_of_turn(self) -> bool:
        """``True`` when the current sub-move is the first of this turn.

        On the very first turn (1 stone only), ``moves_remaining == 1``
        and this is still the first (and only) move.  On all subsequent
        turns, the first sub-move has ``moves_remaining == 2``.
        """
        if self.is_first_turn:
            return self.moves_remaining == 1
        return self.moves_remaining == 2

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None

    # ------------------------------------------------------------------
    # Move application
    # ------------------------------------------------------------------

    def apply_move(self, coord: HexCoord) -> GameState:
        """Place a stone at *coord* for the current player.

        Returns a **new** ``GameState`` reflecting the updated board,
        player, move budget, turn counter, and possible win.

        Raises :class:`ValueError` if the game is already over or the
        cell is occupied.
        """
        if self.is_terminal:
            raise ValueError("Cannot apply move: game is already over")

        # Board.place raises ValueError if occupied.
        new_board = self.board.place(coord, self.current_player)
        new_history = self.move_history + [coord]

        # Check for a win created by this stone.
        win = new_board.check_win(coord, win_length=self.win_length)

        if self.moves_remaining > 1:
            # Same player still has sub-moves left this turn.
            return GameState(
                board=new_board,
                current_player=self.current_player,
                moves_remaining=self.moves_remaining - 1,
                move_history=new_history,
                winner=win,
                turn_number=self.turn_number,
                win_length=self.win_length,
            )
        else:
            # Turn is over -- switch player, reset budget, bump turn.
            next_player = 2 if self.current_player == 1 else 1
            return GameState(
                board=new_board,
                current_player=next_player,
                moves_remaining=2,
                move_history=new_history,
                winner=win,
                turn_number=self.turn_number + 1,
                win_length=self.win_length,
            )

    # ------------------------------------------------------------------
    # Legal-move generation
    # ------------------------------------------------------------------

    def legal_moves(self, zoi_margin: int = 3) -> List[HexCoord]:
        """Return every empty cell within *zoi_margin* hex-distance of
        any existing stone.

        If the board is empty the canonical opening move ``(0, 0)`` is
        returned.
        """
        if not self.board.stones:
            return [HexCoord(0, 0)]

        min_q, min_r, max_q, max_r = self.board.get_bounding_box()
        min_q -= zoi_margin
        min_r -= zoi_margin
        max_q += zoi_margin
        max_r += zoi_margin

        stones = self.board.stones
        occupied = stones.keys()

        # Pre-collect stone list for distance checks.
        stone_list = list(occupied)

        moves: List[HexCoord] = []
        for q in range(min_q, max_q + 1):
            for r in range(min_r, max_r + 1):
                candidate = HexCoord(q, r)
                if candidate in stones:
                    continue
                # Accept if within zoi_margin of *any* stone.
                for s in stone_list:
                    if hex_distance(candidate, s) <= zoi_margin:
                        moves.append(candidate)
                        break

        return moves

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> GameState:
        """Return a deep copy of this game state."""
        return GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            moves_remaining=self.moves_remaining,
            move_history=list(self.move_history),
            winner=self.winner,
            turn_number=self.turn_number,
            win_length=self.win_length,
        )

    def __repr__(self) -> str:
        status = f"winner={self.winner}" if self.is_terminal else f"P{self.current_player}"
        return (
            f"GameState(turn={self.turn_number}, {status}, "
            f"remaining={self.moves_remaining}, stones={len(self.board.stones)})"
        )
