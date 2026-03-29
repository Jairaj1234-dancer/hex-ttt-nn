#!/usr/bin/env python3
"""Interactive replay viewer for Infinite Hexagonal Tic-Tac-Toe games.

Plays back saved game replays step-by-step with an ASCII hex board display.

Usage:
    python replay_viewer.py replay_first.json
    python replay_viewer.py replay_last.json --auto --delay 0.5
    python replay_viewer.py replay_first.json --no-clear

Controls (interactive mode):
    Enter / n  - next move
    p          - previous move
    f          - jump to final position
    r          - reset to start
    g <N>      - go to move N
    q          - quit
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple


# ======================================================================
# Board state tracking
# ======================================================================

def build_board_at_move(moves: List[Tuple[int, int]], up_to: int, rules: dict) -> Dict[Tuple[int, int], int]:
    """Replay moves up to index `up_to` and return {(q,r): player} dict."""
    stones: Dict[Tuple[int, int], int] = {}
    current_player = 1
    moves_remaining = rules.get("first_turn_stones", 1)
    normal_stones = rules.get("normal_turn_stones", 2)

    for i in range(up_to):
        q, r = moves[i]
        stones[(q, r)] = current_player
        moves_remaining -= 1
        if moves_remaining == 0:
            current_player = 3 - current_player
            moves_remaining = normal_stones

    return stones


# ======================================================================
# ASCII hex renderer
# ======================================================================

P1_SYMBOL = "\033[94mX\033[0m"  # Blue X
P2_SYMBOL = "\033[91mO\033[0m"  # Red O
LAST_MOVE_P1 = "\033[94;1;4mX\033[0m"  # Blue bold underline
LAST_MOVE_P2 = "\033[91;1;4mO\033[0m"  # Red bold underline
EMPTY = "\033[90m.\033[0m"  # Gray dot


def render_board(
    stones: Dict[Tuple[int, int], int],
    last_move: Tuple[int, int] | None = None,
    margin: int = 2,
) -> str:
    """Render the hex board as colored ASCII art.

    Uses a brick-wall layout: even rows are flush left, odd rows
    are offset by one space to approximate hex geometry.
    """
    if not stones:
        return "  (empty board)\n"

    qs = [c[0] for c in stones]
    rs = [c[1] for c in stones]
    min_q, max_q = min(qs) - margin, max(qs) + margin
    min_r, max_r = min(rs) - margin, max(rs) + margin

    lines = []

    # Column header
    header = "     "
    for q in range(min_q, max_q + 1):
        header += f"{q:>3}"
    lines.append(header)

    for r in range(min_r, max_r + 1):
        # Offset odd rows for hex effect
        indent = " " if (r % 2 != 0) else ""
        row_str = f" {r:>3} {indent}"

        for q in range(min_q, max_q + 1):
            coord = (q, r)
            if coord in stones:
                player = stones[coord]
                if coord == last_move:
                    sym = LAST_MOVE_P1 if player == 1 else LAST_MOVE_P2
                else:
                    sym = P1_SYMBOL if player == 1 else P2_SYMBOL
            else:
                sym = EMPTY
            row_str += f" {sym} "

        lines.append(row_str)

    return "\n".join(lines) + "\n"


# ======================================================================
# Display helpers
# ======================================================================

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def get_move_info(move_idx: int, moves: List[Tuple[int, int]], rules: dict) -> dict:
    """Get player, turn, and sub-move info for a given move index."""
    current_player = 1
    moves_remaining = rules.get("first_turn_stones", 1)
    normal_stones = rules.get("normal_turn_stones", 2)
    turn = 1

    for i in range(move_idx):
        moves_remaining -= 1
        if moves_remaining == 0:
            current_player = 3 - current_player
            moves_remaining = normal_stones
            turn += 1

    total_this_turn = rules["first_turn_stones"] if turn == 1 else normal_stones
    sub_move = total_this_turn - moves_remaining + 1

    return {
        "player": current_player,
        "turn": turn,
        "sub_move": sub_move,
        "total_sub_moves": total_this_turn,
    }


def display_state(
    move_idx: int,
    total_moves: int,
    moves: List[Tuple[int, int]],
    rules: dict,
    replay: dict,
    use_clear: bool = True,
):
    """Display the board at a given move index."""
    if use_clear:
        clear_screen()

    stones = build_board_at_move(moves, move_idx, rules)
    last_move = tuple(moves[move_idx - 1]) if move_idx > 0 else None

    # Header
    p1 = replay["player1"]
    p2 = replay["player2"]
    print(f"\n  {P1_SYMBOL} {p1} (P1)  vs  {P2_SYMBOL} {p2} (P2)")
    print(f"  Move {move_idx}/{total_moves}")

    if move_idx > 0:
        info = get_move_info(move_idx - 1, moves, rules)
        q, r = moves[move_idx - 1]
        player_name = p1 if info["player"] == 1 else p2
        print(
            f"  Last: {player_name} played ({q}, {r}) "
            f"[turn {info['turn']}, sub-move {info['sub_move']}/{info['total_sub_moves']}]"
        )
    else:
        print("  (start position)")

    print()
    print(render_board(stones, last_move))

    # Show result at final position
    if move_idx == total_moves:
        winner = replay["winner"]
        if winner == 1:
            print(f"  >>> {p1} (P1) WINS! <<<\n")
        elif winner == 2:
            print(f"  >>> {p2} (P2) WINS! <<<\n")
        else:
            print("  >>> DRAW <<<\n")


# ======================================================================
# Interactive mode
# ======================================================================

def interactive_viewer(replay: dict, use_clear: bool = True):
    """Step through a replay interactively."""
    moves = [tuple(m) for m in replay["move_history"]]
    rules = replay.get("rules", {"win_length": 6, "first_turn_stones": 1, "normal_turn_stones": 2})
    total = len(moves)
    idx = 0

    display_state(idx, total, moves, rules, replay, use_clear)

    while True:
        try:
            cmd = input("  [Enter=next, p=prev, f=final, r=reset, g N=goto, q=quit] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("", "n"):
            if idx < total:
                idx += 1
            else:
                print("  (already at end)")
                continue
        elif cmd == "p":
            if idx > 0:
                idx -= 1
            else:
                print("  (already at start)")
                continue
        elif cmd == "f":
            idx = total
        elif cmd == "r":
            idx = 0
        elif cmd.startswith("g"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit():
                target = int(parts[1])
                idx = max(0, min(target, total))
            else:
                print("  Usage: g <move_number>")
                continue
        elif cmd == "q":
            break
        else:
            print("  Unknown command")
            continue

        display_state(idx, total, moves, rules, replay, use_clear)


# ======================================================================
# Auto-play mode
# ======================================================================

def auto_viewer(replay: dict, delay: float = 0.8, use_clear: bool = True):
    """Auto-play through a replay with a delay between moves."""
    moves = [tuple(m) for m in replay["move_history"]]
    rules = replay.get("rules", {"win_length": 6, "first_turn_stones": 1, "normal_turn_stones": 2})
    total = len(moves)

    for idx in range(total + 1):
        display_state(idx, total, moves, rules, replay, use_clear)
        if idx < total:
            time.sleep(delay)

    print("  Replay complete.\n")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Replay viewer for Infinite Hex Tic-Tac-Toe games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replay_viewer.py replay_first.json
  python replay_viewer.py replay_last.json --auto --delay 0.3
  python replay_viewer.py replay_first.json --no-clear
        """,
    )
    parser.add_argument("replay", help="Path to replay JSON file")
    parser.add_argument("--auto", action="store_true", help="Auto-play (no interaction)")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay between moves in auto mode (seconds)")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear screen between frames")

    args = parser.parse_args()

    with open(args.replay) as f:
        replay = json.load(f)

    print(f"\nLoaded replay: {replay['player1']} vs {replay['player2']} ({replay['num_moves']} moves)")

    if args.auto:
        auto_viewer(replay, delay=args.delay, use_clear=not args.no_clear)
    else:
        interactive_viewer(replay, use_clear=not args.no_clear)


if __name__ == "__main__":
    main()
