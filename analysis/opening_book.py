"""Extract and analyze opening patterns from self-play data.

Analyzes the first N moves of self-play games to identify preferred openings,
response patterns, and strategic tendencies.

Game logs are expected to be stored as JSON-lines files, one per game, with
each line containing at minimum a ``move_history`` key that is a list of
[q, r] coordinate pairs representing the sequence of half-moves played.

Alternative format: each file may be a single JSON object with a
``move_history`` key.

Usage:
    python -m analysis.opening_book --games-dir game_logs/ --depth 6
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_game_histories(game_logs_dir: str) -> List[List[Tuple[int, int]]]:
    """Load move histories from all game log files in a directory.

    Supports two formats:
        1. JSON-lines: each line is a JSON object with ``move_history``.
        2. Single JSON: the file is one JSON object with ``move_history``.

    Each move in the history is expected to be a [q, r] list/tuple.

    Args:
        game_logs_dir: directory containing game log files (.json or .jsonl).

    Returns:
        List of move histories, where each history is a list of (q, r) tuples.
    """
    logs_path = Path(game_logs_dir)
    histories: List[List[Tuple[int, int]]] = []

    if not logs_path.exists():
        raise FileNotFoundError(f"Game logs directory not found: {game_logs_dir}")

    for filepath in sorted(logs_path.iterdir()):
        if filepath.suffix not in (".json", ".jsonl"):
            continue

        try:
            with open(filepath) as f:
                content = f.read().strip()

            if not content:
                continue

            # Try JSON-lines first
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                move_history = entry.get("move_history")
                if move_history and isinstance(move_history, list):
                    history = [(int(m[0]), int(m[1])) for m in move_history if len(m) >= 2]
                    if history:
                        histories.append(history)

        except (IOError, json.JSONDecodeError):
            continue

    return histories


def _normalize_opening(moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Normalize an opening sequence by translating so the first move is at (0, 0).

    This makes openings comparable regardless of where on the infinite board
    they were played.

    Args:
        moves: list of (q, r) coordinate tuples.

    Returns:
        Translated move sequence with the first move at the origin.
    """
    if not moves:
        return moves

    q0, r0 = moves[0]
    return [(q - q0, r - r0) for q, r in moves]


def _moves_to_key(moves: List[Tuple[int, int]]) -> str:
    """Convert a move sequence to a canonical string key.

    Args:
        moves: list of (q, r) coordinate tuples.

    Returns:
        A string representation like "(0,0)>(1,0)>(0,1)".
    """
    return ">".join(f"({q},{r})" for q, r in moves)


def _key_to_moves(key: str) -> List[Tuple[int, int]]:
    """Convert a canonical string key back to a move sequence.

    Args:
        key: string in the format "(q1,r1)>(q2,r2)>...".

    Returns:
        List of (q, r) tuples.
    """
    if not key:
        return []
    moves = []
    for part in key.split(">"):
        part = part.strip("() ")
        if "," in part:
            q_str, r_str = part.split(",")
            moves.append((int(q_str.strip()), int(r_str.strip())))
    return moves


def extract_openings(game_logs_dir: str, depth: int = 6) -> Dict[str, int]:
    """Extract opening sequences up to ``depth`` half-moves from game logs.

    Each game's first ``depth`` moves are normalized (translated so the first
    move is at the origin) and counted. This reveals the most common opening
    patterns regardless of absolute board position.

    Args:
        game_logs_dir: directory containing game log files.
        depth: number of half-moves to consider as the "opening".

    Returns:
        A dict mapping opening sequence strings to their occurrence counts,
        sorted by frequency (most common first).
    """
    histories = _load_game_histories(game_logs_dir)
    counter: Counter = Counter()

    for history in histories:
        if len(history) < 1:
            continue

        # Take up to 'depth' moves and normalize
        opening = history[:depth]
        normalized = _normalize_opening(opening)
        key = _moves_to_key(normalized)
        counter[key] += 1

    # Also count all sub-prefixes to see which partial openings are popular
    prefix_counter: Counter = Counter()
    for history in histories:
        if len(history) < 1:
            continue
        normalized = _normalize_opening(history[:depth])
        # Add all prefixes from length 1 to depth
        for length in range(1, len(normalized) + 1):
            prefix_key = _moves_to_key(normalized[:length])
            prefix_counter[prefix_key] += 1

    # Return the full-depth openings sorted by frequency
    return dict(counter.most_common())


def analyze_responses(game_logs_dir: str) -> Dict[str, Counter]:
    """Analyze the most common responses to common opening moves.

    For each distinct first move (after normalization), counts all second
    moves seen in response. For each distinct first two moves, counts all
    third moves, and so on up to depth 4.

    Args:
        game_logs_dir: directory containing game log files.

    Returns:
        A dict where keys are opening prefixes (as strings) and values are
        Counters mapping the next move (as "(q,r)" string) to its count.
    """
    histories = _load_game_histories(game_logs_dir)
    response_map: Dict[str, Counter] = defaultdict(Counter)

    analysis_depth = 4

    for history in histories:
        if len(history) < 2:
            continue

        normalized = _normalize_opening(history[:analysis_depth + 1])

        # For each prefix of length 1..analysis_depth, record the response
        for prefix_len in range(1, min(analysis_depth, len(normalized))):
            prefix = normalized[:prefix_len]
            next_move = normalized[prefix_len]
            prefix_key = _moves_to_key(prefix)
            move_str = f"({next_move[0]},{next_move[1]})"
            response_map[prefix_key][move_str] += 1

    return dict(response_map)


def save_opening_book(openings: dict, output_path: str) -> None:
    """Save opening book as a JSON file.

    The output contains:
        - ``openings``: the opening sequence counts.
        - ``metadata``: basic statistics about the book.

    Args:
        openings: dict mapping opening strings to counts.
        output_path: file path for the output JSON.
    """
    total_games = sum(openings.values())
    unique_openings = len(openings)

    data = {
        "metadata": {
            "total_games": total_games,
            "unique_openings": unique_openings,
            "format": "opening_key -> count",
        },
        "openings": openings,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Opening book saved to {output_path}")
    print(f"  {total_games} total games, {unique_openings} unique openings")


def print_opening_stats(openings: dict, top_k: int = 20) -> None:
    """Pretty-print the most common openings.

    Args:
        openings: dict mapping opening strings to counts.
        top_k: number of top openings to display.
    """
    if not openings:
        print("No openings found.")
        return

    total = sum(openings.values())
    sorted_openings = sorted(openings.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop {min(top_k, len(sorted_openings))} Opening Sequences")
    print(f"Total games analyzed: {total}")
    print("=" * 70)
    print(f"{'Rank':>4}  {'Count':>6}  {'Freq':>6}  {'Opening Sequence'}")
    print("-" * 70)

    cumulative = 0
    for rank, (key, count) in enumerate(sorted_openings[:top_k], 1):
        freq = count / total * 100
        cumulative += freq
        moves = _key_to_moves(key)

        # Format moves nicely
        move_strs = [f"({q},{r})" for q, r in moves]
        sequence_str = " -> ".join(move_strs)

        print(f"{rank:>4}  {count:>6}  {freq:>5.1f}%  {sequence_str}")

    print("-" * 70)
    print(f"  Top {min(top_k, len(sorted_openings))} cover {cumulative:.1f}% of all games")

    # Additional stats
    if len(sorted_openings) > top_k:
        remaining = len(sorted_openings) - top_k
        remaining_count = sum(c for _, c in sorted_openings[top_k:])
        print(f"  {remaining} more unique openings ({remaining_count} games)")


def print_response_stats(responses: Dict[str, Counter], top_k: int = 5) -> None:
    """Pretty-print the most common responses to common openings.

    Args:
        responses: output from analyze_responses().
        top_k: number of top responses to show per opening prefix.
    """
    if not responses:
        print("No response data found.")
        return

    # Sort prefixes by total response count
    sorted_prefixes = sorted(
        responses.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True,
    )

    print(f"\nOpening Response Analysis")
    print("=" * 60)

    for prefix_key, counter in sorted_prefixes[:10]:
        total = sum(counter.values())
        print(f"\nAfter: {prefix_key}  ({total} games)")
        print(f"  {'Response':>12}  {'Count':>6}  {'Freq':>6}")
        print(f"  {'-' * 30}")

        for move_str, count in counter.most_common(top_k):
            freq = count / total * 100
            print(f"  {move_str:>12}  {count:>6}  {freq:>5.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract and analyze opening patterns from self-play game logs."
    )
    parser.add_argument(
        "--games-dir", type=str, required=True,
        help="Directory containing game log files (.json or .jsonl).",
    )
    parser.add_argument(
        "--depth", type=int, default=6,
        help="Number of half-moves to consider as the opening (default: 6).",
    )
    parser.add_argument(
        "--output", type=str, default="opening_book.json",
        help="Output path for the opening book JSON (default: opening_book.json).",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of top openings to display (default: 20).",
    )
    args = parser.parse_args()

    print(f"Loading game logs from: {args.games_dir}")
    print(f"Opening depth: {args.depth} half-moves")

    # Extract openings
    openings = extract_openings(args.games_dir, depth=args.depth)
    print_opening_stats(openings, top_k=args.top_k)

    # Analyze responses
    responses = analyze_responses(args.games_dir)
    print_response_stats(responses)

    # Save the opening book
    save_opening_book(openings, args.output)
