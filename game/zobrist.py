"""Lazy Zobrist hashing for infinite hex grids.

Generates deterministic 64-bit keys for any (q, r, player) triple using
blake2b with a fixed seed. Keys are cached after first computation so
repeated lookups are O(1) dict hits -- critical for MCTS throughput.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Dict, Tuple

from game.hex_grid import HexCoord


class ZobristHasher:
    """Generates deterministic 64-bit Zobrist keys for any (q, r, player) triple.

    Uses blake2b with a fixed seed for reproducibility.  Caches computed
    values so each unique triple is hashed exactly once.
    """

    __slots__ = ("seed", "_cache")

    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed
        self._cache: Dict[Tuple[int, int, int], int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_key(self, coord: HexCoord, player: int) -> int:
        """Return the 64-bit Zobrist key for *player* placing a stone at *coord*."""
        triple = (coord.q, coord.r, player)
        key = self._cache.get(triple)
        if key is not None:
            return key

        # Deterministic derivation: pack (seed, q, r, player) and hash.
        raw = struct.pack(">iqqi", self.seed, coord.q, coord.r, player)
        digest = hashlib.blake2b(raw, digest_size=8).digest()
        key = struct.unpack(">Q", digest)[0]

        self._cache[triple] = key
        return key

    def hash_board(self, stones: Dict[HexCoord, int]) -> int:
        """Compute the full Zobrist hash of a board from scratch."""
        h: int = 0
        for coord, player in stones.items():
            h ^= self.get_key(coord, player)
        return h

    def update_hash(self, current_hash: int, coord: HexCoord, player: int) -> int:
        """Incrementally update *current_hash* by XOR-ing in a new stone."""
        return current_hash ^ self.get_key(coord, player)


# Module-level singleton used by Board and other hot-path code.
HASHER: ZobristHasher = ZobristHasher()
