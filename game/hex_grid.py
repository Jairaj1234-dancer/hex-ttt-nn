"""Hex grid math utilities using axial coordinates (q, r).

All geometry follows the standard *flat-top* hexagonal axial coordinate
system where the three principal axes are:

    axis 0: direction (1, 0)   -- east
    axis 1: direction (1, -1)  -- north-east
    axis 2: direction (0, -1)  -- north

Each axis and its opposite together cover one of the three lines through a
hex cell; scanning both senses of every axis lets ``Board.check_win`` find
6-in-a-row along any direction.

The module also provides the full D6 (dihedral-6) symmetry group --
6 rotations x 2 reflections = 12 transforms -- for board canonicalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


# ======================================================================
# Core data type
# ======================================================================

@dataclass(frozen=True, slots=True)
class HexCoord:
    """Axial hex coordinate."""
    q: int
    r: int

    def __add__(self, other: HexCoord) -> HexCoord:  # type: ignore[override]
        return HexCoord(self.q + other.q, self.r + other.r)

    def __sub__(self, other: HexCoord) -> HexCoord:
        return HexCoord(self.q - other.q, self.r - other.r)

    def __neg__(self) -> HexCoord:
        return HexCoord(-self.q, -self.r)

    def __mul__(self, scalar: int) -> HexCoord:  # type: ignore[override]
        return HexCoord(self.q * scalar, self.r * scalar)

    def __rmul__(self, scalar: int) -> HexCoord:
        return self.__mul__(scalar)


# ======================================================================
# Direction constants
# ======================================================================

#: The six neighbor direction vectors in axial coordinates (flat-top).
HEX_DIRECTIONS: List[HexCoord] = [
    HexCoord(1, 0),
    HexCoord(1, -1),
    HexCoord(0, -1),
    HexCoord(-1, 0),
    HexCoord(-1, 1),
    HexCoord(0, 1),
]

#: The three axis directions for line checking.  For each axis, the
#: opposite direction is ``-axis``.  Together the three axes cover all
#: six hex directions.
HEX_AXES: List[HexCoord] = HEX_DIRECTIONS[:3]


# ======================================================================
# Neighbor / distance / line helpers
# ======================================================================

def hex_neighbors(coord: HexCoord) -> List[HexCoord]:
    """Return the six neighbors of *coord*."""
    return [coord + d for d in HEX_DIRECTIONS]


def hex_distance(a: HexCoord, b: HexCoord) -> int:
    """Hex (Manhattan-like) distance between two axial coordinates.

    Equivalent to the cube-coordinate L-inf distance:
        max(|dq|, |dr|, |dq + dr|)
    """
    dq = a.q - b.q
    dr = a.r - b.r
    return max(abs(dq), abs(dr), abs(dq + dr))


def hex_line(start: HexCoord, direction: HexCoord, length: int) -> List[HexCoord]:
    """Return *length* cells starting at *start*, stepping by *direction*.

    ``hex_line(c, d, 3)`` -> ``[c, c+d, c+2*d]``
    """
    return [start + direction * i for i in range(length)]


# ======================================================================
# Brick-wall (rectangular display grid) <-> axial conversions
# ======================================================================

def axial_to_brick(
    q: int, r: int,
    center_q: int, center_r: int,
    grid_size: int,
) -> Tuple[int, int]:
    """Convert axial (q, r) to (row, col) in a brick-wall display grid.

    ``center_q, center_r`` is the axial coordinate that maps to the
    centre of the grid.  ``grid_size`` is unused for the mapping math
    but kept in the signature for symmetry with the inverse function.

    Brick-wall layout (even-r offset):
        row = r - center_r
        col = (q - center_q) + (r - center_r) // 2
    """
    dr = r - center_r
    dq = q - center_q
    row = dr
    col = dq + dr // 2
    return (row, col)


def brick_to_axial(
    row: int, col: int,
    center_q: int, center_r: int,
    grid_size: int,
) -> HexCoord:
    """Inverse of :func:`axial_to_brick`."""
    r = row + center_r
    q = col - row // 2 + center_q
    return HexCoord(q, r)


# ======================================================================
# D6 symmetry operations (rotations + reflections)
# ======================================================================

def rotate_60(coord: HexCoord, center: HexCoord) -> HexCoord:
    """Rotate *coord* by 60 degrees clockwise around *center*.

    Relative transform: (q, r) -> (-r, q + r)
    """
    dq = coord.q - center.q
    dr = coord.r - center.r
    new_q = -dr
    new_r = dq + dr
    return HexCoord(new_q + center.q, new_r + center.r)


def rotate_n(coord: HexCoord, center: HexCoord, n: int) -> HexCoord:
    """Rotate *coord* by ``n * 60`` degrees clockwise around *center*."""
    n = n % 6
    result = coord
    for _ in range(n):
        result = rotate_60(result, center)
    return result


def reflect(coord: HexCoord, center: HexCoord) -> HexCoord:
    """Reflect *coord* across the q = r line through *center*.

    Relative transform: (q, r) -> (r, q)
    """
    dq = coord.q - center.q
    dr = coord.r - center.r
    return HexCoord(dr + center.q, dq + center.r)


def all_symmetries(coord: HexCoord, center: HexCoord) -> List[HexCoord]:
    """Return all 12 images of *coord* under the D6 dihedral group centred
    at *center* (6 rotations + 6 rotation-then-reflect)."""
    images: List[HexCoord] = []
    for n in range(6):
        rotated = rotate_n(coord, center, n)
        images.append(rotated)
        images.append(reflect(rotated, center))
    return images
